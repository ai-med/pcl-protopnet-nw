from bayes_opt import BayesianOptimization
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import random
from argparse import ArgumentParser

from pcl.builder import *
from pcl.paths import *
from pcl.classifier import *
from pcl.util.utils import get_pcl_encoder_weights

seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = ArgumentParser()
    # PCL encoder
    parser.add_argument("--arch", type=str, choices=["3dresnet", "densenet"])
    parser.add_argument("--pcl_encoder_checkpoint_path", type=str)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--num_prototypes", type=int)

    # Dataset
    parser.add_argument("--adni_fold_idx", type=int)
    parser.add_argument("--adni_labels", default="0,1,2")

    # FC head
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_outputs", type=int, default=3)
    parser.add_argument("--head_init_type", type=str, default="protopnet", choices=["protopnet", "random"])
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--head_bias", action="store_true")

    # FC head training
    parser.add_argument("--use_l1_reg", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulate_grad_batches", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--mlflow_exp_name", type=str)

    args = parser.parse_args()
    args.adni_labels = [int(l) for l in args.adni_labels.split(',')]

    # Set up config
    config = vars(args)
    print(f"Config: {config}")

    # Load encoder
    print("Loading encoder...")
    if args.arch == "densenet":
        encoder = DenseNetEncoder(dim=args.latent_dim)
    elif args.arch == "3dresnet":
        encoder = ThreeDResNet(in_channels=1, n_outputs=args.latent_dim)
    state_dict = get_pcl_encoder_weights(args.pcl_encoder_checkpoint_path, momentum_encoder=True)
    encoder.load_state_dict(state_dict)
    encoder = encoder.cuda()
    encoder.eval()

    # Load pushed prototypes
    print("Loading pushed prototypes...")
    pushed_prototypes = torch.load(os.path.join(args.pcl_encoder_checkpoint_path, f"pushed_prototypes_{config['adni_fold_idx']}-train_{args.num_prototypes}prototypes"))

    def model_score(config):
        # Load classifier
        model = PCLProtoPNet(encoder=encoder,
                            pushed_prototypes=pushed_prototypes,
                            config=config)

        # Set logger
        mlf_logger = MLFlowLogger(experiment_name=config["mlflow_exp_name"], tracking_uri=MLFLOW_DIR)
        mlf_logger.log_hyperparams(config)

        # Set callbacks
        callbacks = []
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", dirpath=os.path.join(MLFLOW_DIR, mlf_logger._experiment_id, mlf_logger._run_id, "checkpoints"),
                                            filename="{epoch}-{val_loss:.2f}", save_last=False)
        callbacks.append(checkpoint_callback)
        lr_monitor_callback = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor_callback)

        # Set trainer
        trainer = Trainer(max_epochs=config["num_epochs"],
                          devices=1,
                          accelerator="gpu",
                          accumulate_grad_batches=config["accumulate_grad_batches"],
                          log_every_n_steps=1,
                          check_val_every_n_epoch=1,
                          logger=mlf_logger,
                          callbacks=callbacks)

        # Train model
        trainer.fit(model)

        return model.final_val_bacc

    def optimize_classifier(config):
        def model_fn(lr, weight_decay, l1_lambda):
            config["lr"] = lr
            config["weight_decay"] = weight_decay
            config["l1_lambda"] = l1_lambda
            return model_score(config)

        optimizer = BayesianOptimization(
            f=model_fn,
            pbounds={
                "lr": (1e-5, 1e-2),
                "weight_decay": (1e-5, 1e-2),
                "l1_lambda": (1e-5, 1e-2)
            },
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(n_iter=25, init_points=1)

    print("Running Bayesian optimization...")
    optimize_classifier(config)