import torch
import os
from tqdm import tqdm
from argparse import ArgumentParser

from pcl.builder import *
from pcl.paths import *
from pcl.classifier import *
from pcl.util.utils import get_pcl_encoder_weights

if __name__ == "__main__":
    parser = ArgumentParser()
    # eval_dataset
    parser.add_argument("--adni_fold_idx", type=int)
    parser.add_argument("--adni_eval_data_split", type=str, default="test", choices=["valid", "test"])

    # Model
    parser.add_argument("--arch", type=str, default="3dresnet", choices=["3dresnet", "densenet"])
    parser.add_argument("--encoder_checkpoint_path", type=str) # Encoder of fixed or trained head.
    parser.add_argument("--model_checkpoint_path", type=str) # Only relevant when head is trained. This is the path to the run in mlflow.
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--num_prototypes", type=int)
    parser.add_argument("--head_trained", action="store_true")

    args = parser.parse_args()

    # Load encoder
    print("Loading encoder...")
    if args.arch == "densenet":
        encoder = DenseNetEncoder(dim=args.latent_dim)
    elif args.arch == "resnet":
        encoder = ThreeDResNet(in_channels=1, n_outputs=args.latent_dim)
    if not args.head_trained:
        state_dict = get_pcl_encoder_weights(args.encoder_checkpoint_path, momentum_encoder=True)
        encoder.load_state_dict(state_dict)
    encoder = encoder.cuda()
    encoder.eval()

    # Load pushed prototypes
    print("Loading pushed prototypes...")
    pushed_prototypes = torch.load(os.path.join(args.encoder_checkpoint_path,
                                                f"pushed_prototypes_{args.adni_fold_idx}-train_{args.num_prototypes}prototypes"))

    # Load classifier
    print("Loading classifier...")
    if not args.head_trained:
        model = PCLProtoPNet(
            encoder=encoder,
            pushed_prototypes=pushed_prototypes,
            config={
                "num_layers": 1,
                "num_outputs": 3,
                "init_type": "protopnet",
                "activation_function": False,
                "dropout": False,
                "dropout_p": 0,
                "head_bias": False,
                "use_l1_reg": False,
                "fold_idx": args.adni_fold_idx,
                "overfit": False,
                "batch_size": 8,
                "optimizer": "adam"
            }
        )
    else:
        checkpoint_filename = os.listdir(os.path.join(args.model_checkpoint_path, "checkpoints"))[0]
        config = {}
        param_names = os.listdir(os.path.join(args.model_checkpoint_path, "params"))
        for param_name in param_names:
            with open(os.path.join(args.model_checkpoint_path, "params", param_name), "r") as f:
                if param_name not in ["init_type", "experiment_name", "activation_function", "optimizer", "encoder_checkpoint", "encoder_path", "prototype_path"]: # string
                    param_val = eval(f.read())
                else:
                    param_val = f.read()
            config[param_name] = param_val
        model = PCLProtoPNet.load_from_checkpoint(
            checkpoint_path=os.path.join(args.model_checkpoint_path, "checkpoints", checkpoint_filename),
            encoder=encoder,
            pushed_prototypes=pushed_prototypes,
            config=config
        )
    model = model.cuda()
    model.eval()

    # Load data
    print("Loading data...")
    model.prepare_data()
    if args.adni_eval_data_split == "valid":
        eval_dataloader = model.val_dataloader()
    if args.adni_eval_data_split == "test":
        eval_dataloader = model.test_dataloader()

    # Evaluate on eval dataset
    print("Evaluating...")
    total_loss = 0
    total_acc = 0
    for batch_idx, batch in tqdm(enumerate(eval_dataloader)):
        if args.adni_eval_data_split == "valid":
            loss, acc = model._final_validation_step(batch, batch_idx, model)
        elif args.adni_eval_data_split == "test":
            loss, acc = model._final_test_step(batch, batch_idx, model)
        total_loss += loss
        total_acc += acc
        
    avg_loss = total_loss / len(eval_dataloader)
    avg_acc = total_acc / len(eval_dataloader)
    if args.adni_eval_data_split == "valid": 
        bacc = model._get_balanced_accuracy_from_confusion_matrix(model.val_cf).item()
    elif args.adni_eval_data_split == "test":
        bacc = model._get_balanced_accuracy_from_confusion_matrix(model.test_cf).item()

    print(f"Eval loss: {avg_loss}")
    print(f"Eval accuracy (acc): {avg_acc}")
    print(f"Eval balanced accuracy (bAcc): {bacc}")



