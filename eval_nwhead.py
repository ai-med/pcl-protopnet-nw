from argparse import ArgumentParser
import os
from tqdm import tqdm

from torchpanic.models.backbones import ThreeDResNet
import torch
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix

from pcl.nwhead.nw import NWNet
from pcl.builder import *
from pcl.loader import *
import pcl.util.metric as metric
from pcl.util.utils import get_pcl_encoder_weights

def nw_step(batch, network, mode='random'):
    img, label = batch
    img = img.float().cuda()
    label = label.cuda()
    with torch.set_grad_enabled(False):
        output = network.predict(img, mode)
        acc = metric.acc(output.argmax(-1), label)

    return {'acc': acc*100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'pred': output.argmax(-1), \
            'gt': label}

if __name__ == "__main__":
    parser = ArgumentParser()
    # Eval dataset
    parser.add_argument("--adni_fold_idx", type=int)
    parser.add_argument("--adni_eval_data_split", type=str, default="test", choices=["valid", "test"])

    # Model
    parser.add_argument("--arch", type=str, default="3dresnet", choices=["3dresnet", "densenet"])
    parser.add_argument("--encoder_checkpoint_path", type=str) # Fixed or finetuned encoder
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--encoder_finetuned", action="store_true")

    # Inference mode for NW
    parser.add_argument("--mode", type=str, default="full", choices=["full", "random", "cluster", "knn", "hnsw"])

    args = parser.parse_args()

    # Load train data (support set)
    print("Loading support set...")
    train_dataset = AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-train.h5"), labels=[0,1,2], return_index=False)
    train_dataset.targets = []
    for sample in tqdm(train_dataset):
        train_dataset.targets.append(sample[-1])

    # Load eval data
    print("Loading eval dataset...")
    eval_dataset = AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-{args.adni_eval_data_split}.h5"), labels=[0,1,2], return_index=False)
    eval_dataset.targets = []
    for sample in tqdm(eval_dataset):
        eval_dataset.targets.append(sample[-1])
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=4, drop_last=False, pin_memory=True, shuffle=False)

    # Load featurizer/encoder
    print("Loading encoder...")
    if args.arch == "3dresnet":
        featurizer = ThreeDResNet(in_channels=1, n_outputs=args.latent_dim)
    elif args.arch == "densenet":
        featurizer = DenseNetEncoder(dim=args.latent_dim)
    
    if args.encoder_finetuned:
        exp_name = os.listdir(args.encoder_checkpoint_path)[0]
        checkpoint_filename = [f for f in os.listdir(os.path.join(args.encoder_checkpoint_path, exp_name, "checkpoints")) if f.endswith(".h5") or f.endswith(".pth.tar")][-1]
        checkpoint = torch.load(os.path.join(args.encoder_checkpoint_path, exp_name, "checkpoints", checkpoint_filename))
        state_dict = checkpoint["network_state_dict"]
    else:
        state_dict = get_pcl_encoder_weights(args.encoder_checkpoint_path, momentum_encoder=True)
        featurizer.load_state_dict(state_dict)
    
    # Load NWNet
    print("Loading model...")
    network = NWNet(featurizer, 
                    n_classes=3,
                    support_dataset=train_dataset,
                    feat_dim=args.latent_dim,
                    proj_dim=0,
                    kernel_type="euclidean",
                    n_shot=1,
                    n_way=None,
                    debug_mode=False)

    if args.encoder_finetuned:
        network.load_state_dict(state_dict)
    
    network = network.cuda()
    network.eval()
    network.precompute()

    # Evaluate on the eval dataset
    preds = []
    probs = []
    gts = []
    for i, batch in tqdm(enumerate(eval_dataloader)):
        step_res = nw_step(batch, network, mode=args.mode)
        preds.append(step_res['pred'])
        probs.append(step_res['prob'])
        gts.append(step_res['gt'])

    # Calculate bAcc
    cf = ConfusionMatrix(task="multiclass", num_classes=3).cuda()
    cf.update(torch.cat(preds, dim=0).squeeze(), torch.cat(gts, dim=0).squeeze())
    cmat = cf.compute()
    per_class = cmat.diag() / cmat.sum(dim=1)
    per_class = per_class[~torch.isnan(per_class)]
    bacc = per_class.mean()

    print(f"Eval balanced accuracy (bAcc): {bacc}")
    print(f"Eval confusion matrix: {cmat}")
    
