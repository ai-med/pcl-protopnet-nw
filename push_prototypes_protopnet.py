import os
from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchpanic.datamodule.adni import AdniDataModule, collate_adni
from torchpanic.datamodule.modalities import ModalityType
from torchpanic.models.backbones import ThreeDResNet

from pcl.paths import *
from pcl.builder import *
from pcl.util.utils import get_pcl_encoder_weights

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--adni_fold_idx", type=int)
    parser.add_argument("--arch", type=str, default="3dresnet", choices=["3dresnet", "densenet"])
    parser.add_argument("--pcl_encoder_checkpoint_path", type=str)
    parser.add_argument("--latent_dim", type=int)
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    fold_idx = args.adni_fold_idx
    datamodule = AdniDataModule(modalities=ModalityType.MRI,
                                train_data=os.path.join(ADNI_DATA_PATH, f"{fold_idx}-train.h5"),
                                valid_data=os.path.join(ADNI_DATA_PATH, f"{fold_idx}-valid.h5"),
                                test_data=os.path.join(ADNI_DATA_PATH, f"{fold_idx}-test.h5"),
                                batch_size=8)
    datamodule.setup()
    train_set = datamodule.train_dataset
    train_dataloader = DataLoader(train_set, batch_size=8, num_workers=4, drop_last=False, pin_memory=True, shuffle=False,
                                collate_fn=collate_adni)

    # Load encoder
    print("Loading encoder weights...")
    if args.arch == "densenet":
        model = DenseNetEncoder(dim=args.latent_dim) 
    elif args.arch == "3dresnet":
        model = ThreeDResNet(in_channels=1, n_outputs=args.latent_dim)
    state_dict = get_pcl_encoder_weights(args.pcl_encoder_checkpoint_path, momentum_encoder=True)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    # Load unpushed prototypes
    print("Loading unpushed prototypes...")
    clusters = [int(f.replace("clusters_", "")) for f in os.listdir(args.pcl_encoder_checkpoint_path) if f.startswith("clusters_")]
    cluster_idx = max(clusters)
    cluster_result = torch.load(os.path.join(args.pcl_encoder_checkpoint_path, f"clusters_{cluster_idx}")) # later to be replaced

    # Extract latent features using the momentum encoder on the train set
    print("Extracting latent features using the momentum encoder...")
    latent_feature_matrix = torch.Tensor([]).cuda()
    for train_batch in tqdm(train_dataloader):
        # Extract latent features
        with torch.no_grad():
            train_batch = train_batch[0][ModalityType.MRI].cuda()
            train_batch = train_batch.transpose(-2, -3)
            latent_features = model(train_batch) # (batch_size, 128)
        # Normalize the latent features (because the prototype is normalized)
        latent_features = nn.functional.normalize(latent_features, dim=1)
        # Append the latent features to a matrix
        latent_feature_matrix = torch.cat([latent_feature_matrix, latent_features], dim=0)

    # Push prototypes
    print("Pushing prototypes...")
    for k in range(0, len(cluster_result["centroids"])):
        # Calculate the L2 distance between each prototype
        prototype_sample_dist_matrix = torch.cdist(cluster_result["centroids"][k], latent_feature_matrix, p=2)
        prototype_sample_sim_matrix = torch.mm(cluster_result["centroids"][k], latent_feature_matrix.t())

        # Push prototypes to the nearest latent feature
        prototype_indices_in_train_set = torch.min(prototype_sample_dist_matrix, dim=1).indices
        prototype_indices_in_train_set = [idx.item() for idx in prototype_indices_in_train_set]
        prototype_labels_in_train_set = [train_set[idx][-1] for idx in prototype_indices_in_train_set]

        # Save pushed prototypes
        num_prototypes = len(prototype_indices_in_train_set)
        pushed_prototypes = {
            "latent_feature": latent_feature_matrix[prototype_indices_in_train_set],
            "indices": prototype_indices_in_train_set,
            "labels": prototype_labels_in_train_set,
            "fold": fold_idx,
            "num_prototypes": num_prototypes,
        }
        torch.save(pushed_prototypes, os.path.join(args.pcl_encoder_checkpoint_path, f"pushed_prototypes_{fold_idx}-train_{num_prototypes}prototypes"))
