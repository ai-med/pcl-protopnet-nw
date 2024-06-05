import pandas as pd
from typing import Union, Sequence, Dict, Any, List
from torch.utils.data import Dataset
from monai.data import NibabelReader
from monai.transforms import Transform, Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, RandFlip, RandAffine
from torchpanic.datamodule.adni import AdniDataset
from torchpanic.datamodule.modalities import ModalityType
import torch

from pcl.paths import *

class AddChannel(Transform):
    """
    A custom Transform to add a channel dimension (as the first dimension) to the input.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        return img.unsqueeze(0)
    
class UkbbMRIDataset(Dataset):
    """
    Wrapper for the UK Biobank (UKBB) dataset, used for PCL pre-training.
    Args:
        split (str): Either 'train' or 'val'. This is important to determine the kind of image transformations to apply and what to return for __getitem__.
        csv_file_path (str): Path to the CSV that contains the image paths.
    """
    def __init__(self, split, csv_file_path="data_split/img.csv"):
        assert split in ["train", "val"]

        self.split = split
        self.img_paths = list(pd.read_csv(csv_file_path)["path"])

        if self.split == "train":
            self.transforms = Compose([
                LoadImage(reader=NibabelReader(), image_only=True),
                EnsureChannelFirst(channel_dim=1),
                AddChannel(),
                ScaleIntensity(minv=0.0, maxv=1.0),
                RandFlip(prob=0.9),
                RandAffine(prob=0.9, rotate_range=(-90, 90), scale_range=(-0.05, 0.05), translate_range=(-10, 10))
            ])
        elif self.split == "val":
            self.transforms = Compose([
                LoadImage(reader=NibabelReader(), image_only=True),
                EnsureChannelFirst(channel_dim=1),
                AddChannel(),
                ScaleIntensity(minv=0.0, maxv=1.0)
            ])
        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        q = self.transforms(self.img_paths[idx])
        if self.split == "val":
            return q, idx, torch.Tensor([])
        elif self.split == "train":
            k = self.transforms(self.img_paths[idx])
            return [q, k], idx, torch.Tensor([])
        
class UkbbMRITabDataset(Dataset):
    """
    Wrapper for the UK Biobank (UKBB) dataset, used for PCL pre-training.
    Difference with UkbbMRIDataset: It also returns tabular features ('labels').
    Args:
        split (str): Either 'train' or 'val'. This is important to determine the kind of image transformations to apply and what to return for __getitem__.
        labels (list): A list of features that should be returned by the Dataset, taken from the CSV file specified in csv_file_path.
        csv_file_path (str): Path to the CSV that contains the image paths and tabular features.
    """
    def __init__(self, split, labels=["Age"], csv_file_path="data_split/img_tab.csv"):
        assert split in ["train", "val"]

        self.split = split
        self.data_df = pd.read_csv(csv_file_path)
        self.labels = labels

        if self.split == "train":
            self.transforms = Compose([
                LoadImage(reader=NibabelReader(), image_only=True),
                EnsureChannelFirst(channel_dim=1),
                AddChannel(),
                ScaleIntensity(minv=0.0, maxv=1.0),
                RandFlip(prob=0.9),
                RandAffine(prob=0.9, rotate_range=(-90, 90), scale_range=(-0.05, 0.05), translate_range=(-10, 10))
            ])
        elif self.split == "val":
            self.transforms = Compose([
                LoadImage(reader=NibabelReader(), image_only=True),
                EnsureChannelFirst(channel_dim=1),
                AddChannel(),
                ScaleIntensity(minv=0.0, maxv=1.0)
            ])
        
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        q = self.transforms(self.data_df["path"].values[idx])
        sample_labels = []
        for l in self.labels:
            sample_labels.append(self.data_df[l].values[idx])
        if self.split == "val":
            return q, idx, torch.Tensor(sample_labels)
        elif self.split == "train":
            k = self.transforms(self.data_df["path"].values[idx])
            return [q, k], idx, torch.Tensor(sample_labels)

class AdniMRIDataset(AdniDataset):
    """
    Wrapper for the ADNI dataset, used for PCL pre-training.
    Inherits from torchpanic's AdniDataset.
    Args:
        split (str): Either 'train' or 'val'. This is important to determine the kind of image transformations to apply and what to return for __getitem__.
        path (str): Path to the .h5 file containing the data.
        modalities (list): List of modalities to output.
        augmentation (dict): Legacy features from AdniDataset, not used here.
        labels (list): A list of labels (diagnosis, not features) that should be considered.
    """
    def __init__(
        self,
        split: str,
        path: str,
        modalities: Union[ModalityType, int, Sequence[str]] = ModalityType.MRI,
        augmentation: Dict[str, Any] = {"p": 0.0},
        labels: List[int] = [0, 1, 2]):
        
        assert split in ["train", "val"]
        super().__init__(path, modalities, augmentation)
        self.split = split
        
        if self.split == "train":
            self.transforms = Compose([
                EnsureChannelFirst(channel_dim=1),
                AddChannel(),
                RandFlip(prob=0.9),
                RandAffine(prob=0.9, rotate_range=(-90, 90), scale_range=(-0.05, 0.05), translate_range=(-10, 10))
            ])
        elif self.split == "val":
            self.transforms = Compose([
                EnsureChannelFirst(channel_dim=1),
                AddChannel()
            ])

        # Filter to only the labels given
        _selected_indices = [i for i in range(len(self._diagnosis)) if self._diagnosis[i] in labels]
        self._diagnosis = [self._diagnosis[idx] for idx in _selected_indices]
        self._data_points[self.modalities] = [self._data_points[self.modalities][idx] for idx in _selected_indices]
    
    def __len__(self):
        return len(self._diagnosis)

    def __getitem__(self, idx):
        q = self.transforms(self._data_points[self.modalities][idx].image.data[0]) # 0 because .data results in shape (1, 113, 137, 113) and we wanna ensure we can move 137 first
        label = self._diagnosis[idx]
        if self.split == "val":
            return q, idx, torch.Tensor([])
        elif self.split == "train":
            k = self.transforms(self._data_points[self.modalities][idx].image.data[0])
            return [q, k], idx, torch.Tensor([])

class AdniMRIDataset_nonPCL(AdniDataset):
    """
    Wrapper for the ADNI dataset, NOT used for PCL pre-training but supervised learning.
    Inherits from torchpanic's AdniDataset.
    Args:
        path (str): Path to the .h5 file containing the data.
        modalities (list): List of modalities to output.
        augmentation (dict): Legacy features from AdniDataset, not used here.
        transforms (torchvision.transforms.Compose): A list of image transformations that should be applied.
        labels (list): A list of labels (diagnosis, not features) that should be considered.
    """
    def __init__(
        self,
        path: str,
        modalities: Union[ModalityType, int, Sequence[str]] = ModalityType.MRI,
        augmentation: Dict[str, Any] = {"p": 0.0},
        transforms = Compose([
            EnsureChannelFirst(channel_dim=1),
            AddChannel()
        ]),
        labels: List[int] = [0, 1, 2],
        return_index: bool = True):
        
        super().__init__(path, modalities, augmentation)
        
        self.transforms = transforms

        # Filter to only the labels given
        self._selected_indices = [i for i in range(len(self._diagnosis)) if self._diagnosis[i] in labels]
        self._diagnosis = [self._diagnosis[idx] for idx in self._selected_indices]
        self._data_points[self.modalities] = [self._data_points[self.modalities][idx] for idx in self._selected_indices]
        self.return_index = return_index
    
    def __len__(self):
        return len(self._diagnosis)

    def __getitem__(self, idx):
        img = self.transforms(self._data_points[self.modalities][idx].image.data[0]) # 0 because .data results in shape (1, 113, 137, 113) and we wanna ensure we can move 137 first
        label = self._diagnosis[idx]
        index = self._selected_indices[idx] # Original index in the original dataset containing all labels
        if self.return_index:
            return img, label, index
        else:
            return img, label