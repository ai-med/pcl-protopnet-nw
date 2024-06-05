import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchmetrics import ConfusionMatrix
import numpy as np
import lightning as L
import monai

import pcl
import pcl.loader
from pcl.loader import *
from pcl.builder import *
from pcl.paths import *
from pcl.resnet import *
from pcl.nwhead.nw import NWNet
from torchpanic.datamodule.adni import AdniDataModule, collate_adni
from torchpanic.datamodule.modalities import ModalityType
from torchpanic.models.backbones import ThreeDResNet
import torchvision.transforms as transforms

loss_fn = nn.CrossEntropyLoss()

class PCLProtoPNet(L.LightningModule):
    def __init__(self, encoder, pushed_prototypes, config):
        super(PCLProtoPNet, self).__init__()
        self.encoder = encoder
        self.prototypes = pushed_prototypes
        self.num_layers = int(config["num_layers"])
        self.num_outputs = config["num_outputs"]
        self.head_init_type = config["head_init_type"]
        self.activation_function = config["activation_function"]
        self.head_bias = config["head_bias"]
        self.use_l1_reg = config["use_l1_reg"]
        self.num_prototypes = self.prototypes["latent_feature"].shape[0]
        self.overfit = config["overfit"]
        if "adni_labels" in config.keys():
            self.adni_labels = config["adni_labels"]
        else:
            self.adni_labels = [0, 1, 2]

        layers = []
        for i in range(0, self.num_layers):
            if i == self.num_layers - 1:
                layers.append(nn.Linear(self.num_prototypes, self.num_outputs, bias=self.head_bias))
            else:
                layers.append(nn.Linear(self.num_prototypes, self.num_prototypes, bias=self.head_bias))
                if self.activation_function == "relu":
                    layers.append(nn.ReLU())
        
        if len(layers) > 1:
            self.classification_head = nn.Sequential(*layers)
        else:
            self.classification_head = layers[0]

        # Initialize the head's weight
        if self.head_init_type == "protopnet":
            self._head_weight_init()

        # Initialize the L1 mask
        if self.use_l1_reg:
            prototype_labels = torch.Tensor(self.prototypes["labels"]).to(torch.int64) # Index tensor
            self.l1_mask = F.one_hot(prototype_labels, num_classes=self.num_outputs).t().cuda()

        # Turn off gradient computation for the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Set config
        self.config = config

        # Set confusion matrix
        self.train_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)
        self.val_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)
        self.test_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)

    def prepare_data(self):
        if self.adni_labels == [0, 1, 2]:
            datamodule = AdniDataModule(modalities=ModalityType.MRI, train_data=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-train.h5"),
                                        valid_data=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-valid.h5"),
                                        test_data=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-test.h5"),
                                        batch_size=8)
            datamodule.setup(stage="fit")
            self.train_set = datamodule.train_dataset
            self.val_set = datamodule.eval_dataset
            datamodule.setup(stage="test")
            self.test_set = datamodule.test_dataset
        else:
            self.train_set = pcl.loader.AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-train.h5"),
                                                            labels=self.adni_labels)
            self.val_set = pcl.loader.AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-valid.h5"),
                                                            labels=self.adni_labels)
            self.test_set = pcl.loader.AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-test.h5"),
                                                            labels=self.adni_labels)

    def forward(self, x):
        latent_feature = nn.functional.normalize(self.encoder(x), dim=1) # (B, latent_dim)
        sim_scores = torch.mm(latent_feature, self.prototypes["latent_feature"].t()) # (B, latent_dim) * (latent_dim, num_prototypes)
        output = F.softmax(self.classification_head(sim_scores), dim=1)
        return output

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

    def on_train_end(self):
        best_model = PCLProtoPNet.load_from_checkpoint(checkpoint_path=self.trainer.checkpoint_callback.best_model_path,
                                                              encoder=self.encoder, pushed_prototypes=self.prototypes, config=self.config)
        best_model.eval()
        val_dataloader = self.val_dataloader()
        total_val_loss = 0
        total_val_acc = 0
        for batch_idx, val_batch in enumerate(val_dataloader):
            val_loss, val_acc = self._final_validation_step(val_batch, batch_idx, best_model)
            total_val_loss += val_loss
            total_val_acc += val_acc
            
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_acc = total_val_acc / len(val_dataloader)
        val_bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cf).item()
            
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_loss", value=avg_val_loss)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_acc", value=avg_val_acc)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_bacc", value=val_bacc)

        print("Final validation loss:", avg_val_loss)
        print("Final validation accuracy:", avg_val_acc)
        print("Final validation balanced accuracy:", val_bacc)

        self.final_val_loss = avg_val_loss
        self.final_val_acc = avg_val_acc
        self.final_val_bacc = val_bacc

        test_dataloader = self.test_dataloader()
        total_test_loss = 0
        total_test_acc = 0
        for batch_idx, test_batch in enumerate(test_dataloader):
            test_loss, test_acc = self._final_test_step(test_batch, batch_idx, best_model)
            total_test_loss += test_loss
            total_test_acc += test_acc
            
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_acc = total_test_acc / len(test_dataloader)
        test_bacc = self._get_balanced_accuracy_from_confusion_matrix(self.test_cf).item()
            
        self.logger.experiment.log_metric(self.logger._run_id, key="final_test_loss", value=avg_test_loss)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_test_acc", value=avg_test_acc)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_test_bacc", value=test_bacc)

        print("Final test loss:", avg_test_loss)
        print("Final test accuracy:", avg_test_acc)
        print("Final test balanced accuracy:", test_bacc)

        self.final_test_loss = avg_test_loss
        self.final_test_acc = avg_test_acc
        self.final_test_bacc = test_bacc
        
    def training_step(self, batch, batch_idx):
        self.encoder.eval()
        if self.adni_labels == [0, 1, 2]:
            images = batch[0][ModalityType.MRI]
            images = images.transpose(-2, -3)
            target = batch[-1]
        else:
            images = batch[0]
            target = batch[1]
        pred = self(images)
        loss = loss_fn(pred, target)
        if self.use_l1_reg: # Assumption num_layers == 1
            l1 = (self.classification_head.weight * (1 - self.l1_mask)).norm(p=1)
            loss += self.config["l1_lambda"] * l1
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.train_cf.update(pred_labels, target)
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.train_cf)
        self.train_cf.reset()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_bacc", bacc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.encoder.eval()
        if self.adni_labels == [0, 1, 2]:
            images = batch[0][ModalityType.MRI]
            images = images.transpose(-2, -3)
            target = batch[-1]
        else:
            images = batch[0]
            target = batch[1]
        pred = self(images)
        loss = loss_fn(pred, target)
        if self.use_l1_reg: # Assumption num_layers == 1
            l1 = (self.classification_head.weight * (1 - self.l1_mask)).norm(p=1)
            loss += self.config["l1_lambda"] * l1
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.val_cf.update(pred_labels, target)
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cf)
        self.val_cf.reset()
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log("val_bacc", bacc, on_step=True, on_epoch=True)
        return loss

    def _final_validation_step(self, batch, batch_idx, best_model):
        best_model.encoder.eval()
        if self.adni_labels == [0, 1, 2]:
            images = batch[0][ModalityType.MRI]
            images = images.transpose(-2, -3).cuda()
            target = batch[-1].cuda()
        else:
            images = batch[0].cuda()
            target = batch[1].cuda()
        with torch.no_grad():
            pred = best_model(images)
        loss = loss_fn(pred, target)
        if self.use_l1_reg: # Assumption num_layers == 1
            l1 = (self.classification_head.weight * (1 - self.l1_mask.cuda())).norm(p=1)
            loss += self.config["l1_lambda"] * l1
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.val_cf.update(pred_labels, target)
        return loss.item(), acc
    
    def _final_test_step(self, batch, batch_idx, best_model):
        best_model.encoder.eval()
        if self.adni_labels == [0, 1, 2]:
            images = batch[0][ModalityType.MRI]
            images = images.transpose(-2, -3).cuda()
            target = batch[-1].cuda()
        else:
            images = batch[0].cuda()
            target = batch[1].cuda()
        with torch.no_grad():
            pred = best_model(images)
        loss = loss_fn(pred, target)
        if self.use_l1_reg: # Assumption num_layers == 1
            l1 = (self.classification_head.weight * (1 - self.l1_mask.cuda())).norm(p=1)
            loss += self.config["l1_lambda"] * l1
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.test_cf.update(pred_labels, target)
        return loss.item(), acc
        
    def train_dataloader(self):
        if self.overfit:
            overfit_set = Subset(self.train_set, [0])
            train_dataloader = DataLoader(overfit_set, batch_size=1,
                                          num_workers=4, drop_last=False, pin_memory=True,
                                          shuffle=True, collate_fn=collate_adni if self.adni_labels == [0, 1, 2] else None)
        else:
            train_dataloader = DataLoader(self.train_set, batch_size=self.config["batch_size"],
                                        num_workers=4, drop_last=False, pin_memory=True,
                                        shuffle=True, collate_fn=collate_adni if self.adni_labels == [0, 1, 2] else None)
        return train_dataloader

    def val_dataloader(self):
        if self.overfit:
            overfit_set = Subset(self.train_set, [0])
            val_dataloader = DataLoader(overfit_set, batch_size=1,
                                        num_workers=4, drop_last=False, pin_memory=True,
                                        collate_fn=collate_adni if self.adni_labels == [0, 1, 2] else None)
        else:
            val_dataloader = DataLoader(self.val_set, batch_size=self.config["batch_size"],
                                        num_workers=2, pin_memory=True, collate_fn=collate_adni if self.adni_labels == [0, 1, 2] else None)
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_set, batch_size=self.config["batch_size"],
                                     num_workers=2, pin_memory=True, collate_fn=collate_adni if self.adni_labels == [0, 1, 2] else None)
        return test_dataloader

    def configure_optimizers(self):
        if self.config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config["lr"])
        return optimizer
        
    def _head_weight_init(self):
        weight_matrix = torch.Tensor([])
        for label in self.prototypes["labels"]:
            weight = torch.Tensor([[-0.5 for i in range(self.num_outputs)]])
            weight[0][label] = 1
            weight_matrix = torch.cat([weight_matrix, weight], dim=0)
        self.classification_head.weight.data = weight_matrix.t()
    
    def _get_balanced_accuracy_from_confusion_matrix(self, confusion_matrix: ConfusionMatrix):
        # Confusion matrix whose i-th row and j-th column entry indicates
        # the number of samples with true label being i-th class and
        # predicted label being j-th class.
        cmat = confusion_matrix.compute()
        per_class = cmat.diag() / cmat.sum(dim=1)
        per_class = per_class[~torch.isnan(per_class)]  # remove classes that are not present in this dataset
        return per_class.mean()

class DenseNetClassifier(L.LightningModule):
    def __init__(self, config, num_outputs=3):
        super(DenseNetClassifier, self).__init__()
        self.model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=3)
        self.num_outputs = num_outputs

        # Set config
        self.config = config

        # Set confusion matrix
        self.train_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)
        self.val_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)

    def prepare_data(self):
        datamodule = AdniDataModule(modalities=ModalityType.MRI, train_data=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-train.h5"),
                            valid_data=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-valid.h5"),
                            test_data=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-test.h5"),
                            batch_size=8)
        datamodule.setup()
        self.train_set = datamodule.train_dataset
        self.val_set = datamodule.eval_dataset

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

    def on_train_end(self):
        best_model = DenseNetClassifier.load_from_checkpoint(checkpoint_path=self.trainer.checkpoint_callback.best_model_path, config=self.config)
        best_model.eval()
        val_dataloader = self.val_dataloader()
        total_val_loss = 0
        total_val_acc = 0
        for batch_idx, val_batch in enumerate(val_dataloader):
            val_loss, val_acc = self._final_validation_step(val_batch, batch_idx, best_model)
            total_val_loss += val_loss
            total_val_acc += val_acc
            
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_acc = total_val_acc / len(val_dataloader)
        val_bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cf).item()
            
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_loss", value=avg_val_loss)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_acc", value=avg_val_acc)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_bacc", value=val_bacc)

        print("Final validation loss:", avg_val_loss)
        print("Final validation accuracy:", avg_val_acc)
        print("Final validation balanced accuracy:", val_bacc)

        self.final_val_loss = avg_val_loss
        self.final_val_acc = avg_val_acc
        self.final_val_bacc = val_bacc
        
    def training_step(self, batch, batch_idx):
        images = batch[0][ModalityType.MRI]
        images = images.transpose(-2, -3)
        pred = self.model(images)
        target = batch[-1]
        loss = loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.train_cf.update(pred_labels, target)
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.train_cf)
        self.train_cf.reset()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_bacc", bacc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch[0][ModalityType.MRI]
        images = images.transpose(-2, -3)
        pred = self.model(images)
        target = batch[-1]
        loss = loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.val_cf.update(pred_labels, target)
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cf)
        self.val_cf.reset()
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log("val_bacc", bacc, on_step=True, on_epoch=True)
        return loss

    def _final_validation_step(self, batch, batch_idx, best_model):
        images = batch[0][ModalityType.MRI].cuda()
        images = images.transpose(-2, -3)
        with torch.no_grad():
            pred = best_model.model(images)
        target = batch[-1].cuda()
        loss = loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.val_cf.update(pred_labels, target)
        return loss.item(), acc
        
    def train_dataloader(self):
        if self.config["overfit"]:
            overfit_set = Subset(self.train_set, [0])
            train_dataloader = DataLoader(overfit_set, batch_size=1,
                                          num_workers=4, drop_last=False, pin_memory=True,
                                          shuffle=True, collate_fn=collate_adni)
        else:
            train_dataloader = DataLoader(self.train_set, batch_size=self.config["batch_size"],
                                        num_workers=4, drop_last=False, pin_memory=True,
                                        shuffle=True, collate_fn=collate_adni)
        return train_dataloader

    def val_dataloader(self):
        if self.config["overfit"]:
            overfit_set = Subset(self.train_set, [0])
            val_dataloader = DataLoader(overfit_set, batch_size=1,
                                        num_workers=4, drop_last=False, pin_memory=True,
                                        collate_fn=collate_adni)
        else:
            val_dataloader = DataLoader(self.val_set, batch_size=self.config["batch_size"],
                                        num_workers=2, pin_memory=True, collate_fn=collate_adni)
        return val_dataloader

    def configure_optimizers(self):
        if self.config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config["lr"])
        return optimizer
    
    def _get_balanced_accuracy_from_confusion_matrix(self, confusion_matrix: ConfusionMatrix):
        # Confusion matrix whose i-th row and j-th column entry indicates
        # the number of samples with true label being i-th class and
        # predicted label being j-th class.
        cmat = confusion_matrix.compute()
        per_class = cmat.diag() / cmat.sum(dim=1)
        per_class = per_class[~torch.isnan(per_class)]  # remove classes that are not present in this dataset
        return per_class.mean()
    
class ResNetClassifier(L.LightningModule):
    def __init__(self, config, num_outputs=3):
        super(ResNetClassifier, self).__init__()
        self.model = ThreeDResNet(in_channels=1, n_outputs=num_outputs)
        self.num_outputs = num_outputs

        # Set config
        self.config = config

        # Set confusion matrix
        self.train_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)
        self.val_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)
        self.test_cf = ConfusionMatrix(task="multiclass", num_classes=self.num_outputs)

    def prepare_data(self):
        if self.config["data_aug"]:
            transform_train = transforms.Compose([
                EnsureChannelFirst(channel_dim=1),
                AddChannel(),
                ScaleIntensity(minv=0.0, maxv=1.0),
                RandFlip(prob=0.9),
                RandAffine(prob=0.9, rotate_range=(-90, 90), scale_range=(-0.05, 0.05), translate_range=(-10, 10))
            ])
        else:
            transform_train = transforms.Compose([
                EnsureChannelFirst(channel_dim=1),
                AddChannel()
            ])

        transform_val = transforms.Compose([
            EnsureChannelFirst(channel_dim=1),
            AddChannel()
        ])
        self.train_set = AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-train.h5"),
                                              transforms=transform_train, labels=[0,1,2], return_index=False)
        self.val_set = AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-valid.h5"),
                                            transforms=transform_val, labels=[0,1,2], return_index=False)
        self.test_set = AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{self.config['adni_fold_idx']}-test.h5"),
                                            transforms=transform_val, labels=[0,1,2], return_index=False)
        self.train_set.num_classes = 3
        self.train_set.targets = []
        for sample in self.train_set:
            self.train_set.targets.append(sample[-1])

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        self.logger.log_hyperparams(self.config)

    def on_train_end(self):
        best_model = ResNetClassifier.load_from_checkpoint(checkpoint_path=self.trainer.checkpoint_callback.best_model_path, config=self.config)
        best_model.eval()
        val_dataloader = self.val_dataloader()
        total_val_loss = 0
        total_val_acc = 0
        for batch_idx, val_batch in enumerate(val_dataloader):
            val_loss, val_acc = self._final_validation_step(val_batch, batch_idx, best_model)
            total_val_loss += val_loss
            total_val_acc += val_acc
            
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_acc = total_val_acc / len(val_dataloader)
        val_bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cf).item()
            
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_loss", value=avg_val_loss)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_acc", value=avg_val_acc)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_val_bacc", value=val_bacc)

        print("Final validation loss:", avg_val_loss)
        print("Final validation accuracy:", avg_val_acc)
        print("Final validation balanced accuracy:", val_bacc)

        self.final_val_loss = avg_val_loss
        self.final_val_acc = avg_val_acc
        self.final_val_bacc = val_bacc

        test_dataloader = self.test_dataloader()
        total_test_loss = 0
        total_test_acc = 0
        for batch_idx, test_batch in enumerate(test_dataloader):
            test_loss, test_acc = self._final_test_step(test_batch, batch_idx, best_model)
            total_test_loss += test_loss
            total_test_acc += test_acc
            
        avg_test_loss = total_test_loss / len(test_dataloader)
        avg_test_acc = total_test_acc / len(test_dataloader)
        test_bacc = self._get_balanced_accuracy_from_confusion_matrix(self.test_cf).item()
            
        self.logger.experiment.log_metric(self.logger._run_id, key="final_test_loss", value=avg_test_loss)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_test_acc", value=avg_test_acc)
        self.logger.experiment.log_metric(self.logger._run_id, key="final_test_bacc", value=test_bacc)

        print("Final test loss:", avg_test_loss)
        print("Final test accuracy:", avg_test_acc)
        print("Final test balanced accuracy:", test_bacc)

        self.final_test_loss = avg_test_loss
        self.final_test_acc = avg_test_acc
        self.final_test_bacc = test_bacc
        
    def training_step(self, batch, batch_idx):
        images, target = batch
        pred = self.model(images)
        loss = loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.train_cf.update(pred_labels, target)
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.train_cf)
        self.train_cf.reset()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        self.log("train_bacc", bacc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, target = batch
        pred = self.model(images)
        loss = loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.val_cf.update(pred_labels, target)
        bacc = self._get_balanced_accuracy_from_confusion_matrix(self.val_cf)
        self.val_cf.reset()
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True)
        self.log("val_bacc", bacc, on_step=True, on_epoch=True)
        return loss

    def _final_validation_step(self, batch, batch_idx, best_model):
        best_model.eval()
        images = batch[0].cuda()
        target = batch[-1].cuda()
        with torch.no_grad():
            pred = best_model(images)
        loss = loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.val_cf.update(pred_labels, target)
        return loss.item(), acc

    def _final_test_step(self, batch, batch_idx, best_model):
        best_model.eval()
        images = batch[0].cuda()
        target = batch[-1].cuda()
        with torch.no_grad():
            pred = best_model(images)
        loss = loss_fn(pred, target)
        pred_labels = torch.argmax(pred, dim=1)
        correct_preds = (pred_labels == target).sum().item()
        acc = correct_preds / target.shape[0]
        self.test_cf.update(pred_labels, target)
        return loss.item(), acc
        
    def train_dataloader(self):
        if self.config["overfit"]:
            overfit_set = Subset(self.train_set, [0])
            train_dataloader = DataLoader(overfit_set, batch_size=1,
                                          num_workers=4, drop_last=False, pin_memory=True,
                                          shuffle=True)
        else:
            train_dataloader = DataLoader(self.train_set, batch_size=self.config["batch_size"],
                                        num_workers=4, drop_last=False, pin_memory=True,
                                        shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        if self.config["overfit"]:
            overfit_set = Subset(self.train_set, [0])
            val_dataloader = DataLoader(overfit_set, batch_size=1,
                                        num_workers=4, drop_last=False, pin_memory=True)
        else:
            val_dataloader = DataLoader(self.val_set, batch_size=self.config["batch_size"],
                                        num_workers=2, pin_memory=True)
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(self.test_set, batch_size=self.config["batch_size"],
                                     num_workers=2, pin_memory=True)
        return test_dataloader

    def configure_optimizers(self):
        if self.config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config["lr"])
        return optimizer
    
    def _get_balanced_accuracy_from_confusion_matrix(self, confusion_matrix: ConfusionMatrix):
        # Confusion matrix whose i-th row and j-th column entry indicates
        # the number of samples with true label being i-th class and
        # predicted label being j-th class.
        cmat = confusion_matrix.compute()
        per_class = cmat.diag() / cmat.sum(dim=1)
        per_class = per_class[~torch.isnan(per_class)]  # remove classes that are not present in this dataset
        return per_class.mean()
    
