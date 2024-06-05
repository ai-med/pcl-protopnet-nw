#################################################################
# extended and adapted from:
# https://github.com/alanqrwang/nwhead
#################################################################

import os
import random
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
from pprint import pprint
import json
from torchmetrics import ConfusionMatrix

from pcl.util.metric import Metric, ECELoss
from pcl.util.utils import summary, save_checkpoint, get_pcl_encoder_weights
from pcl.util import metric
from pcl.nwhead.nw import NWNet

from pcl.loader import *
from pcl.builder import *
from torchpanic.models.backbones import ThreeDResNet
from torch.utils.tensorboard import SummaryWriter

class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='NW Head Training')
        # I/O parameters
        self.add_argument('--exp_dir', default='./',
                  type=str, help='directory to save models')
        self.add_argument('--workers', type=int, default=0,
                  help='Num workers')
        self.add_argument('--gpu_id', type=int, default=0,
                  help='gpu id to train on')
        self.add_bool_arg('debug_mode', False)

        # Machine learning parameters
        self.add_argument('--lr', type=float, default=1e-3,
                  help='Learning rate')
        self.add_argument('--batch_size', type=int,
                  default=1, help='Batch size')
        self.add_argument('--num_steps_per_epoch', type=int,
                  default=10000000, help='Num steps per epoch')
        self.add_argument('--num_val_steps_per_epoch', type=int,
                  default=10000000, help='Num validation steps per epoch')
        self.add_argument('--num_epochs', type=int, default=200,
                  help='Total training epochs')
        self.add_argument('--scheduler_milestones', nargs='+', type=int,
                  default=(100, 150), help='Step size for scheduler')
        self.add_argument('--scheduler_gamma', type=float,
                  default=0.1, help='Multiplicative factor for scheduler')
        self.add_argument('--seed', type=int,
                  default=0, help='Seed')
        self.add_argument('--weight_decay', type=float,
                  default=1e-4, help='Weight decay')
        self.add_argument('--arch', type=str, default='3dresnet', choices=["3dresnet", "densenet"])
        self.add_bool_arg('freeze_featurizer', False)

        # NW head parameters
        self.add_argument('--kernel_type', type=str, default='euclidean',
                  help='Kernel type')
        self.add_argument('--proj_dim', type=int,
                  default=0)
        self.add_argument('--n_shot', type=int,
                  default=1, help='Number of examples per class in support')
        self.add_argument('--n_way', type=int,
                  default=None, help='Number of training classes per query in support')
        
        # PCL-NW
        self.add_argument('--pcl_encoder_checkpoint_path', type=str)
        self.add_argument('--latent_dim', type=int)
        self.add_bool_arg('use_pretrained_encoder', True)
        self.add_bool_arg('data_aug', True)
        self.add_argument('--adni_fold_idx', type=int, default=0)

    def add_bool_arg(self, name, default=True):
        """Add boolean argument to argparse parser"""
        group = self.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no_' + name, dest=name, action='store_false')
        self.set_defaults(**{name: default})

    def parse(self):
        args = self.parse_args()
        print("--use_pretrained_encoder", args.use_pretrained_encoder)
        print("--data_aug", args.data_aug)
        args.run_dir = os.path.join(TENSORBOARD_DIR, args.exp_dir,
                      'arch{arch}_lr{lr}_bs{batch_size}_projdim{proj_dim}_nshot{nshot}_nway{nway}_wd{wd}_seed{seed}'.format(
                        arch=args.arch,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        proj_dim=args.proj_dim,
                        nshot=args.n_shot,
                        nway=args.n_way,
                        wd=args.weight_decay,
                        seed=args.seed
                      ))
        args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
        if not os.path.exists(args.run_dir):
            os.makedirs(args.run_dir)
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        # Print args and save to file
        print('Arguments:')
        pprint(vars(args))
        with open(args.run_dir + "/args.txt", 'w') as args_file:
            json.dump(vars(args), args_file, indent=4)
        return args


def main():
    # Parse arguments
    args = Parser().parse()

    # Set random seed
    seed = args.seed
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Set device
    if torch.cuda.is_available():
        args.device = torch.device('cuda:'+str(args.gpu_id))
    else:
        args.device = torch.device('cpu')
        print('No GPU detected... Training will be slow!')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Set Tensorboard writer
    tb_writer = SummaryWriter(log_dir=f"{TENSORBOARD_DIR}/{args.exp_dir}")

    # Get dataloaders
    if args.data_aug:
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
    train_dataset = AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-train.h5"), transforms=transform_train, labels=[0,1,2], return_index=False)
    val_dataset = AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-valid.h5"), labels=[0,1,2], return_index=False)
    train_dataset.num_classes = 3
    train_dataset.targets = []
    for sample in tqdm(train_dataset):
        train_dataset.targets.append(sample[-1])

    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    latent_dim = args.latent_dim
    if args.arch == "3dresnet":
        featurizer = ThreeDResNet(in_channels=1, n_outputs=latent_dim)
        if args.use_pretrained_encoder:
            state_dict = get_pcl_encoder_weights(args.pcl_encoder_checkpoint_path, momentum_encoder=True)
            featurizer.load_state_dict(state_dict)
    elif args.arch == "densenet":
        featurizer = DenseNetEncoder(dim=latent_dim)
        if args.use_pretrained_encoder:
            state_dict = get_pcl_encoder_weights(args.pcl_encoder_checkpoint_path, momentum_encoder=True)
            featurizer.load_state_dict(state_dict)     

    featurizer = featurizer.cuda()
    
    if args.freeze_featurizer:
        for param in featurizer.parameters():
            param.requires_grad = False
    
    network = NWNet(featurizer, 
                    num_classes,
                    support_dataset=train_dataset,
                    feat_dim=latent_dim,
                    proj_dim=args.proj_dim,
                    kernel_type=args.kernel_type,
                    n_shot=args.n_shot,
                    n_way=args.n_way,
                    debug_mode=args.debug_mode)
    
    summary(network)
    network.to(args.device)

    # Set loss, optimizer, and scheduler
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(network.parameters(), 
                                lr=args.lr, 
                                momentum=0.9, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                          milestones=args.scheduler_milestones,
                          gamma=args.scheduler_gamma)

    
    # Tracking metrics
    list_of_metrics = [
        'loss:train',
        'acc:train',
        'bacc:train'
    ]
    list_of_val_metrics = [
        'loss:val:random',
        'loss:val:full',
        'loss:val:cluster',
        'acc:val:random',
        'acc:val:full',
        'acc:val:cluster',
        'bacc:val:random',
        'bacc:val:full',
        'bacc:val:cluster',
        'ece:val:random',
        'ece:val:full',
        'ece:val:cluster',
    ] 
   
    args.metrics = {}
    args.metrics.update({key: Metric() for key in list_of_metrics})
    args.val_metrics = {}
    args.val_metrics.update({key: Metric() for key in list_of_val_metrics})

    # Training loop
    start_epoch = 1
    best_bacc1 = 0
    for epoch in range(start_epoch, args.num_epochs+1):
        print('Epoch:', epoch)
        network.eval()
        network.precompute()
        print('Evaluating on random mode...')
        eval_epoch(val_loader, network, criterion, optimizer, args, mode='random')
        print('Evaluating on full mode...')
        bacc1 = eval_epoch(val_loader, network, criterion, optimizer, args, mode='full')
        print('Evaluating on cluster mode...')
        eval_epoch(val_loader, network, criterion, optimizer, args, mode='cluster')

        print('Training...')
        train_epoch(train_loader, network, criterion, optimizer, args)
        scheduler.step()

        # Remember best acc and save checkpoint
        is_best = bacc1 > best_bacc1
        best_bacc1 = max(bacc1, best_bacc1)

        if is_best:
            save_checkpoint(epoch, network, optimizer,
                      args.ckpt_dir, scheduler, is_best=is_best)
            
        print("Train loss={:.6f}, train acc={:.6f}, train bacc={:.6f}, lr={:.6f}".format(
            args.metrics['loss:train'].result(), args.metrics['acc:train'].result(), args.metrics['bacc:train'].result(), scheduler.get_last_lr()[0]))
        tb_writer.add_scalar("train_loss", args.metrics['loss:train'].result(), epoch)
        tb_writer.add_scalar("train_acc", args.metrics['acc:train'].result(), epoch)
        tb_writer.add_scalar("train_bacc", args.metrics['bacc:train'].result(), epoch)

        print("Val loss={:.6f}, val acc={:.6f}, val bacc={:.6f}".format(
            args.val_metrics['loss:val:random'].result(), args.val_metrics['acc:val:random'].result(), args.val_metrics['bacc:val:random'].result()))
        print("Val loss={:.6f}, val acc={:.6f}, val bacc={:.6f}".format(
            args.val_metrics['loss:val:full'].result(), args.val_metrics['acc:val:full'].result(), args.val_metrics['bacc:val:full'].result()))
        print("Val loss={:.6f}, val acc={:.6f}, val bacc={:.6f}".format(
            args.val_metrics['loss:val:cluster'].result(), args.val_metrics['acc:val:cluster'].result(), args.val_metrics['bacc:val:cluster'].result()))
        print()
        tb_writer.add_scalar("val_loss_random", args.val_metrics['loss:val:random'].result(), epoch)
        tb_writer.add_scalar("val_acc_random", args.val_metrics['acc:val:random'].result(), epoch)
        tb_writer.add_scalar("val_bacc_random", args.val_metrics['bacc:val:random'].result(), epoch)
        tb_writer.add_scalar("val_loss_full", args.val_metrics['loss:val:full'].result(), epoch)
        tb_writer.add_scalar("val_acc_full", args.val_metrics['acc:val:full'].result(), epoch)
        tb_writer.add_scalar("val_bacc_full", args.val_metrics['bacc:val:full'].result(), epoch)
        tb_writer.add_scalar("val_loss_cluster", args.val_metrics['loss:val:cluster'].result(), epoch)
        tb_writer.add_scalar("val_acc_cluster", args.val_metrics['acc:val:cluster'].result(), epoch)
        tb_writer.add_scalar("val_bacc_cluster", args.val_metrics['bacc:val:cluster'].result(), epoch)

        # Reset metrics
        for _, metric in args.metrics.items():
            metric.reset_state()
        for _, metric in args.val_metrics.items():
            metric.reset_state()

def train_epoch(train_loader, network, criterion, optimizer, args):
    """Train for one epoch."""
    network.train()

    preds = []
    gts = []
    for i, batch in tqdm(enumerate(train_loader), 
        total=min(len(train_loader), args.num_steps_per_epoch)):
        step_res = nw_step(batch, network, criterion, optimizer, args, is_train=True)
        args.metrics['loss:train'].update_state(step_res['loss'], step_res['batch_size'])
        args.metrics['acc:train'].update_state(step_res['acc'], step_res['batch_size'])
        preds.append(step_res['pred'])
        gts.append(step_res['gt'])
        if i == args.num_steps_per_epoch:
            break

    # Calculate bacc
    cf = ConfusionMatrix(task="multiclass", num_classes=3).to(args.device)
    cf.update(torch.cat(preds, dim=0).squeeze(), torch.cat(gts, dim=0).squeeze())
    cmat = cf.compute()
    per_class = cmat.diag() / cmat.sum(dim=1)
    per_class = per_class[~torch.isnan(per_class)]
    bacc = per_class.mean()
    args.metrics['bacc:train'].update_state(bacc, 1)

def eval_epoch(val_loader, network, criterion, optimizer, args, mode='random'):
    '''Eval for one epoch.'''
    network.eval()

    preds = []
    probs = []
    gts = []
    for i, batch in tqdm(enumerate(val_loader), 
        total=min(len(val_loader), args.num_val_steps_per_epoch)):
        step_res = nw_step(batch, network, criterion, optimizer, args, is_train=False, mode=mode)
        args.val_metrics[f'loss:val:{mode}'].update_state(step_res['loss'], step_res['batch_size'])
        args.val_metrics[f'acc:val:{mode}'].update_state(step_res['acc'], step_res['batch_size'])
        preds.append(step_res['pred'])
        probs.append(step_res['prob'])
        gts.append(step_res['gt'])
        if i == args.num_val_steps_per_epoch:
            break
    
    ece = (ECELoss()(torch.cat(probs, dim=0), torch.cat(gts, dim=0)) * 100).item()

    # Calculate bacc
    cf = ConfusionMatrix(task="multiclass", num_classes=3).to(args.device)
    cf.update(torch.cat(preds, dim=0).squeeze(), torch.cat(gts, dim=0).squeeze())
    cmat = cf.compute()
    per_class = cmat.diag() / cmat.sum(dim=1)
    per_class = per_class[~torch.isnan(per_class)]
    bacc = per_class.mean()

    args.val_metrics[f'ece:val:{mode}'].update_state(ece, 1)
    args.val_metrics[f'bacc:val:{mode}'].update_state(bacc, 1)
    return args.val_metrics[f'bacc:val:{mode}'].result()

def nw_step(batch, network, criterion, optimizer, args, is_train=True, mode='random'):
    '''Train/val for one step.'''
    img, label = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        if is_train:
            output = network(img, label)
        else:
            output = network.predict(img, mode)
        loss = criterion(output, label)
        if is_train:
            loss.backward()
            optimizer.step()
        acc = metric.acc(output.argmax(-1), label)

    return {'loss': loss.cpu().detach().numpy(), \
            'acc': acc*100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'pred': output.argmax(-1), \
            'gt': label}

if __name__ == '__main__':
    main()