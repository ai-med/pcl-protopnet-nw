#################################################################
# extended and adapted from:
# https://github.com/salesforce/PCL
#################################################################

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier

import pcl.loader
import pcl.builder
from pcl.paths import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='3dresnet',
                    choices=["3dresnet", "densenet"],
                    help='encoder architecture')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--accumulation-steps', default=1, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--latent_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--pcl-r', default=16384, type=int,
                    help='queue size; number of negative pairs; needs to be smaller than num_cluster (default: 16384)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco-v2/SimCLR data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

parser.add_argument('--num-cluster', default='25000,50000,100000', type=str, 
                    help='number of clusters')
parser.add_argument('--warmup-epoch', default=20, type=int,
                    help='number of warm-up epochs to only train with InfoNCE loss')
parser.add_argument('--exp-dir', default='experiment_pcl', type=str,
                    help='experiment directory')

# Which dataset?
parser.add_argument('--dataset', default='ukbb', choices=["ukbb", "adni", "ukbb_adni"],
                    help='Which dataset to use for PCL.')

# For ADNI
parser.add_argument('--adni-labels', default='0,1,2', type=str,
                    help='The labels of ADNI samples that are considered for PCL')
parser.add_argument('--adni_fold_idx', default=0, type=int,
                    help="Which fold idx of the ADNI dataset")

# For using UKBB metadata
parser.add_argument('--use_ukbb_metadata', action='store_true', default=False,
                    help='If True, PCL on images + metadata (y-aware InfoNCE). Else, PCL only on images.')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    args.num_cluster = args.num_cluster.split(',')
    args.adni_labels = [int(label) for label in args.adni_labels.split(',')]
    
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # suppress printing if not master    
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = pcl.builder.MoCo(
        base_encoder_arch=args.arch,
        dim=args.latent_dim, r=args.pcl_r, m=args.moco_m, T=args.temperature, mlp=args.mlp, y_aware=args.use_ukbb_metadata)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # ----- Training data -----
    # UKBB dataset (img + tab)
    if args.dataset == "ukbb" and args.use_ukbb_metadata:
        train_dataset = pcl.loader.UkbbMRITabDataset(split="train", csv_file_path="data_split/img_tab.csv")
        eval_dataset = pcl.loader.UkbbMRITabDataset(split="val", csv_file_path="data_split/img_tab.csv")
    # UKBB dataset
    elif args.dataset == "ukbb" and not args.use_ukbb_metadata:
        train_dataset = pcl.loader.UkbbMRIDataset(split="train", csv_file_path="data_split/img.csv")
        eval_dataset = pcl.loader.UkbbMRIDataset(split="val", csv_file_path="data_split/img.csv")
    # ADNI dataset
    elif args.dataset == "adni":
        train_dataset = pcl.loader.AdniMRIDataset(split="train", path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-train.h5"),
                                                  labels=args.adni_labels)
        eval_dataset = pcl.loader.AdniMRIDataset(split="val", path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-train.h5"),
                                                 labels=args.adni_labels)
    # UKBB + ADNI datasets
    elif args.dataset == "ukbb_adni":
        train_ukbb_dataset = pcl.loader.UkbbMRIDataset(split="train", csv_file_path="data_split/img.csv")
        train_adni_dataset = pcl.loader.AdniMRIDataset(split="train", path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-train.h5"))
        train_dataset = ConcatDataset([train_ukbb_dataset, train_adni_dataset])
        eval_ukbb_dataset = pcl.loader.UkbbMRIDataset(split="val", csv_file_path="data_split/img.csv")
        eval_adni_dataset = pcl.loader.AdniMRIDataset(split="val", path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-train.h5"))
        eval_dataset = ConcatDataset([eval_ukbb_dataset, eval_adni_dataset])

    # ----- Validation data (ADNI) -----
    train_adni_dataset = pcl.loader.AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-train.h5"),
                                                          labels=args.adni_labels)
    eval_adni_dataset = pcl.loader.AdniMRIDataset_nonPCL(path=os.path.join(ADNI_DATA_PATH, f"{args.adni_fold_idx}-valid.h5"),
                                                         labels=args.adni_labels)
    train_adni_dataloader = DataLoader(train_adni_dataset, batch_size=2,
                                       num_workers=4, drop_last=False, pin_memory=False,
                                       shuffle=False)
    val_adni_dataloader = DataLoader(eval_adni_dataset, batch_size=2,
                                     num_workers=2, pin_memory=False,
                                     shuffle=False)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, drop_last=True)
    
    # dataloader for center-cropped images, use larger batch size to increase speed
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size*5, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=False)
    
    # Set up writer
    tb_writer = SummaryWriter(log_dir=f"{TENSORBOARD_DIR}/{args.exp_dir}")

    # Log hyperparams
    hparams = {
        "epochs": args.epochs,
        "start_epoch": args.start_epoch,
        "batch_size": args.batch_size,
        "acc_steps": args.accumulation_steps,
        "lr": args.lr,
        "schedule": ",".join([str(num) for num in args.schedule]),
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "latent_dim": args.latent_dim,
        "pcl_r": args.pcl_r,
        "moco_m": args.moco_m,
        "temperature": args.temperature,
        "mlp": args.mlp,
        "cos": args.cos,
        "num_cluster": ",".join([str(num) for num in args.num_cluster]),
        "warmup_epoch": args.warmup_epoch
    }
    print("hparams", hparams)
    tb_writer.add_hparams(hparams, {})

    # Set up variables to store best losses
    # best_protonce_loss = np.inf
    best_total_loss = np.inf
    save_next_cluster = False
    
    for epoch in range(args.start_epoch, args.epochs):
        
        cluster_result = None
        if epoch>=args.warmup_epoch:
            # compute momentum features for center-cropped images
            features = compute_features(eval_loader, model, args)         
            
            # placeholder for clustering result
            cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
            for num_cluster in args.num_cluster:
                cluster_result['im2cluster'].append(torch.zeros(len(eval_dataset),dtype=torch.long).cuda())
                cluster_result['centroids'].append(torch.zeros(int(num_cluster),args.latent_dim).cuda())
                cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda()) 

            if args.gpu == 0:
                features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice  
                features = features.numpy()
                cluster_result = run_kmeans(features,args)  #run kmeans clustering on master node
                # Save the cluster that resulted from the best model (from the previous epoch)
                if save_next_cluster:
                    torch.save(cluster_result, os.path.join(TENSORBOARD_DIR, args.exp_dir, 'clusters_%d'%epoch))
                    save_next_cluster = False
                
            dist.barrier()  
            # broadcast clustering result
            for k, data_list in cluster_result.items():
                for data_tensor in data_list:                
                    dist.broadcast(data_tensor, 0, async_op=False)     
    
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        print("Training...")
        total_loss = train(train_loader, model, criterion, optimizer, epoch, args, tb_writer, cluster_result)
        if total_loss < best_total_loss and epoch >= args.warmup_epoch: # The second condition ensures that the total_loss includes ProtoNCE loss
            # Save best model (encoder + momentum encoder)
            saved_checkpoints = [f for f in os.listdir(args.exp_dir) if f.startswith("checkpoint_")]
            if len(saved_checkpoints) > 0:
                os.remove(os.path.join(args.exp_dir, saved_checkpoints[0]))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename='{}/{}/checkpoint_{:04d}.pth.tar'.format(TENSORBOARD_DIR, args.exp_dir, epoch))
            # Save the cluster that resulted in this best model
            saved_clusters = [f for f in os.listdir(args.exp_dir) if f.startswith("clusters_")]
            if len(saved_clusters) > 0:
                for sc in saved_clusters:
                    os.remove(os.path.join(args.exp_dir, sc))
            torch.save(cluster_result, os.path.join(args.exp_dir, 'clusters_%d'%epoch))
            best_total_loss = total_loss
            save_next_cluster = True
        
        # Validation on ADNI dataset with kNN
        print("Validating...")
        model.eval()
        neigh = KNeighborsClassifier(n_neighbors=5)
        train_latent_features = torch.Tensor([])
        train_targets = torch.Tensor([])
        for train_batch in tqdm(train_adni_dataloader):
            with torch.no_grad():
                images = train_batch[0].cuda()
                latent_features = model(im_q=images, is_eval=True)
            train_latent_features = torch.cat([train_latent_features, latent_features.detach().cpu()], dim=0)
            target = train_batch[1]
            train_targets = torch.cat([train_targets, target], dim=0)

        neigh.fit(train_latent_features, train_targets)
        
        total_val_acc = 0
        for i, val_batch in tqdm(enumerate(val_adni_dataloader)):
            with torch.no_grad():
                batch_idx = epoch * len(val_adni_dataloader) + i
                images = val_batch[0].cuda()
                latent_features = model(im_q=images, is_eval=True)
                pred = torch.Tensor(neigh.predict(latent_features.cpu()))
                target = val_batch[1]
                acc = (pred == target).sum() / len(target)
                tb_writer.add_scalar("val_adni_acc", acc, batch_idx)
                total_val_acc += acc
        avg_val_acc = total_val_acc / len(val_adni_dataloader)
        tb_writer.add_scalar("val_adni_acc_epoch", avg_val_acc, epoch)

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        
def train(train_loader, model, criterion, optimizer, epoch, args, tb_writer, cluster_result=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_inst = AverageMeter('Acc@Inst', ':6.2f')   
    acc_proto = AverageMeter('Acc@Proto', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc_inst, acc_proto],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    total_losses = []
    for i, (images, index, labels) in enumerate(train_loader): # labels here are not labels of the samples, but y_aware labels
        batch_idx = epoch * len(train_loader) + i

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=False) # Original: non_blocking=True
            images[1] = images[1].cuda(args.gpu, non_blocking=False)

        if labels.size()[1] > 0: # The dimension of the labels is at least 1 (There is at least a label) -> (B, num_labels)
            loss, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index, labels=labels)
            # InfoNCE loss
            tb_writer.add_scalar("train_y_aware_infonce_loss", loss, batch_idx)
        else: # If no labels are given, then labels.shape is (B, 0)
            labels = None
            output, target, output_proto, target_proto = model(im_q=images[0], im_k=images[1], cluster_result=cluster_result, index=index, labels=labels)
            # InfoNCE loss
            loss = criterion(output, target)
            tb_writer.add_scalar("train_infonce_loss", loss, batch_idx)
            acc = accuracy(output, target)[0]
            tb_writer.add_scalar("train_acc", acc, batch_idx)
            acc_inst.update(acc[0], images[0].size(0))
        
        # ProtoNCE loss
        if output_proto is not None:
            loss_proto = 0
            for proto_out, proto_target in zip(output_proto, target_proto): # Iterate through all sets of prototypes
                loss_proto += criterion(proto_out, proto_target)  
                accp = accuracy(proto_out, proto_target)[0] 
                acc_proto.update(accp[0], images[0].size(0))
                
            # average loss across all sets of prototypes
            loss_proto /= len(args.num_cluster) 
            tb_writer.add_scalar("train_protonce_loss", loss_proto, batch_idx)
            loss += loss_proto
        
        tb_writer.add_scalar("train_loss", loss, batch_idx)

        losses.update(loss.item(), images[0].size(0))
        total_losses.append(loss.item())
        
        # compute gradient and do SGD step
        loss.backward()

        if (i + 1) % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    if len(total_losses) > 0:
        avg_total_loss = np.sum(np.array(total_losses)) / len(total_losses)
    else:
        avg_total_loss = np.inf
    
    return avg_total_loss

            
def compute_features(eval_loader, model, args):
    print('Computing features...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset),args.latent_dim).cuda()
    for i, (images, index, labels) in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images = images.cuda(non_blocking=False) # Original: non_blocking=True
            feat = model(images,is_eval=True) 
            features[index] = feat
    dist.barrier()        
    dist.all_reduce(features, op=dist.ReduceOp.SUM)     
    return features.cpu()

    
def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
