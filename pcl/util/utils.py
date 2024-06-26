#################################################################
# extended and adapted from:
# https://github.com/alanqrwang/nwhead
#################################################################

import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
import argparse

try:
    import wandb
except ImportError as e:
    pass

def summary(network):
    """Print model summary."""
    print()
    print('Model Summary')
    print('---------------------------------------------------------------')
    print(network)
    print('---------------------------------------------------------------')
    print()
    print('Trainable parameters:')
    print('---------------------------------------------------------------')
    for name, param in network.named_parameters():
        if param.requires_grad:
            print(name)
    print()
    print('Total parameters:', sum(p.numel() for p in network.parameters() if p.requires_grad))
    print('---------------------------------------------------------------')
    print()

######### Saving/Loading checkpoints ############
def load_checkpoint(network, path, optimizer=None, scheduler=None, verbose=True):
    if verbose:
        print('Loading checkpoint from', path)
        if optimizer:
            print('Loading optimizer')
        if scheduler:
            print('Loading scheduler')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
  
    network.load_state_dict(checkpoint['network_state_dict'])

    if optimizer is not None and scheduler is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        return network, optimizer, scheduler

    elif optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return network, optimizer

    else:
        return network

def save_checkpoint(epoch, network, optimizer, model_folder, scheduler=None, is_best=False):
    state = {
        'epoch': epoch,
        'network_state_dict': network.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()

    if is_best:
        saved_checkpoints = [f for f in os.listdir(model_folder) if f.startswith("model.")]
        if len(saved_checkpoints) > 0:
            os.remove(os.path.join(model_folder, saved_checkpoints[0]))
        filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')
        torch.save(state, filename.format(epoch=epoch))
        print('Saved checkpoint to', filename.format(epoch=epoch))

def parse_bool(v):
    if v.lower()=='true':
        return True
    elif v.lower()=='false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_pcl_encoder_weights(path, momentum_encoder=True):
    checkpoint_filename = [f for f in os.listdir(path) if f.endswith(".tar")][0]
    checkpoint = torch.load(os.path.join(path, checkpoint_filename))
    state_dict = checkpoint['state_dict']
    if momentum_encoder:
        key = "encoder_k"
    else:
        key = "encoder_q"
    for k in list(state_dict.keys()):
        if k.startswith(f'module.{key}'):
            state_dict[k[len(f"module.{key}."):]] = state_dict[k]
        del state_dict[k]
    return state_dict

# Taken from https://sumit-ghosh.com/articles/parsing-dictionary-key-value-pairs-kwargs-argparse-python/
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value_str = value.split('=')
            if value_str.replace('-','').isnumeric():
                processed_val = int(value_str)
            elif value_str.replace('-','').replace('.','').isnumeric():
                processed_val = float(value_str)
            elif value_str in ['True', 'true']:
                processed_val = True
            elif value_str in ['False', 'false']:
                processed_val = False
            else:
                processed_val = value_str
            getattr(namespace, self.dest)[key] = processed_val