import argparse
import os
import time
import numpy as np

from importlib import import_module
import shutil
from utils import *
import sys
sys.path.append('../')
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_training import config as config_training

from layers import acc

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='dpn3d26',
                    help='model')
parser.add_argument('--config', '-c', default='config_training', type=str)
parser.add_argument('-j', '--workers', default=30, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshod for get pbb')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=4, type=int, metavar='N',
                    help='number of gpu for test')

def main():
    global args
    args = parser.parse_args()
    config_training = import_module(args.config)
    config_training = config_training.config
    # from config_training import config as config_training
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()

    
    net = net.cuda()
    net = torch.nn.DataParallel(net).cuda()

    # Load model checkpoint
    checkpoint = torch.load('dpnmodel/fd9044.ckpt')
    
    # Create a new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        # Remove the `module.` prefix
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    # Load the new_state_dict into your model
    model.load_state_dict(new_state_dict)

    # Load preprocessed data
    data = np.load('/mnt/md1/VH_test/test_VH_processed.npy')
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()  # Add batch and channel dimensions
    
    # Define SplitComb for handling large 3D images
    margin = 32
    sidelen = 144
    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
    output, _ = singletest(data, net, config, split_comber.split, split_comber.combine, 1)
    
    # Get pbb and save
    thresh = args.testthresh
    pbb, _ = get_pbb(output, thresh, ismask=True)
    np.save('mnt/md1/VH_test/prediction_test_VH_processed.npy', pbb)
    print('Prediction saved.')

def singletest(data, net, config, splitfun, combinefun, n_per_run, margin=64, isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    data = splitfun(data, config['max_stride'], margin)
    splitlist = range(0, len(data) + 1, n_per_run)
    outputlist = []
    for i in range(len(splitlist) - 1):
        input = data[splitlist[i]:splitlist[i + 1]]
        output = net(input)
        outputlist.append(output.data.cpu().numpy())
    output = np.concatenate(outputlist, 0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    return output

if __name__ == '__main__':
    main()
