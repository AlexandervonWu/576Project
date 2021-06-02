import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
# from unet import UNet
from models.UNet import UNet

from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
# from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from loss import bceLoss
dir_img = './data/PhC-C2DL-PSC/01/'
dir_mask = './data/PhC-C2DL-PSC/01_ST/SEG/'



if __name__ == '__main__':
    try:
        train_net(net=net,
                  epochs=1,
                  batch_size=1,
                  lr=0.00001,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
        except Exception as err:
        # print (123)
        print(err)
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
