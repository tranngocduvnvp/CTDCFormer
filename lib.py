import random
import cv2
import torch
from torch.utils import data
import torchvision.transforms.functional as TF
import numpy as np
import random
import multiprocessing
import timm
import torchmetrics
import sys
import os
import argparse
import time
import numpy as np
import glob
from PIL import Image

import torch.nn as nn
import torchmetrics.functional.classification as Fmstric
import torchgeometry as tgm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import _cfg


from sklearn.model_selection import train_test_split
from torchvision import transforms
import glob
import matplotlib.pyplot as plt


class Args:
    def __init__(self,root, epochs, batch_size, dataset, mgpu, lrs_min,\
                 lrs, lr, type_lr, checkpoint_path, backbone, optim):
        self.root = root
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = dataset
        self.mgpu = mgpu
        self.lrs_min = lrs_min
        self.lrs = lrs
        self.lr = lr
        self.type_lr = type_lr
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.optim = optim
        
args = Args(
    root="/home/tranngocdu/Documents/paper/CTDCFormer/data_new/TestDataset/CVC-ColonDB", 
    epochs=40, 
    batch_size=4, 
    dataset="Kvasir_18_12",
    mgpu="false",
    lrs="true",
    lrs_min=1e-6,
    lr = 1e-4,
    type_lr = "StepLR",
    checkpoint_path = "/home/tranngocdu/Documents/paper/CTDCFormer/CTDCFormer/checkpoint/CTDCformer_epoch_backbonePvitB4_generalizability_20_12.pt",
    backbone="PvtB3",
    optim="AdamW"
)
