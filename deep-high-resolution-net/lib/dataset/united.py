import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models

class UnitedDataset(Dataset):
    def __init__(self, cfg, valid_flag, dataset_name):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.valid_flag = valid_flag

        if self.valid_flag:
            self.dataset = eval('dataset.'+ dataset_name)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False, self.transform
            )
        else:
            self.dataset = eval('dataset.' + dataset_name)(
                cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True, self.transform
            )

        self.gesture_images = []
        for img_name in Path(cfg.GESTURE_DRAWINGS_DIR).iterdir():
            if img_name.is_dir():
                continue
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, cfg.MODEL.IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.gesture_images.append(img)

        # self.gesture_images = [
        #     cv2.resize(cv2.imread(str(img_name)), cfg.MODEL.IMAGE_SIZE)
        #     for img_name in Path(cfg.GESTURE_DRAWINGS_DIR).iterdir()
        # ]

        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.num_joints = 21

    def __getitem__(self, idx):
        if idx < len(self.dataset):
            # input, target, target_weight, meta = self.dataset[idx]
            # return input, target, target_weight, meta, 1
            input, target, target_weight, meta = self.dataset[idx]
            return input, target, target_weight, meta, 1

        idx -= len(self.dataset)
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target = np.zeros((self.num_joints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': '',
            'filename': '',
            'imgnum': 0,
            'joints': np.zeros((self.num_joints, 3), dtype=np.float32),
            'joints_vis': np.zeros((self.num_joints, 3), dtype=np.float32),
            'center': np.zeros((2, ), dtype=np.float32),
            'scale': np.zeros((2, ), dtype=np.float32),
            'rotation': 0,
            'score': 0
        }

        return self.transform(self.gesture_images[idx]), target, target_weight, meta, 0
        # return self.transform(self.gesture_images[idx]), target, target_weight, {}, 0

    def __len__(self):
        return len(self.dataset) + len(self.gesture_images)
