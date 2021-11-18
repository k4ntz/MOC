import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import skvideo.io as skv
from PIL import Image
import PIL
import os.path as osp
from .labels import get_labels, get_labels_moving, to_relevant, filter_relevant_boxes
import pandas as pd
import cv2 as cv
from skimage.morphology import (disk, square)
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat, skeletonize)
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as draw_bb

class Atari(Dataset):
    def __init__(self, cfg, mode):
        assert mode in ['train', 'val', 'test'], f'Invalid dataset mode "{mode}"'
        mode = 'validation' if mode == 'val' else mode
        self.image_path = cfg.dataset_roots.ATARI + cfg.dataset_style
        img_folder = "space_like" + cfg.dataset_style
        self.motion_path = self.image_path.replace(img_folder, cfg.arch.motion_kind)
        self.bb_base_path = self.image_path.replace(img_folder, 'bb')
        self.mode = mode
        self.game = cfg.gamelist[0]
        self.arch = cfg.arch
        self.transform = transforms.ToTensor()
        self.motion = cfg.arch.motion
        self.motion_kind = cfg.arch.motion_kind
        self.valid_flow_threshold = 20
        if len(cfg.gamelist) > 1:
            print(f"Evaluation currently only supported for exactly one game not {cfg.gamelist}")
        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(self.image_path)
                    if cfg.gamelist is None or fn in cfg.gamelist
                    for img in os.listdir(os.path.join(self.image_path, fn, mode)) if img.endswith(".png")]
        self.image_fn = image_fn

    def __getitem__(self, stack_idx):
        imgs = torch.stack([self.transform(self.read_img(stack_idx, i)) for i in range(4)])
        # fn = self.image_fn[index:index + 4]
        motion = torch.stack([self.read_tensor(stack_idx, i, postfix=f'{self.arch.img_shape[0]}') for i in range(4)])
        motion_z_pres = torch.stack([self.read_tensor(stack_idx, i, postfix="z_pres") for i in range(4)])
        motion_z_where = torch.stack([self.read_tensor(stack_idx, i, postfix="z_where") for i in range(4)])
        return imgs, (motion > motion.mean() * 0.1).float(), motion_z_pres, motion_z_where

    def __len__(self):
        return len(self.image_fn) // 4

    def read_img(self, stack_idx, i):
        path = os.path.join(self.image_path, self.game, self.mode, f'{stack_idx:05}_{i}.png')
        return np.array(Image.open(path).convert('RGB'))

    def read_tensor(self, stack_idx, i, postfix=None):
        path = os.path.join(self.motion_path, self.game, self.mode,
                            f'{stack_idx:05}_{i}_{postfix}.pt'
                            if postfix else f'{stack_idx:05}_{i}.pt')
        return torch.load(path)

    @property
    def bb_path(self):
        path = osp.join(self.bb_base_path, self.game, self.mode)
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path

    def get_labels(self, batch_start, batch_end, boxes_batch):
        labels = []
        bbs = []
        for stack_idx in range(batch_start, batch_end):
            for img_idx in range(4):
                bbs.append(pd.read_csv(os.path.join(self.bb_path, f"{stack_idx:05}_{img_idx}.csv"), header=None))
        for gt_bbs, boxes in zip(bbs, boxes_batch):
            labels.append(get_labels(gt_bbs, self.game, boxes))
        return labels

    def get_labels_moving(self, batch_start, batch_end, boxes_batch):
        labels = []
        bbs = []
        for stack_idx in range(batch_start, batch_end):
            for img_idx in range(4):
                bbs.append(pd.read_csv(os.path.join(self.bb_path, f"{stack_idx:05}_{img_idx}.csv"), header=None))
        for gt_bbs, boxes in zip(bbs, boxes_batch):
            labels.append(get_labels_moving(gt_bbs, self.game, boxes))
        return labels

    def to_relevant(self, labels_moving):
        return to_relevant(self.game, labels_moving)

    def filter_relevant_boxes(self, boxes_batch, boxes_gt):
        return filter_relevant_boxes(self.game, boxes_batch, boxes_gt)
