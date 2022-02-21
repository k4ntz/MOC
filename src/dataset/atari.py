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

class Atari(Dataset):
    def __init__(self, cfg, mode):
        assert mode in ['train', 'val', 'test'], f'Invalid dataset mode "{mode}"'
        mode = 'validation' if mode == 'val' else mode
        self.image_path = cfg.dataset_roots.ATARI + cfg.dataset_style
        img_folder = "space_like" + cfg.dataset_style
        self.bb_base_path = self.image_path.replace(img_folder, 'bb')
        self.mode = mode
        self.game = cfg.gamelist[0]
        self.arch = cfg.arch
        if len(cfg.gamelist) > 1:
            print(f"Evaluation currently only supported for exactly one game not {cfg.gamelist}")
        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(self.image_path)
                    if cfg.gamelist is None or fn in cfg.gamelist
                    for img in os.listdir(os.path.join(self.image_path, fn, mode)) if img.endswith(".png")]
        self.image_fn = image_fn
        self.image_fn.sort()

    def __getitem__(self, index):
        fn = self.image_fn[index]
        
        pil_img = Image.open(os.path.join(self.image_path, fn)).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        
        image = np.array(pil_img)
        
        image_t = torch.from_numpy(image / 255).permute(2, 0, 1).float()
        
        return image_t
    
    def __len__(self):
        return len(self.image_fn)


    @property
    def bb_path(self):
        path = osp.join(self.bb_base_path, self.game, self.mode)
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path

    def get_labels_impl(self, batch_start, batch_end, boxes_batch, extractor):
        labels = []
        bbs = []
        for stack_idx in range(batch_start, batch_end):
            for img_idx in range(4):
                bbs.append(pd.read_csv(os.path.join(self.bb_path, f"{stack_idx:05}_{img_idx}.csv"), header=None))
        idx = 0
        sub_labels = []
        for gt_bbs, boxes in zip(bbs, boxes_batch):
            sub_labels.append(extractor(gt_bbs, self.game, boxes))
            idx += 1
            if idx == 4:
                labels.append(sub_labels)
                sub_labels = []
                idx = 0
        return labels

    def get_labels(self, batch_start, batch_end, boxes_batch):
        return self.get_labels_impl(batch_start, batch_end, boxes_batch, get_labels)

    def get_labels_moving(self, batch_start, batch_end, boxes_batch):
        return self.get_labels_impl(batch_start, batch_end, boxes_batch, get_labels_moving)

    def to_relevant(self, labels_moving):
        return to_relevant(self.game, labels_moving)

    def filter_relevant_boxes(self, boxes_batch, boxes_gt):
        return filter_relevant_boxes(self.game, boxes_batch, boxes_gt)