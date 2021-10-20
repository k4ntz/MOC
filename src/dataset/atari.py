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
    def __init__(self, root, mode, gamelist=None):
        assert mode in ['train', 'validation', 'test'], f'Invalid dataset mode "{mode}"'
        
        self.image_path = root
        self.game = gamelist[0]
        self.mode = mode

        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root) \
                    if gamelist is None or fn in gamelist \
                    for img in os.listdir(os.path.join(root, fn, mode)) if img.endswith(".png")]
        self.image_fn = image_fn
        self.all_labels = pd.read_csv(os.path.join(self.image_path, self.game, f"{mode}_labels.csv"))
        if "MsPacman" in self.game:
            self.all_labels = self.all_labels.rename(columns={'player_y': 'pacman_y', 'player_x': 'pacman_x'})
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
        path = osp.join(self.image_path, self.game, self.mode, 'bb')
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path

    def get_labels(self, batch_start, batch_end, boxes_batch):
        labels = []
        for img_idx, boxes in zip(range(batch_start * 4, batch_end * 4), boxes_batch):
            labels.append(get_labels(self.all_labels.iloc[[img_idx]], self.game, boxes))
        return labels

    def get_labels_moving(self, batch_start, batch_end, boxes_batch):
        labels = []
        for img_idx, boxes in zip(range(batch_start * 4, batch_end * 4), boxes_batch):
            labels.append(get_labels_moving(self.all_labels.iloc[[img_idx]], self.game, boxes))
        return labels

    def to_relevant(self, labels_moving):
        return to_relevant(self.game, labels_moving)

    def filter_relevant_boxes(self, boxes_batch):
        return filter_relevant_boxes(self.game, boxes_batch)
