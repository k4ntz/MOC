import os
import sys
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import PIL

class Atari(Dataset):
    def __init__(self, root, mode, cfg, gamelist=None):
        assert mode in ['train', 'validation', 'test'], f'Invalid dataset mode "{mode}"'

        # self.image_path = os.checkpointdir.join(root, f'{key_word}')
        self.image_path = root

        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root) \
                    if gamelist is None or fn in gamelist \
                    for img in os.listdir(os.path.join(root, fn, mode))]
        self.image_fn = image_fn
        self.image_fn.sort()
        # preprocessing flags
        self.black_bg = getattr(cfg, mode).black_background
        self.dilation = getattr(cfg, mode).dilation

    def __getitem__(self, index):
        fn = self.image_fn[index]

        pil_img = Image.open(os.path.join(self.image_path, fn)).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        # convert image to opencv
        opencv_img = np.asarray(pil_img).copy()
        if self.black_bg:
            # get most dominant color
            colors, count = np.unique(opencv_img.reshape(-1,opencv_img.shape[-1]), axis=0, return_counts=True)
            most_dominant_color = colors[count.argmax()]
            # create the mask and use it to change the colors
            bounds_size = 1
            lower = most_dominant_color - [bounds_size, bounds_size, bounds_size]
            upper = most_dominant_color + [bounds_size, bounds_size, bounds_size]
            mask = cv2.inRange(opencv_img, lower, upper)
            opencv_img[mask != 0] = [0,0,0]
        # dilation 
        if self.dilation:
            kernel = np.ones((3,3), np.uint8)
            opencv_img = cv2.dilate(opencv_img, kernel, iterations=1)
        # convert to tensor
        image_t = torch.from_numpy(opencv_img / 255).permute(2, 0, 1).float()

        return image_t

    def __len__(self):
        return len(self.image_fn)
