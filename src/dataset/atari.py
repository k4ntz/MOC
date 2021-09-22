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


class Atari(Dataset):
    def __init__(self, root, mode, gamelist=None, flow=False):
        assert mode in ['train', 'val', 'test'], f'Invalid dataset mode "{mode}"'
        mode = 'validation' if mode == 'val' else mode
        self.image_path = root
        self.flow_path = self.image_path.replace('space_like', 'median_delta')
        self.mode = mode
        self.game = gamelist[0]
        self.ground_truth = False
        self.flow = flow
        self.valid_flow_threshold = 20
        self.all_labels = pd.read_csv(os.path.join(self.image_path, self.game, f"{mode}_labels.csv"))
        if "MsPacman" in self.game:
            self.all_labels = self.all_labels.rename(columns={'player_y': 'pacman_y', 'player_x': 'pacman_x'})
        if len(gamelist) > 1:
            print(f"Evaluation currently only supported for exactly one game not {gamelist}")
        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root) \
                    if gamelist is None or fn in gamelist \
                    for img in os.listdir(os.path.join(root, fn, mode))
                    if img.endswith(".png")]
        self.image_fn = image_fn
        self.image_fn.sort()

    def __getitem__(self, index):
        index *= 4
        index += self.flow
        fn = self.image_fn[index:index + 4]
        torch_stack = torch.stack([self.img_path_to_tensor(i) for i in range(index, index + 4)])
        return torch_stack

    def __len__(self):
        return (len(self.image_fn) - self.flow) // 4

    def img_path_to_tensor(self, i):
        path = os.path.join(self.image_path, self.game, self.mode, f'{i:05}.png')
        pil_img = Image.open(path).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)

        image = np.array(pil_img)
        if self.flow:
            if self.ground_truth:
                filename = os.path.join(self.bb_path, 'bb_{}.txt'.format(i))
                gt = np.zeros((128, 128, 1))
                with open(filename, 'r') as f:
                    for line in f:
                        if line.strip():
                            x, y, width, height = [float(x) for x in
                                                                 line.strip().replace(",S", "").replace(",M", "").split(
                                                                     ',')]
                            width //= 2
                            height //= 2
                            gt[int(y + height), int(x + width), 0] = 255
                image = np.append(image, gt, axis=2)
            else:
                # flow = np.load(os.path.join(self.flow_path, path.replace('.png', '.npy')))
                # flow = np.expand_dims((flow * flow).sum(axis=2), axis=2)
                # flow = flow * 255 / flow.max()
                # image = np.append(image, flow, axis=2)
                flow = np.load(path.replace(f'{i:05}.png', f'{i:05}.npy').replace('space_like', 'median_delta'))
                flow = np.expand_dims(flow, axis=2) * 255
                image = np.append(image, flow, axis=2)
        return torch.from_numpy(image / 255).permute(2, 0, 1).float()

    @property
    def bb_path(self):
        path = osp.join(self.image_path, self.game, self.mode, 'bb')
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path

    def get_labels(self, batch_start, batch_end, boxes_batch):
        labels = []
        for i, boxes in zip(range(batch_start * 4, batch_end * 4), boxes_batch):
            img_idx = i + self.flow
            labels.append(get_labels(self.all_labels.iloc[[img_idx]], self.game, boxes))
        return labels

    def get_labels_moving(self, batch_start, batch_end, boxes_batch):
        labels = []
        for i, boxes in zip(range(batch_start * 4, batch_end * 4), boxes_batch):
            img_idx = i + self.flow
            labels.append(get_labels_moving(self.all_labels.iloc[[img_idx]], self.game, boxes))
        return labels

    def to_relevant(self, labels_moving):
        return to_relevant(self.game, labels_moving)

    def filter_relevant_boxes(self, boxes_batch):
        return filter_relevant_boxes(self.game, boxes_batch)

