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
from skimage.morphology import (disk, square)
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat, skeletonize)


class Atari(Dataset):
    def __init__(self, cfg, mode, ):
        assert mode in ['train', 'val', 'test'], f'Invalid dataset mode "{mode}"'
        mode = 'validation' if mode == 'val' else mode
        self.image_path = cfg.dataset_roots.ATARI
        self.flow_path = self.image_path.replace('space_like', 'median_delta')
        self.mode = mode
        self.game = cfg.gamelist[0]
        self.motion = cfg.arch.motion
        self.motion_kind = cfg.arch.motion_kind
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

    def __getitem__(self, stack_idx):
        fn = self.image_fn[index:index + 4]
        tensors = [self.img_path_to_tensor(stack_idx, i) for i in range(4)]
        if self.motion:
            torch_stack = torch.stack([t[0] for t in tensors])
            motion_z_pres = torch.stack([t[1] for t in tensors])
            motion_z_where = torch.stack([t[2] for t in tensors])
            return torch_stack, motion_z_pres, motion_z_where
        else:
            return torch.stack(tensors)

    def __len__(self):
        return len(self.image_fn) // 4

    def img_path_to_tensor(self, stack_idx, i):
        path = os.path.join(self.image_path, self.game, self.mode, f'{stack_idx:05}_{i}.png')
        pil_img = Image.open(path).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)

        image = np.array(pil_img)
        if self.motion:
            motion = np.load(path.replace(f'{i:05}.png', f'{i:05}.npy').replace('space_like', self.motion_kind))
            z_pres, z_where = process_motion(motion)
            return torch.from_numpy(image / 255).permute(2, 0, 1).float(), torch.from_numpy(z_pres), torch.from_numpy(
                z_where)
        else:
            return torch.from_numpy(image / 255).permute(2, 0, 1).float()

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


def process_motion(motion):
    """
    Converts the motion as magnitude of flow / delta to the median or mode of the last few images into
    z_pres and z_where, i.e. giving hints where the objects are so SPACE can one the one hand imitate it and
    on the other concentrate on finding sensible z_what representations
    :param motion: (B, 1, H_img, W_img)
    :return z_pres: (B, 1, G, G) in (-1, 1) (tanh)
    :return z_where: (B, 4, G, G), where all grid cells without z_pres are only contain zero
    """
    motion = motion > motion.mean()
    motion = (closing(motion, square(3)) * 255).astype(np.uint8)
    canny_output = cv.Canny(motion, 100, 200)  # Parameters irrelevant for our binary case
    contours = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    motion_z_pres = torch.zeros((arch.G * arch.G, 1))
    motion_z_where = torch.zeros((arch.G * arch.G, 4))
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        selection = np.apply_along_axis(unique_color, axis=2, arr=frame_copy[y:y + h, x:x + w])
        if w * h >= 30 and np.var(selection) > 1e-4:
            z_where_x = ((x + w / 2) / 128) * 2 - 1
            z_where_y = ((y + h / 2) / 128) * 2 - 1
            motion_z_where[(y + h // 2) // 8 * (x + w // 2) // 8] = np.array([w / 128, h / 128, z_where_x, z_where_y])
            motion_z_pres[(y + h // 2) // 8 * (x + w // 2) // 8, 0] = 1.0
    return motion_z_pres, motion_z_where


def unique_color(color):
    """
    Computes a unique value for uint8 array, e.g. for identifying the input color to make variance computation easy
    :param color: nd.array<n>
    """
    return sum([255 ** i * c for i, c in enumerate(color)])
