import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import skvideo.io as skv
from PIL import Image
import PIL
import os.path as osp


class Atari(Dataset):
    def __init__(self, root, mode, gamelist=None):
        assert mode in ['train', 'val', 'test'], f'Invalid dataset mode "{mode}"'
        mode = 'validation' if mode == 'val' else mode
        self.image_path = root
        self.mode = mode
        self.game = gamelist[0]
        if len(gamelist) > 1:
            print(f"Evaluation currently only supported for exactly one game not {gamelist}")
        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root) \
                    if gamelist is None or fn in gamelist \
                    for img in os.listdir(os.path.join(root, fn, mode))
                    if img.endswith(".png")]
        self.image_fn = image_fn
        self.image_fn.sort()
    
    def __getitem__(self, index):
        base_idx = min(index, len(self.image_fn) - 4)
        fn = self.image_fn[base_idx:base_idx + 4]
        torch_stack = torch.stack([self.img_path_to_tensor(f) for f in fn])
        return torch_stack

    def __len__(self):
        return len(self.image_fn)

    def img_path_to_tensor(self, path):
        pil_img = Image.open(os.path.join(self.image_path, path)).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)

        image = np.array(pil_img)
        return torch.from_numpy(image / 255).permute(2, 0, 1).float()

    @property
    def bb_path(self):
        path = osp.join(self.image_path, self.game, self.mode, 'bb')
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path
