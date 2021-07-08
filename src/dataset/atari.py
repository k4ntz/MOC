import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import skvideo.io as skv
from PIL import Image
import PIL


class Atari(Dataset):
    def __init__(self, root, mode, gamelist=None):
        assert mode in ['train', 'val', 'test'], f'Invalid dataset mode "{mode}"'
        
        self.image_path = root
        
        image_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root) \
                    if gamelist is None or fn in gamelist \
                    for img in os.listdir(os.path.join(root, fn, mode))]
        # self.video_path = os.checkpointdir.join(root, f'{key_word}')
        # self.video_path = root
        # self.video_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root)
        #                 if gamelist is None or fn in gamelist
        #                 for img in os.listdir(os.path.join(root, fn, mode))]
        # self.video_fn.sort()
        self.image_fn = image_fn
        self.image_fn.sort()
    
    def __getitem__(self, index):
        # fn = self.video_fn[index]
        # video = skv.vread(os.path.join(self.video_path, fn), outputdict={
        #     "-sws_flags": "bilinear",
        #     "-s": "128x128"
        # })
        # ([10, 3, 128, 128])
        # video_arr = torch.from_numpy(video / 255).permute(0, 3, 1, 2)
        # ([4, 3, 128, 128])
        # video_t = video_arr[:8:2].float()
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
        path = osp.join(self.root, self.mode, 'bb')
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path
