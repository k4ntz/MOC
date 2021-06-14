import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
import skvideo.io as skv


class Atari(Dataset):
    def __init__(self, root, mode, gamelist=None):
        assert mode in ['train', 'validation', 'test'], f'Invalid dataset mode "{mode}"'

        # self.video_path = os.checkpointdir.join(root, f'{key_word}')
        self.video_path = root
        print(os.listdir(root))
        print(os.listdir(os.path.join(root, "Pong-v0", mode)))
        self.video_fn = [os.path.join(fn, mode, img) for fn in os.listdir(root)
                         if gamelist is None or fn in gamelist
                         for img in os.listdir(os.path.join(root, fn, mode))]
        self.video_fn.sort()
        print(self.video_path)
        print(self.video_fn[3])


    def __getitem__(self, index):
        fn = self.video_fn[index]
        video = skv.vread(os.path.join(self.video_path, fn))
        print(os.path.join(self.video_path, fn))
        # pil_img = Image.open(os.path.join(self.video_path, fn)).convert('RGB')
        # pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)

        video_t = torch.from_numpy(video / 255).permute(0, 3, 1, 2).float()

        return video_t

    def __len__(self):
        return len(self.video_fn)
