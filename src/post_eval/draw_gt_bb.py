import gym
from PIL import Image
import PIL
from time import sleep
import argparse
import os
from collections import namedtuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import torch
from torchvision.utils import draw_bounding_boxes as draw_bb
from torchvision.utils import save_image
folder_sizes = {"train": 50, "test": 5000, "validation": 5000}


parser = argparse.ArgumentParser(
    description='Create the dataset for a specific game of ATARI Gym')
parser.add_argument('-g', '--game', type=str, help='An atari game',
                    # default='SpaceInvaders')
                    # default='Pong')
                    # default='Tennis')
                    default='MsPacman')
parser.add_argument('-f', '--folder', type=str, choices=folder_sizes.keys(),
                    required=True,
                    help='folder to write to: train, test or validation')
args = parser.parse_args()
bb_folder = f"../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/bb"
rgb_folder = f"../aiml_atari_data/with_bounding_boxes/{args.game}-v0/{args.folder}"
os.makedirs(rgb_folder, exist_ok=True)
rgb_folder_src = f"../aiml_atari_data/rgb/{args.game}-v0/{args.folder}"
pbar = tqdm(total=folder_sizes[args.folder])

for idx in range(folder_sizes[args.folder]):
    pil_img = Image.open(f'{rgb_folder_src}/{idx:05}.png', ).convert('RGB')
    pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
    image = np.array(pil_img)
    objects = torch.from_numpy(np.loadtxt(f'{bb_folder}/bb_{idx}.txt', delimiter=','))
    objects[:, 2:] += objects[:, :2]
    torch_img = torch.from_numpy(image).permute(2, 0, 1)
    bb_img = draw_bb(torch_img, objects)
    result = Image.fromarray(bb_img.permute(1, 2, 0).numpy())
    result.save(f'{rgb_folder}/{idx:05}.png')
    pbar.update(1)

