import gym
from PIL import Image
from time import sleep
import argparse
import os
from collections import namedtuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
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
from eval import convert_to_boxes, read_boxes


def draw_images(args):
    bb_folder = f"../aiml_atari_data/space_like/{args['game']}-v0/{args['folder']}/bb"
    rgb_folder = f"../aiml_atari_data/with_bounding_boxes/{args['game']}-v0/{args['folder']}"
    os.makedirs(rgb_folder, exist_ok=True)
    rgb_folder_src = f"../aiml_atari_data/rgb/{args['game']}-v0/{args['folder']}"
    pbar = tqdm(total=folder_sizes[args['folder']])

    for idx in range(folder_sizes[args['folder']]):
        pil_img = Image.open(f'{rgb_folder_src}/{idx:05}.png', ).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        image = np.array(pil_img)
        objects = torch.from_numpy(np.loadtxt(f'{bb_folder}/bb_{idx}.txt', usecols=[0, 1, 2, 3], delimiter=','))
        objects[:, 2:] += objects[:, :2]
        torch_img = torch.from_numpy(image).permute(2, 0, 1)
        bb_img = draw_bb(torch_img, objects)
        result = Image.fromarray(bb_img.permute(1, 2, 0).numpy())
        result.save(f'{rgb_folder}/{idx:05}.png')
        pbar.update(1)

def _bb_pacman():
    img_gt = gt.iloc[[idx]]
    enemy_list = ['sue', 'inky', 'pinky', 'blinky']
    pieces = {"save_fruit": (171, 139, "S"), "score0": (183, 86, "S"),
              "pacman": (img_gt['player_y'].item(), img_gt['player_x'].item(), "M"),
              "fruit": (img_gt['fruit_y'].item(), img_gt['fruit_x'].item(), "M")}
    for en in enemy_list:
        if img_gt[f'{en}_visible'].item():
            pieces[en] = (img_gt[f'{en}_y'].item(), img_gt[f'{en}_x'].item(), "M")
    if img_gt['lives'].item() >= 3:
        pieces["life2"] = (170, 42, "S")
    if img_gt['lives'].item() >= 2:
        pieces["life1"] = (170, 25, "S")
    return pd.DataFrame.from_dict({k: [xy[1] * 128 / 160.0 - 11, xy[0] * 128 / 210.0,
                                     0.07 * 128 if k != 'score0' else 0.2 * 128, 0.07 * 128, xy[2]] for k, xy in
                                 pieces.items()},
                                orient='index')


def _bb_carnival():
    import ast
    print("Warning: Carnival does not yet mark moving objects")
    img_gt = gt.iloc[[idx]]
    bbs = [(img_gt['bullets_x'].item() * 128 / 160.0 - 2, 198 * 128 / 210.0, 0.04 * 128, 0.04 * 128)]
    for animal in ['shooters', 'rabbits', 'owls', 'ducks', 'flying_ducks']:
        for x, y in ast.literal_eval(img_gt[animal].item()):
            bbs.append((x * 128 / 160.0 - 5, y * 128 / 210.0 - 5, 0.08 * 128, 0.08 * 128))
    for x, y in ast.literal_eval(img_gt['refills'].item()):
        bbs.append((x * 128 / 160.0 - 4, y * 128 / 210.0 - 1, 0.07 * 128, 0.05 * 128))

    if img_gt['bonus'].item():
        bbs.append((12 * 128 / 160.0, 29 * 128 / 210.0, 21 * 128 / 210.0, 8 * 128 / 210.0))

    return pd.DataFrame.from_dict({i: [bb[0], bb[1], bb[2], bb[3]] for i, bb in enumerate(bbs)}, orient='index')


if __name__ == "__main__":
    folder_sizes = {"train": 50000, "test": 5000, "validation": 5000}

    parser = argparse.ArgumentParser(
        description='Check the dataset for a specific game of ATARI Gym')
    parser.add_argument('-g', '--game', type=str, help='An atari game',
                        default='MsPacman')
    parser.add_argument('-f', '--folder', type=str, choices=folder_sizes.keys(),
                        required=True,
                        help='folder to evaluate to: train, test or validation')
    args = parser.parse_args()

    bgr_folder = f"../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/bb"
    os.makedirs(bgr_folder, exist_ok=True)
    rgb_folder = f"../aiml_atari_data/rgb/{args.game}-v0/{args.folder}/bb"
    os.makedirs(rgb_folder, exist_ok=True)
    gt = pd.read_csv(f'{bgr_folder}/../../{args.folder}_labels.csv')
    print(f"Loaded Dataset from everything in {bgr_folder}/../../{args.folder}_labels.csv")
    pbar = tqdm(total=folder_sizes[args.folder])

    for idx in range(folder_sizes[args.folder]):
        if args.game == "MsPacman":
            bb = _bb_pacman()
        elif args.game == "Carnival":
            bb = _bb_carnival()
        else:
            raise ValueError(f'Unsupported Game supplied: {args.game}')
        bb = bb[(bb[0] > 0) & (bb[0] < 128) & (bb[1] > 0) & (bb[1] < 128)]
        bb.to_csv(f'{bgr_folder}/bb_{idx}.txt', header=False, index=False)
        bb.to_csv(f'{rgb_folder}/bb_{idx}.txt', header=False, index=False)
        pbar.update(1)

    print("Writing bb.txt is done... \n\n")
    draw_images({'folder': args.folder, 'game': args.game})
