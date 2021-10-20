import os
import pandas as pd
import numpy as np
import random
import torch
from torchvision.utils import draw_bounding_boxes as draw_bb
from torchvision.utils import save_image

IMG_SIZE = 128


def _bb_pacman(img_gt):
    enemy_list = ['sue', 'inky', 'pinky', 'blinky']
    pieces = {"save_fruit": (171, 139, "S"), "score0": (183, 86, "S"),
              "pacman": (img_gt['player_y'], img_gt['player_x'], "M"),
              "fruit": (img_gt['fruit_y'], img_gt['fruit_x'], "M")}
    for en in enemy_list:
        if img_gt[f'{en}_visible']:
            pieces[en] = (img_gt[f'{en}_y'], img_gt[f'{en}_x'], "M")
    if img_gt['lives'] >= 2:
        pieces["life1"] = (170, 25, "S")
    if img_gt['lives'] >= 3:
        pieces["life2"] = (170, 42, "S")
    return pd.DataFrame.from_dict({
        k: [xy[1] * IMG_SIZE / 160.0 - 11, xy[0] * IMG_SIZE / 210.0,
            0.07 * IMG_SIZE if k != 'score0' else 0.2 * IMG_SIZE, 0.07 * IMG_SIZE, xy[2]]
        for k, xy in pieces.items()
    }, orient='index')


def _bb_carnival(img_gt):
    import ast
    print("Warning: Carnival does not yet mark moving objects")
    bbs = [(img_gt['bullets_x'].item() * 128 / 160.0 - 2, 198 * 128 / 210.0, 0.04 * 128, 0.04 * 128)]
    for animal in ['shooters', 'rabbits', 'owls', 'ducks', 'flying_ducks']:
        for x, y in ast.literal_eval(img_gt[animal].item()):
            bbs.append((x * 128 / 160.0 - 5, y * 128 / 210.0 - 5, 0.08 * 128, 0.08 * 128))
    for x, y in ast.literal_eval(img_gt['refills'].item()):
        bbs.append((x * 128 / 160.0 - 4, y * 128 / 210.0 - 1, 0.07 * 128, 0.05 * 128))

    if img_gt['bonus'].item():
        bbs.append((12 * 128 / 160.0, 29 * 128 / 210.0, 21 * 128 / 210.0, 8 * 128 / 210.0))

    return pd.DataFrame.from_dict({i: [bb[0], bb[1], bb[2], bb[3]] for i, bb in enumerate(bbs)}, orient='index')


def _bb_pong(img_gt):
    bbs = [(img_gt['enemy_x'] * 128 / 160.0 - 2, img_gt['enemy_y'] * 128 / 210.0 - 2, 0.05 * 128, 0.10 * 128, "M"),
           (img_gt['player_x'] * 128 / 160.0 - 2, img_gt['player_y'] * 128 / 210.0 - 2, 0.05 * 128, 0.10 * 128, "M"),
           (
               16, 0, 24, 13, "S"
           ),
           (
               79, 0, 24, 13, "S"
           )]
    if 'ball_x' in img_gt:
        bbs.append((img_gt['ball_x'] * 128 / 160.0 - 2, img_gt['ball_y'] * 128 / 210.0 - 2,
                    0.04 * 128, 0.04 * 128, "M"))
    return pd.DataFrame.from_dict({i: [bb[0], bb[1], bb[2], bb[3], bb[4]] for i, bb in enumerate(bbs)}, orient='index')

#TODO: after labels are done
def _bb_space_invaders(img_gt):
    bbs = [(
               16, 0, 24, 13, "S"
           ),
           (
               79, 0, 24, 13, "S"
           )]
    return pd.DataFrame.from_dict({i: [bb[0], bb[1], bb[2], bb[3], bb[4]] for i, bb in enumerate(bbs)}, orient='index')

def save(args, frame, info, output_path, visualizations):
    if args.game == "MsPacman":
        bb = _bb_pacman(info)
    elif args.game == "Carnival":
        bb = _bb_carnival(info)
    elif args.game == "Pong":
        bb = _bb_pong(info)
    elif args.game == "SpaceInvaders":
        bb = _bb_space_invaders(info)
    else:
        raise ValueError(f'Unsupported Game supplied: {args.game}')
    bb = bb[(bb[0] >= 0) & (bb[0] <= 128) & (bb[1] >= 0) & (bb[1] <= 128)]
    bb.to_csv(output_path, header=False, index=False)
    for vis in visualizations:
        vis.save_vis(frame, bb)
