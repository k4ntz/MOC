import gym
from PIL import Image
from time import sleep
import argparse
import os
from pprint import pprint
from augmentation import augment_dict, draw_names, show_image, load_agent, \
    dict_to_serie, put_lives, set_plot_bb, image_offset
from glob import glob
from mushroom_rl.environments import Atari
import json
from collections import namedtuple
from utils_rl import make_deterministic
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from src.dataset import bb
from src.motion import median
from src.motion import flow
from src.motion import mode
from src.motion.motion_processing import ProcessingVisualization, BoundingBoxes, \
    ClosingMeanThreshold, IteratedCentroidSelection, Skeletonize, Identity, FlowBoundingBox, ZWhereZPres, \
    set_color_hist, set_special_color_weight
import contextlib
from procgen import ProcgenGym3Env, ProcgenEnv


rgb_folder = None
bb_eval_folder = None
bb_folder = None


def main():
    parser = argparse.ArgumentParser(
        description='Create the dataset for a specific game of ATARI Gym')
    parser.add_argument('-g', '--game', type=str, help='An atari game',
                        # default='SpaceInvaders')
                        # default='MsPacman')
                        # default='Tennis')
                        default='SpaceInvaders')
    args = parser.parse_args()
    print("=============" * 5)
    print("Settings:", args)
    print("=============" * 5)
    data_base_folder = "aiml_atari_data"
    create_folders(args, data_base_folder)
    bb_vis = BoundingBoxes(bb_eval_folder, '', max_vis=50, every_n=1)

    for i in range(50):
        stack_idx = np.random.randint(0, 8192)
        frame_idx = np.random.randint(0, 4)
        bb = pd.read_csv(os.path.join(f"{bb_folder}", f"{stack_idx:05}_{frame_idx}.csv"), header=None)
        frame = np.array(Image.open(f"{rgb_folder}/{stack_idx:05}_{frame_idx}.png"))
        bb_vis.save_vis(frame, bb)
    print(f"Dataset Generation is completed. Everything is saved in {data_base_folder}.")


def configure(args):
    global env
    # env = AtariARIWrapper(gym.make(f'{arguments.game}Deterministic-v4'))
    with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
        data = f'{json.load(f)}'.replace("'", '"')
        config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    if "Augmented" not in config.game_name:
        print("\n\n\t\tYou are not using an Augmented environment\n\n")
    augmented = "Augmented" in config.game_name
    print(f"Playing {config.game_name}...")
    if config.game_name.lower() == 'coinrun':
        env_name = "ecoinrun"
        env = ProcgenEnv(num_envs=1, env_name=env_name, center_agent=False) # use_backgrounds=False, restrict_themes=True
        state = None
    else:
        env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
                    history_length=config.history_length, max_no_op_actions=30)
        env.augmented = True
        state = env.reset()
        make_deterministic(0 if args.folder == "train" else 1 if args.folder == "validation" else 2, env)
    agent = load_agent(args, env)
    return agent, augmented, state


def create_folders(args, data_base_folder):
    global rgb_folder, bb_eval_folder, bb_folder
    rgb_folder = f"{data_base_folder}/rgb/{args.game}-v0/train"
    bb_eval_folder = f"{data_base_folder}/bb_eval/{args.game}-v0/train"
    bb_folder = f"{data_base_folder}/bb/{args.game}-v0/train"
    os.makedirs(bb_eval_folder, exist_ok=True)


if __name__ == '__main__':
    main()
