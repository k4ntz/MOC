import gym
from PIL import Image
from time import sleep
import argparse
import os
from pprint import pprint
from augmentation import augment_dict, draw_names, show_image, load_agent, \
    dict_to_serie, put_lives, set_plot_bb
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
    ClosingMeanThreshold, IteratedCentroidSelection, Skeletonize, Identity, FlowBoundingBox, ZWhereZPres, set_color_hist
import contextlib

"""
If you look at the atari_env source code, essentially:

v0 vs v4: v0 has repeat_action_probability of 0.25 (meaning 25% of the time the
previous action will be used instead of the new action),
while v4 has 0 (always follow your issued action)
Deterministic: a fixed frameskip of 4, while for the env without Deterministic,
frameskip is sampled from (2,5)
There is also NoFrameskip-v4 with no frame skip and no action repeat
stochasticity.
"""


def some_steps(agent, state):
    env.reset()
    action = None
    for _ in range(10):
        action = agent.draw_action(state)
        state, reward, done, info, obs = env.step(action)
    return env.step(action)


def draw_images(obs, image_n):
    ## RAW IMAGE
    img = Image.fromarray(obs, 'RGB')
    img.save(f'{rgb_folder}/{image_n:05}.png')
    ## BGR SPACE IMAGES
    img = Image.fromarray(
        obs[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
    img.save(f'{bgr_folder}/{image_n:05}.png')  # better quality than jpg


def draw_action(args, agent, state):
    action = agent.draw_action(state)
    return env.step(action)


bgr_folder = None
bgr84_folder = None
bgr64_folder = None
rgb_folder = None
flow_folder = None
median_folder = None
mode_folder = None
bb_folder = None
vis_folder = None
env = None


def compute_root_median(args, data_base_folder):
    imgs = [np.array(Image.open(f), dtype=np.uint8) for f in glob(f"{rgb_folder}/*") if ".png" in f]
    img_arr = np.stack(imgs)
    # Ensures median exists in any image at least, even images lead to averaging
    if len(img_arr) % 2:
        print("Removing one image for median computation to ensure P(median|game) != 0")
        img_arr = img_arr[:-1]
    median = np.median(img_arr, axis=0).astype(np.uint8)
    mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=img_arr).astype(np.uint8)
    frame = Image.fromarray(median)
    frame.save(f"{data_base_folder}/vis/{args.game}-v0/median.png")
    frame = Image.fromarray(mode)
    frame.save(f"{data_base_folder}/vis/{args.game}-v0/mode.png")

def main():
    parser = argparse.ArgumentParser(
        description='Create some images and corresponding mode-motion object detection')
    parser.add_argument('-g', '--game', type=str, help='An atari game',
                        # default='SpaceInvaders')
                        default='MsPacman')
                        # default='Tennis')
                        # default='SpaceInvaders')
    args = parser.parse_args()
    print("=============" * 5)
    print("Settings:", args)
    print("=============" * 5)
    data_base_folder = "aiml_atari_data_mid"
    create_folders(args, data_base_folder)
    visualizations_mode = [
        Identity(vis_folder, "Mode", every_n=1, max_vis=100000000, saturation=10),
        ZWhereZPres(vis_folder, "Mode", every_n=1, max_vis=100000000),
    ]
    agent, augmented, state = configure(args)
    limit = 100
    for _ in range(200):
        action = agent.draw_action(state)
        if augmented:
            state, reward, done, info, obs = env.step(action)
        else:
            state, reward, done, info = env.step(action)

    pbar = tqdm(total=limit)
    mode = np.array(Image.open(f"{data_base_folder}/vis/{args.game}-v0/mode.png"))
    print("Ensuring that global median (mode) is used.")
    for i in range(limit):
        state, reward, done, info, obs = draw_action(args, agent, state)
        Image.fromarray(obs).save(f'{vis_folder}/Mode/{i:05}.png')
        mode_delta = np.abs(obs - mode)
        mode_delta = np.max(mode_delta, axis=-1)
        delta_max = mode_delta.max()
        mode_delta = mode_delta / delta_max if delta_max > 0 else mode_delta
        for vis in visualizations_mode:
            vis.save_vis(obs, mode_delta)
        pbar.update(1)


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
    env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
                history_length=config.history_length, max_no_op_actions=30)
    state = env.reset()
    make_deterministic(0, env)
    agent_path = glob(f'agents/*{args.game}*')[0]
    agent = load_agent(agent_path)
    return agent, augmented, state


def create_folders(args, data_base_folder):
    global vis_folder
    vis_folder = f"{data_base_folder}/vis/{args.game}-v0/train/Gifs"
    os.makedirs(vis_folder + "/Mode", exist_ok=True)


if __name__ == '__main__':
    main()
