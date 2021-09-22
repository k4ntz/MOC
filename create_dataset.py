import gym
from PIL import Image
from time import sleep
import argparse
import os
from pprint import pprint
from utils import augment_dict, draw_names, show_image, load_agent, \
    dict_to_serie
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


def if_done(state, done):
    if done:
        env.reset()
        action = None
        for _ in range(99):
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
    if args.render:
        env.render()
        sleep(0.001)
    return env.step(action)


bgr_folder = None
rgb_folder = None
flow_folder = None
median_folder = None
bb_folder = None
vis_folder = None
env = None


def main():
    folder_sizes = {"train": 50, "test": 5000, "validation": 5000}
    data_base_folder = "aiml_atari_data2"

    parser = argparse.ArgumentParser(
        description='Create the dataset for a specific game of ATARI Gym')
    parser.add_argument('-g', '--game', type=str, help='An atari game',
                        # default='SpaceInvaders')
                        # default='MsPacman')
                        # default='Tennis')
                        default='Pong')
    parser.add_argument('--render', default=False, action="store_true",
                        help='renders the environment')
    parser.add_argument('-s', '--stacks', default=True, action="store_false",
                        help='should render in correlated stacks of 4')
    parser.add_argument('--median', default=True, action="store_false",
                        help='should compute median-delta')
    parser.add_argument('--bb', default=True, action="store_false",
                        help='should compute bounding_boxes')
    parser.add_argument('--flow', default=True, action="store_false",
                        help='should compute flow with default settings')
    parser.add_argument('--vis', default=True, action="store_false",
                        help='visualizes 100 images with different processing methods specified in motion_processing')
    parser.add_argument('-r', '--random', default=False, action="store_true",
                        help='shuffle the data')
    parser.add_argument('-f', '--folder', type=str, choices=folder_sizes.keys(),
                        required=True,
                        help='folder to write to: train, test or validation')
    args = parser.parse_args()

    agent, augmented, state = configure(args)
    limit = folder_sizes[args.folder]
    if args.random:
        np.random.shuffle(index)
    image_count = 0
    consecutive_images = []
    consecutive_images_info = []
    visualizations = []
    series = []
    for _ in range(200):
        action = agent.draw_action(state)
        if augmented:
            state, reward, done, info, obs = env.step(action)
        else:
            state, reward, done, info = env.step(action)
    create_folders(args, data_base_folder)
    pbar = tqdm(total=limit)
    while True:
        state, reward, done, info, obs = draw_action(args, agent, state)
        if (not args.random) or np.random.rand() < 0.01:
            augment_dict(obs if augmented else state, info, args.game)
            if args.stacks:
                consecutive_images += [obs]
                # dict_to_series removed
                consecutive_images_info.append(info)
                if len(consecutive_images) == 10:
                    imgs_space = [Image.fromarray(cons_img[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
                                  for cons_img in consecutive_images]
                    for i, (frame, img_info, frame_space) in enumerate(zip(consecutive_images[-4:],
                                                                           consecutive_images_info[-4:],
                                                                           imgs_space[-4:])):
                        img = Image.fromarray(frame, 'RGB')
                        img.save(f'{rgb_folder}/{image_count:05}_{i}.png')
                        frame_space.save(f'{bgr_folder}/{image_count:05}_{i}.png')
                        bb.save(args, img_info, f'{bb_folder}/{image_count:05}_{i}.txt')
                        series.append(img_info)
                    space_stack = np.stack([np.array(frame_space) for frame_space in imgs_space])
                    if args.flow:
                        flow.save(space_stack, f'{median_folder}/{image_count:05}_{{}}.png', visualizations)
                    if args.median:
                        median.save(space_stack, f'{median_folder}/{image_count:05}_{{}}.png', visualizations)
                    for _ in range(50):
                        if done:
                            state, reward, done, info, obs = some_steps(state)
                            consecutive_images, consecutive_images_info = 0, []
                        action = agent.draw_action(state)
                        state, reward, done, info, obs = env.step(action)
                    for i in range(10):
                        with contextlib.suppress(FileNotFoundError):
                            os.remove(f'{rgb_folder}/{i:05}.png')
                            os.remove(f'{bgr_folder}/{i:05}.png')
                    consecutive_images = []
                    pbar.update(1)
                    image_count += 1
                else:
                    if done:
                        state, reward, done, info, obs = some_steps(state)
                        consecutive_images, consecutive_images_info = 0, []
            else:
                draw_images(obs, image_count)
                series.append(info)
                for _ in range(50):
                    if_done(state, done)
                    action = agent.draw_action(state)
                    state, reward, done, info, obs = env.step(action)
                pbar.update(1)
                image_count += 1
            if image_count == limit:
                break

    df = pd.DataFrame(series, dtype=int)
    # TODO: Move
    if args.game == "MsPacman":
        df.drop(["player_score", "num_lives", "ghosts_count", "player_direction"], axis=1, inplace=True)
        df["nb_visible"] = df[['sue_visible', 'inky_visible', 'pinky_visible', 'blinky_visible']].sum(1)
    if args.random:
        print("Shuffling...")
        shuffle_indices = np.random.permutation(limit)
        mapping = dict(zip(np.arange(limit), shuffle_indices))
        df = df.iloc[shuffle_indices]
        df.reset_index(drop=True)
        folders = [bgr_folder, rgb_folder] + ([median_folder] if args.median else []) \
                  + ([flow_folder] if args.flow else []) + ([bb_folder] if args.bb else [])
        endings = ['.png', '.png', '.npy', '.npy', 'txt']
        for dataset_folder, ending in zip(folders, endings):
            for i, j in mapping.items():
                for file in glob.glob(f'{dataset_folder}/{i:05}*'):
                    os.rename(file, file.replace(f'{i:05}', f'{j:05}') + ".tmp")
            for i, _ in mapping.items():
                for file in glob.glob(f'{dataset_folder}/*'):
                    os.rename(file, file.replace(".tmp", ""))
        print("Shuffling done!")
    df.to_csv(f'{bgr_folder}/../{args.folder}_labels.csv')
    print(f"Saved everything in {bgr_folder}")
    df.to_csv(f'{rgb_folder}/../{args.folder}_labels.csv')
    print(f"Saved everything in {rgb_folder}")


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
    global rgb_folder, bgr_folder, flow_folder, median_folder, bb_folder
    rgb_folder = f"{data_base_folder}/rgb/{args.game}-v0/{args.folder}"
    bgr_folder = f"{data_base_folder}/space_like/{args.game}-v0/{args.folder}"
    bb_folder = f"{data_base_folder}/space_like/{args.game}-v0/{args.folder}/bb"
    flow_folder = f"{data_base_folder}/flow/{args.game}-v0/{args.folder}"
    median_folder = f"{data_base_folder}/median-delta/{args.game}-v0/{args.folder}"
    os.makedirs(bgr_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(flow_folder, exist_ok=True)
    os.makedirs(median_folder, exist_ok=True)
    os.makedirs(bb_folder, exist_ok=True)


if __name__ == '__main__':
    main()
