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

folder_sizes = {"train": 50000, "test": 5000, "validation": 5000}

parser = argparse.ArgumentParser(
    description='Create the dataset for a specific game of ATARI Gym')
parser.add_argument('-g', '--game', type=str, help='An atari game',
                    # default='SpaceInvaders')
                    # default='MsPacman')
                    # default='Tennis')
                    default='Carnival')
# parser.add_argument('-raw', '--raw_image', default=False, action="store_true",
#                     help='Wether to store original image from the gamef
parser.add_argument('--render', default=False, action="store_true",
                    help='renders the environment')
parser.add_argument('-s', '--stacks', default=True, action="store_false",
                    help='renders the environment')
parser.add_argument('-r', '--random', default=False, action="store_true",
                    help='renders the environment')
parser.add_argument('-f', '--folder', type=str, choices=folder_sizes.keys(),
                    required=True,
                    help='folder to write to: train, test or validation')
args = parser.parse_args()

# arguments.folder = None

"""
If you look at the atari_env source code, essentially:

v0 vs v4: v0 has repeat_action_probability of 0.25 (meaning 25% of the time the
previous action will be used instead of the new action),
while v4 has 0 (always follow your issued action)
Deterministic: a fixed frameskip of 4, while for the env without Deterministic,
frameskip is sampled from (2,5)
There is also NoFrameskip-v4 with no frame skip and no action repeat
vstochasticity.
"""

# env = AtariARIWrapper(gym.make(f'{arguments.game}Deterministic-v4'))
with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
    data = f'{json.load(f)}'.replace("'", '"')
    config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
if "Augmented" not in config.game_name:
    print("\n\n\t\tYou are not using an Augmented environment\n\n")
augmented = "Augmented" in config.game_name
env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
            history_length=config.history_length, max_no_op_actions=30)
state = env.reset()

make_deterministic(0, env)

bgr_folder = f"aiml_atari_data/space_like/{args.game}-v0/{args.folder}"
os.makedirs(bgr_folder, exist_ok=True)
rgb_folder = f"aiml_atari_data/rgb/{args.game}-v0/{args.folder}"
os.makedirs(rgb_folder, exist_ok=True)

agent_path = glob(f'agents/*{args.game}*')[0]

agent = load_agent(agent_path)

limit = folder_sizes[args.folder]
if args.random:
    np.random.shuffle(index)
image_count = 0
consecutive_images = 0
consecutive_images_info = []
series = []
# if arguments.game == "MsPacman":
for _ in range(200):
    action = agent.draw_action(state)
    # action = env.action_space.sample()
    if augmented:
        state, reward, done, info, obs = env.step(action)
    else:
        state, reward, done, info = env.step(action)
#     env.render()
#     sleep(0.01)
# exit()
i = 0
pbar = tqdm(total=limit)
obs = None


def if_done(state, done):
    global consecutive_images, consecutive_images_info
    if done:
        consecutive_images, consecutive_images_info = 0, []
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


# Add Flow
# Make sure dataset works with the new setting
while True:
    state, reward, done, info, obs = draw_action(args, agent, state)
    if (not args.random) or np.random.rand() < 0.01:
        augment_dict(obs if augmented else state, info, args.game)
        if args.stacks:
            draw_images(obs, consecutive_images)
            consecutive_images += 1
            consecutive_images_info.append(dict_to_serie(info))
            if consecutive_images == 4:
                consecutive_images = 0
                for i in range(4):
                    os.rename(f'{rgb_folder}/{i:05}.png', f'{rgb_folder}/{image_count:05}_{i}.png')
                    os.rename(f'{bgr_folder}/{i:05}.png', f'{bgr_folder}/{image_count:05}_{i}.png')
                    series.append(consecutive_images_info[i])
                for _ in range(20):
                    if_done(state, done)
                    action = agent.draw_action(state)
                    state, reward, done, info, obs = env.step(action)
                pbar.update(1)
                image_count += 1
            else:
                if_done(state, done)
        else:
            draw_images(obs, image_count)
            series.append(dict_to_serie(info))
            for _ in range(20):
                if_done(state, done)
                action = agent.draw_action(state)
                state, reward, done, info, obs = env.step(action)
            pbar.update(1)
            image_count += 1
        if image_count == limit:
            break

df = pd.DataFrame(series, dtype=int)
if args.game == "MsPacman":
    df.drop(["player_score", "num_lives", "ghosts_count", "player_direction"], axis=1, inplace=True)
    df["nb_visible"] = df[['sue_visible', 'inky_visible', 'pinky_visible', 'blinky_visible']].sum(1)
if args.random:
    mapping = dict(zip(np.arange(limit), index))
    df = df.iloc[index.argsort(kind='mergesort')]
    df.rename(mapping, inplace=True)
df.to_csv(f'{bgr_folder}/../{args.folder}_labels.csv')
print(f"Saved everything in {bgr_folder}")
df.to_csv(f'{rgb_folder}/../{args.folder}_labels.csv')
print(f"Saved everything in {rgb_folder}")
