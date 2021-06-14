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

folder_sizes = {"train": 50000, "test": 5000, "validation": 5000}


parser = argparse.ArgumentParser(
    description='Create the dataset for a specific game of ATARI Gym')
parser.add_argument('-g', '--game', type=str, help='An atari game',
                    # default='SpaceInvaders')
                    # default='MsPacman')
                    # default='Tennis')
                    default='Pong')
# parser.add_argument('-raw', '--raw_image', default=False, action="store_true",
#                     help='Wether to store original image from the gamef
parser.add_argument('--render', default=False, action="store_true",
                    help='renders the environment')
parser.add_argument('-r', '--random', default=False, action="store_true",
                    help='renders the environment')
parser.add_argument('-f', '--folder', type=str, choices=folder_sizes.keys(),
                    required=True,
                    help='folder to write to: train, test or validation')
args = parser.parse_args()

# args.folder = None

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

# env = AtariARIWrapper(gym.make(f'{args.game}Deterministic-v4'))
with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
    data = f'{json.load(f)}'.replace("'", '"')
    config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
if "Augmented" not in config.game_name:
    print("\n\n\t\tYou are not using an Augmented environment\n\n")
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

# for _ in range(63):  # When the game is not moving
# for _ in range(120):  # When the game is not moving
#     state, _, _, info, _ = env.step(env.action_space.sample())

limit = folder_sizes[args.folder]
index = np.arange(limit)
if args.random:
    np.random.shuffle(index)
image_count = 0
series = []
# if args.game == "MsPacman":
for _ in range(200):
    action = agent.draw_action(state)
    # action = env.action_space.sample()
    state, reward, done, info, obs = env.step(action)
#     env.render()
#     sleep(0.01)
# exit()
i = 0
pbar = tqdm(total=limit)
while True:
    action = agent.draw_action(state)
    # action = env.action_space.sample()
    state, reward, done, info, obs = env.step(action)
    if args.render:
        env.render()
        sleep(0.01)
    # img = draw_names(obs, info)  # to see where the objects are
    if args.random and np.random.rand() < 0.01:
        image_n = index[image_count]
        try:
            if not augment_dict(obs, info, args.game): # wrong image
                # print("Wrong image")
                if done:
                    env.reset()
                    for _ in range(200):
                        action = agent.draw_action(state)
                        # action = env.action_space.sample()
                        state, reward, done, info, obs = env.step(action)
                continue
            ## RAW IMAGE
            img = Image.fromarray(obs, 'RGB')
            img.save(f'{rgb_folder}/{image_n:05}.png')

            ## BGR SPACE IMAGES
            img = Image.fromarray(
                obs[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
            img.save(f'{bgr_folder}/{image_n:05}.png')  # better quality than jpg
            series.append(dict_to_serie(info))
            if done:
                env.reset()
                for _ in range(200):
                    action = agent.draw_action(state)
                    # action = env.action_space.sample()
                    state, reward, done, info, obs = env.step(action)
            pbar.update(1)
            image_count += 1
            if image_count == limit:
                break
        except IndexError:
            # print("Wrong image")
            continue


df = pd.DataFrame(series, dtype=int)
# enemy_list = ['sue', 'inky', 'pinky', 'blinky', 'player']
# for en in enemy_list:
#     df.drop([f"{en}_x", f"{en}_y"], axis=1, inplace=True)
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
