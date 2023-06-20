import gym
from PIL import Image
from time import sleep
import argparse
import os
from pprint import pprint
from augmentation import augment_dict, draw_names, show_image, load_agent, \
    dict_to_serie, put_lives, set_plot_bb, image_offset
from glob import glob
from utils_rl import Atari
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
from src.niceprint import pprint as print
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
    env.env.reset()
    action = None
    for _ in range(10):
        state, reward, done, info, obs = draw_action(agent, state)
    return draw_action(agent, state)


def draw_images(obs, image_n):
    ## RAW IMAGE
    img = Image.fromarray(obs, 'RGB')
    img.save(f'{rgb_folder}/{image_n:05}.png')
    ## BGR SPACE IMAGES
    img = Image.fromarray(
        obs[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
    img.save(f'{bgr_folder}/{image_n:05}.png')  # better quality than jpg


def draw_action(agent, state):
    action = agent.draw_action(state)
    # if agent.game == "coinrun":
    #     action = np.array([action])
    #     observation, reward, done, info = env.step(action)
    #     image = np.array(Image.fromarray(observation['rgb'][0], 'RGB').resize((160, 210), Image.ANTIALIAS))
    #
    #     return observation, reward, done, {}, observation['rgb'][0][-210:, :500, :3]
    state, reward, done, info, obs = env.step(action)
    return state, reward, done, info, obs[:210, :160, :3]


bgr_folder = None
rgb_folder = None
flow_folder = None
median_folder = None
mode_folder = None
bb_folder = None
vis_folder = None
env = None


# def compute_root_images(args, data_base_folder):
#     imgs = [np.array(Image.open(f), dtype=np.uint8) for f in glob(f"{rgb_folder}/*") if ".png" in f]

def compute_root_images(imgs, data_base_folder, game):
    img_arr = np.stack(imgs)
    # Ensures median exists in any image at least, even images lead to averaging
    if len(img_arr) % 2:
        print("Removing one image for median computation to ensure P(median|game) != 0")
        img_arr = img_arr[:-1]
    median = np.median(img_arr, axis=0).astype(np.uint8)
    mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=img_arr).astype(np.uint8)
    frame = Image.fromarray(median)
    os.makedirs(f"{data_base_folder}/{game}-v0/background", exist_ok=True)
    frame.save(f"{data_base_folder}/{game}-v0/background/median.png")
    frame = Image.fromarray(mode)
    frame.save(f"{data_base_folder}/{game}-v0/background/mode.png")
    print("blue", f"Saved mode.png and median.png in {data_base_folder}/{game}-v0/background/")


def main():
    parser = argparse.ArgumentParser(
        description='Create the dataset for a specific game of ATARI Gym')
    parser.add_argument('-g', '--game', type=str, help='An atari game',
                        # default='SpaceInvaders')
                        # default='MsPacman')
                        # default='Tennis')
                        default='SpaceInvaders')
    parser.add_argument('--compute_root_images', default=False, action="store_true",
                        help='instead compute the median and mode of images found images')
    # parser.add_argument('--root', default=True, action="store_true",
    #                     help='use the root-mode (or root-median --median) instead of the trail')
    parser.add_argument('--no_color_hist', default=False, action="store_true",
                        help='use the color_hist to filter')
    parser.add_argument('--render', default=False, action="store_true",
                        help='renders the environment')
    parser.add_argument('-s', '--stacks', default=True, action="store_false",
                        help='should render in correlated stacks of 4')
    parser.add_argument('--median', default=False, action="store_true",
                        help='should compute median-delta instead of mode')
    parser.add_argument('--trail', default=False, action="store_true",
                        help='Use the trail to compute the delta instead of root image (default False)')
    parser.add_argument('--bb', default=True, action="store_false",
                        help='should compute bounding_boxes')
    parser.add_argument('--plot_bb', default=False, action="store_true",
                        help='should plot bounding_boxes')
    parser.add_argument('--no_flow', action="store_true",
                        help='should not compute flow information (default False)')
    parser.add_argument('--vis', default=True, action="store_false",
                        help='visualizes 100 images with different processing methods specified in motion_processing')
    parser.add_argument('-r', '--random', default=False, action="store_true",
                        help='shuffle the data')
    parser.add_argument('-f', '--folder', type=str, choices=["train", "test", "validation"],
                        required=True,
                        help='folder to write to: train, test or validation')
    parser.add_argument('-fs', '--folder_size', type=str, choices=["train", "test", "validation"],
                        help='nb of image in the folder\n \
                        default train: 8192, test: 1024, validation: 1024',
                        default=None)
    args = parser.parse_args()
    print("box", "Settings:", args)
    folder_sizes = {"train": 8192, "test": 1024, "validation": 1024}
    limit = args.folder_size if args.folder_size else folder_sizes[args.folder]
    data_base_folder = "aiml_atari_data"
    mode_base_folder = "aiml_atari_data"
    REQ_CONSECUTIVE_IMAGE = 30 if args.trail else 20
    create_folders(args, data_base_folder)
    visualizations_flow = [
        Identity(vis_folder, "Flow", max_vis=50, every_n=1),
    ]
    visualizations_median = [
    ]
    visualizations_mode = [
        Identity(vis_folder, "Mode", max_vis=50, every_n=1),
        ZWhereZPres(vis_folder, "Mode", max_vis=20, every_n=2),
    ]
    visualizations_bb = [BoundingBoxes(vis_folder, '', max_vis=20, every_n=1)]

    agent, augmented, state = configure(args)
    print("configuration done")

    if args.compute_root_images:
        limit = 1000
        imgs = []
        pbar = tqdm(total=limit)
        while len(imgs) < limit:
            state, reward, done, info, obs = draw_action(agent, state)
            if np.random.rand() < 0.01:
                imgs.append(obs)
                pbar.update(1)
            if done:
                env.reset()
                for _ in range(100):
                    state, reward, done, info, obs = draw_action(agent, state)
        pbar.close()
        compute_root_images(imgs, data_base_folder, args.game)
        exit(0)
    set_plot_bb(args.plot_bb)
    # Trick for easier labeling
    if "Tennis" in args.game:
        image_offset(f"offsets/tennis.png")
    if "Riverraid" in args.game:
        import ipdb; ipdb.set_trace()
        image_offset(f"offsets/riverraid.png")
    if args.random:
        np.random.shuffle(index)
    image_count = 0
    consecutive_images = []
    consecutive_images_info = []

    series = []
    print("Init steps...")
    for _ in range(50):
        state, reward, done, info, obs = draw_action(agent, state)

    mode_path = f"{mode_base_folder}/{args.game}-v0/background/"
    if not args.trail:
        if args.median and not os.path.exists(f"{mode_path}/median.png"):
            print("red", f"Counldn't find {mode_path}/median.png, use --trail to use the trail instead")
            exit()
        elif not os.path.exists(f"{mode_path}/mode.png"):
            print("red", f"Counldn't find {mode_path}/mode.png, use --trail to use the trail instead")
            exit(1)
        root_median = np.array(Image.open(f"{mode_base_folder}/{args.game}-v0/background/median.png"))[:, :, :3]
        root_mode = np.array(Image.open(f"{mode_base_folder}/{args.game}-v0/background/mode.png"))[:, :, :3]
        print("Ensuring that global median (mode) is used.")
        if not args.no_color_hist:
            set_color_hist(root_mode)
            # Exceptions where mode-delta is not working well, but it is easily fixed,
            # by marking some colors interesting or uninteresting respectively.
            # Those would be no issue with FlowNet
            if "Pong" in args.game:
                set_special_color_weight(15406316, 8)
            if "Airraid" in args.game:
                set_special_color_weight(0, 20000)
            if "Riverraid" in args.game:
                set_special_color_weight(3497752, 20000)
    else:
        print("Ensuring that trail median or mode is used.")
        root_median, root_mode = None, None
    pbar = tqdm(total=limit)
    enemy_scored = False
    while True:
        state, reward, done, info, obs = draw_action(agent, state)
        if enemy_scored and info["labels"]["enemy_score"] == 0:
            some_steps(agent, state)
            enemy_scored = False
        elif not enemy_scored and info["labels"]["enemy_score"]:
            enemy_scored = True
        if (obs==0).all(): # black screen
            continue
        if args.render:
            env.render()
        if (not args.random) or np.random.rand() < 0.01:
            augment_dict(obs, info, args.game)
            if args.stacks:
                consecutive_images += [obs]
                consecutive_images_info.append(put_lives(info))
                if len(consecutive_images) == REQ_CONSECUTIVE_IMAGE:
                    space_stack = []
                    for frame in consecutive_images[:-4]:
                        space_stack.append(frame)
                    resize_stack = []
                    for i, (frame, img_info) in enumerate(zip(consecutive_images[-4:], consecutive_images_info[-4:])):
                        space_stack.append(frame)
                        frame_space = Image.fromarray(frame[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
                        resize_stack.append(np.array(frame_space))
                        frame_space.save(f'{bgr_folder}/{image_count:05}_{i}.png')
                        img = Image.fromarray(frame, 'RGB')
                        img.save(f'{rgb_folder}/{image_count:05}_{i}.png')
                        try:
                            bb.save(args, frame_space, img_info, f'{bb_folder}/{image_count:05}_{i}.csv',
                                    visualizations_bb)
                        except RuntimeError:
                            import ipdb; ipdb.set_trace()
                    # for i, fr in enumerate(space_stack):
                    #     Image.fromarray(fr, 'RGB').save(f'{vis_folder}/Mode/Stack_{image_count:05}_{i:03}.png')
                    resize_stack = np.stack(resize_stack)
                    space_stack = np.stack(space_stack)
                    if not args.no_flow:
                        flow.save(space_stack, f'{flow_folder}/{image_count:05}_{{}}.pt', visualizations_flow)
                    if args.median:
                        median.save(space_stack, f'{median_folder}/{image_count:05}_{{}}.pt', visualizations_median,
                                    median=root_median)
                    else:
                        if args.game == "coinrun":
                            mode.save_coinrun(space_stack, f'{mode_folder}/{image_count:05}_{{}}.pt',
                                              visualizations_mode, space_frame=resize_stack)
                        else:
                            mode.save(space_stack, f'{mode_folder}/{image_count:05}_{{}}.pt', visualizations_mode,
                                      mode=root_mode, space_frame=resize_stack)
                    while done:
                        state, reward, done, info, obs = some_steps(agent, state)
                        step = 0
                    consecutive_images, consecutive_images_info = [], []
                    pbar.update(1)
                    image_count += 1
                else:
                    while done:
                        state, reward, done, info, obs = some_steps(agent, state)
                        consecutive_images, consecutive_images_info = [], []
            else:
                # Untested
                draw_images(obs, image_count)
                series.append(put_lives(info))
                for _ in range(50):
                    while done:
                        state, reward, done, info, obs = some_steps(agent, state)
                    action = agent.draw_action(state)
                    state, reward, done, info, obs = env.step(action)
                pbar.update(1)
                image_count += 1
            if image_count == limit:
                break

    if args.random:
        print("Random is untested, Shuffling...")
        shuffle_indices = np.random.permutation(limit)
        mapping = dict(zip(np.arange(limit), shuffle_indices))
        folders = [bgr_folder, rgb_folder] + ([median_folder] if args.median else []) \
                  + ([] if args.no_flow else [flow_folder]) + ([bb_folder] if args.bb else [])
        endings = ['.png', '.png', '.npy', '.npy', 'txt']
        for dataset_folder, ending in zip(folders, endings):
            for i, j in mapping.items():
                for file in glob.glob(f'{dataset_folder}/{i:05}*'):
                    os.rename(file, file.replace(f'{i:05}', f'{j:05}') + ".tmp")
            for i, _ in mapping.items():
                for file in glob.glob(f'{dataset_folder}/*'):
                    os.rename(file, file.replace(".tmp", ""))
        print("Shuffling done!")
    print(f"Dataset Generation is completed. Everything is saved in {data_base_folder}.")


def configure(args):
    global env
    # env = AtariARIWrapper(gym.make(f'{arguments.game}Deterministic-v4'))
    with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
        data = f'{json.load(f)}'.replace("'", '"')
        config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    if "Augmented" not in config.game_name:
        print("blue", "box", "You are not using an Augmented environment")
    augmented = "Augmented" in config.game_name
    print(f"Playing {config.game_name}...")
    if config.game_name.lower() == 'coinrun':
        env_name = "ecoinrun"
        env = ProcgenEnv(num_envs=1, env_name=env_name, center_agent=False) # use_backgrounds=False, restrict_themes=True
        state = None
    else:
        env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
                    history_length=config.history_length, max_no_op_actions=3)
        env.augmented = True
        state = env.reset()
        make_deterministic(0 if args.folder == "train" else 1 if args.folder == "validation" else 2, env)
    agent = load_agent(args, env)
    return agent, augmented, state


def create_folders(args, data_base_folder):
    global rgb_folder, bgr_folder, flow_folder, median_folder, bb_folder, vis_folder, mode_folder
    rgb_folder = f"{data_base_folder}/{args.game}-v0/rgb/{args.folder}"
    bgr_folder = f"{data_base_folder}/{args.game}-v0/space_like/{args.folder}"
    bb_folder = f"{data_base_folder}/{args.game}-v0/bb/{args.folder}"
    flow_folder = f"{data_base_folder}/{args.game}-v0/flow/{args.folder}"
    median_folder = f"{data_base_folder}/{args.game}-v0/median/{args.folder}"
    mode_folder = f"{data_base_folder}/{args.game}-v0/mode/{args.folder}"
    vis_folder = f"{data_base_folder}/{args.game}-v0/vis/{args.folder}"
    os.makedirs(bgr_folder, exist_ok=True)
    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(flow_folder, exist_ok=True)
    os.makedirs(median_folder, exist_ok=True)
    os.makedirs(mode_folder, exist_ok=True)
    os.makedirs(bb_folder, exist_ok=True)
    os.makedirs(vis_folder + "/BoundingBox", exist_ok=True)
    os.makedirs(vis_folder + "/Median", exist_ok=True)
    os.makedirs(vis_folder + "/Flow", exist_ok=True)
    os.makedirs(vis_folder + "/Mode", exist_ok=True)


if __name__ == '__main__':
    main()
