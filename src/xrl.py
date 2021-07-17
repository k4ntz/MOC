# q learning with given shallow representation of atari game

import gym
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
import cv2
import os
import pandas as pd
import seaborn as sn

from itertools import count
from PIL import Image
from atariari.benchmark.wrapper import AtariARIWrapper

from rtpt import RTPT

import dqn.dqn_logger
import dqn.utils as utils

cfg, _ = utils.get_config()

# function to plot live while training
def plot_screen(env, episode, step):
    plt.figure(3)
    plt.clf()
    plt.title('Training - Episode: ' + str(episode) + " - Step: " + str(step))
    plt.imshow(env.render(mode='rgb_array'),
           interpolation='none')
    plt.plot()
    plt.pause(0.0001)  # pause a bit so that plots are updated

# helper function to calc linear equation
def get_target_x(x1, x2, y1, y2, player_x):
    x = [x1, x2]
    y = [y1, y2]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # now calc target pos
    # y = mx + c
    return np.int16(m * player_x + c)

# helper function to convert env info into custom list
# state contains player x, y, ball x, y, oldplayer x, y, oldball x, y, 
# features are processed stuff for policy
def preprocess_state(env_info, last_state=None):
    features = []
    labels = env_info["labels"]
    player_x = labels["player_x"].astype(np.int16)
    player_y = labels["player_y"].astype(np.int16)
    enemy_x = labels["enemy_x"].astype(np.int16)
    enemy_y = labels["enemy_y"].astype(np.int16)
    ball_x = labels["ball_x"].astype(np.int16)
    ball_y = labels["ball_y"].astype(np.int16)
    # set new state
    state = last_state
    if state is None:
        state = [player_x, player_y, ball_x, ball_y, 
            np.int16(0), np.int16(0), np.int16(0), np.int16(0)]
        features.append(0)
    else:
        # refresh state with new coordiates
        state[4] = state[0]
        state[5] = state[1]
        state[6] = state[2]
        state[7] = state[3]
        state[0] = player_y
        state[1] = player_y
        state[2] = ball_x
        state[3] = ball_y   
        # calc target point and put distance in features
        target_y = get_target_x(state[6], ball_x, state[7], ball_y, player_x) 
        features.append(target_y - player_y)
    # append other distances
    features.append(ball_x - player_x) # distance x ball and player
    features.append(ball_y - player_y) # distance y ball and player
    # euclidean distance between old and new ball coordinates to represent current speed per frame
    features.append(math.sqrt((ball_x - state[6])**2 + (ball_y - state[7])**2)) 
    return state, features

# helper function to get state
def get_state(env, action=1, state=None):
    obs, reward, done, info = env.step(action)
    state, features = preprocess_state(info, state)
    return state, features, reward, done

# function to select action by given state
# 0: "NOOP",
# 1: "FIRE",
# 2: "UP",
# 3: "RIGHT",
# 4: "LEFT",
# 5: "DOWN",
def select_action(n_actions, features, global_step, logger):
    disty = features[0]
    buffery = 6
    if disty > buffery:
        return 5
    elif disty < -buffery:
        return 2
    else:
        return 0

# train main function
def train():
    print('Experiment name:', cfg.exp_name)
    print('Loading', cfg.env_name)
    env = AtariARIWrapper(gym.make(cfg.env_name))
    _ = env.reset()
    _, _, done, info = env.step(1)

    i_episode = 0
    global_step = 0
    video_every = cfg.video_steps

    LIVEPLOT = cfg.liveplot
    DEBUG = cfg.debug
    print('Liveplot:', LIVEPLOT)
    print('Debug Mode:', DEBUG)

    exp_name = cfg.exp_name
    num_episodes = cfg.train.num_episodes
    log_steps = cfg.train.log_steps

    # init tensorboard
    log_path = os.getcwd() + cfg.logdir
    log_name = exp_name
    logger = dqn.dqn_logger.DQN_Logger(log_path, exp_name, vfolder="/xrl/video/")

    # Get number of actions from gym action space
    n_actions = env.action_space.n

    #Getting the state space
    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))
    print("State {}".format(info))

    i_episode = 0

    rtpt = RTPT(name_initials='DV', experiment_name=exp_name,
                    max_iterations=num_episodes)
    rtpt.start()
    # train mode
    if cfg.mode == "train":
        while i_episode < num_episodes:
            obs = env.reset()
            state, features, _, _ = get_state(env)
            # timer stuff
            start = time.perf_counter()
            episode_start = start
            episode_time = 0
            # track for logging
            episode_steps = 0
            pos_reward_count = 0
            neg_reward_count = 0
            for t in count():
                # logging stuff
                global_step += 1
                # timer stuff
                end = time.perf_counter()
                episode_time = end - episode_start
                start = end
                # select action
                action = select_action(n_actions, features, global_step, logger)
                #Taking the action and getting the reward and outcome state
                next_state, features, current_reward, done = get_state(env, action, state)
                #TODO: optimize??
                state = next_state
                # for logging
                if current_reward > 0:
                    pos_reward_count += 1
                if current_reward < 0:
                    neg_reward_count += 1
                # timer stuff
                end = time.perf_counter()
                step_time = end - start
                if LIVEPLOT:
                    plot_screen(env, i_episode+1, t+1)
                if done:
                    print(
                        'exp: {}, episode: {}, step: {}, +reward: {:.2f}, -reward: {:.2f}, s-time: {:.4f}s, e-time: {:.4f}s        '.format(
                            exp_name, i_episode + 1, t + 1, pos_reward_count, neg_reward_count, 
                            step_time, episode_time), end="\r")
                    episode_steps = t + 1
                    break
            # log episode
            if i_episode % 20 == 0:
                #logger.log_episode(episode_steps, pos_reward_count, neg_reward_count, i_episode, global_step, len(Q))
                None
            # iterate to next episode
            i_episode += 1
            # checkpoint saver
            if i_episode % cfg.train.save_every == 0:
                #TODO: saver?
                None
            rtpt.step()
    # now eval
    if cfg.mode == "eval":
        print("Evaluating...")
        # Initialize the environment and state
        obs = env.reset()
        state, _, _ = get_state(env)
        pos_reward_count = 0
        neg_reward_count = 0
        for t in count():
            # log current frame
            screen = env.render(mode='rgb_array')
            pil_img = Image.fromarray(screen).resize((128, 128), PIL.Image.BILINEAR)
            # convert image to opencv
            opencv_img = np.asarray(pil_img).copy()
            # fill video buffer
            if i_episode % video_every == 0:
                logger.fill_video_buffer(opencv_img)
            # select action
            action = select_action(n_actions, state, global_step, logger)
            #Taking the action and getting the reward and outcome state
            next_state, current_reward, done = get_state(env, action)
            state = next_state
            # for logging
            if current_reward > 0:
                pos_reward_count += 1
            if current_reward < 0:
                neg_reward_count += 1
            if done:
                print(
                    'exp: {}, step: {}, +reward: {:.2f}, -reward: {:.2f}        '.format(
                        exp_name, t + 1, pos_reward_count, neg_reward_count))
                episode_steps = t + 1
                break
        logger.save_video(exp_name)


if __name__ == '__main__':
    train()
