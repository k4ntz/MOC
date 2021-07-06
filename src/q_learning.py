# q learning with given shallow representation of atari game

import sys
import gym
import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
import ql.ql_saver as saver

import argparse


cfg, _ = utils.get_config()

print('Experiment name:', cfg.exp_name)
print('Loading', cfg.env_name)
env = AtariARIWrapper(gym.make(cfg.env_name))
obs = env.reset()
obs, reward, done, info = env.step(1)

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
logger = dqn.dqn_logger.DQN_Logger(log_path, exp_name, vfolder="/ql/video/")

# Get number of actions from gym action space
n_actions = env.action_space.n

#Getting the state space
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
print("State {}".format(info))

# helper function to convert env info into custom list
def convert_to_state(env_info):
    state = []
    divide_factor = 10
    labels = env_info["labels"]
    state.append(int(labels["player_y"]/divide_factor))
    state.append(int(labels["enemy_y"]/divide_factor))
    state.append(int(labels["ball_x"]/divide_factor))
    state.append(int(labels["ball_y"]/divide_factor))
    return str(state)

# helper function to get state
def get_state(Q, action = 1):
    obs, reward, done, info = env.step(action)
    state = convert_to_state(info)
    if not state in Q.keys():
        # entry doesnt exists, fill entry with 0
        Q[state] = np.zeros(n_actions)
    return Q, state, reward, done

# calc len of all possible states = all possible position combinations
state_len = env.observation_space.shape[0] * 2 \
        + env.observation_space.shape[1] \
        + env.observation_space.shape[0]

#Initializing the Q-table as dictionary
Q = {}

# function to select action by given state
def select_action(state, global_step, logger):
    sample = random.random()
    eps_threshold = cfg.train.eps_start
    eps_threshold = cfg.train.eps_end + (cfg.train.eps_start - cfg.train.eps_end) * \
        math.exp(-1. * global_step / cfg.train.eps_decay)
    # log eps_treshold
    if global_step % log_steps == 0 and cfg.mode == "train":
        logger.log_eps(eps_threshold, global_step)
    if sample > eps_threshold or cfg.mode == "eval":
        return np.argmax(Q[state])
    else:
        return random.randrange(n_actions)

i_episode = 0

# check if q table loading is not null
tmp_Q, tmp_i_episode, tmp_global_step = saver.load_qtable(exp_name)
if tmp_Q is not None:
    Q = tmp_Q
    i_episode = tmp_i_episode
    global_step = tmp_global_step

rtpt = RTPT(name_initials='DV', experiment_name=exp_name,
                max_iterations=num_episodes)
rtpt.start()
# train mode
if cfg.mode == "train":
    while i_episode < num_episodes:
        obs = env.reset()
        Q, state, _, _ = get_state(Q)
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
            action = select_action(state, global_step, logger)
            #Taking the action and getting the reward and outcome state
            Q, next_state, current_reward, done = get_state(Q, action)
            Q[state][action] = Q[state][action] + cfg.train.learning_rate * (current_reward + cfg.train.gamma * \
                np.max(Q[next_state]) - Q[state][action])
            state = next_state
            # for logging
            if current_reward > 0:
                pos_reward_count += 1
            if current_reward < 0:
                neg_reward_count += 1
            # timer stuff
            end = time.perf_counter()
            step_time = end - start
            #TODO: Remove
            #if t % 60 == 0:
            #    logger.save_video(exp_name)
            #if (t) % 10 == : # print every 10 steps
                #print(
                #    'exp: {}, episode: {}, step: {}, +reward: {:.2f}, -reward: {:.2f}, s-time: {:.4f}s, e-time: {:.4f}s        '.format(
                #        exp_name, i_episode + 1, t + 1, pos_reward_count, neg_reward_count,
                #        step_time, episode_time), end="\r")
            if done:
                print(
                    'exp: {}, episode: {}, step: {}, +reward: {:.2f}, -reward: {:.2f}, s-time: {:.4f}s, e-time: {:.4f}s, Q-table size: {}        '.format(
                        exp_name, i_episode + 1, t + 1, pos_reward_count, neg_reward_count,
                        step_time, episode_time, len(Q)), end="\r")
                episode_steps = t + 1
                #plot_durations()
                break
        # log episode
        if i_episode % 10 == 0:
            logger.log_episode(episode_steps, pos_reward_count, neg_reward_count, i_episode, global_step, len(Q))
        # iterate to next episode
        i_episode += 1
        # checkpoint saver
        if i_episode % cfg.train.save_every == 0:
            saver.save_qtable(exp_name, Q, i_episode, global_step)
        rtpt.step()
# now eval
if cfg.mode == "eval":
    print("Evaluating...")
    # Initialize the environment and state
    obs = env.reset()
    Q, state, _, _ = get_state(Q)
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
        action = select_action(state, global_step, logger)
        #Taking the action and getting the reward and outcome state
        Q, next_state, current_reward, done = get_state(Q, action)
        state = next_state
        # for logging
        if current_reward > 0:
            pos_reward_count += 1
        if current_reward < 0:
            neg_reward_count += 1
        if done:
            print(
                'exp: {}, step: {}, +reward: {:.2f}, -reward: {:.2f}, Q-table size: {}        '.format(
                    exp_name, t + 1, pos_reward_count, neg_reward_count, len(Q)))
            episode_steps = t + 1
            #plot_durations()
            break
    logger.save_video(exp_name)
    # 0: "NOOP",
    # 1: "FIRE",
    # 2: "UP",
    # 3: "RIGHT",
    # 4: "LEFT",
    # 5: "DOWN",
    Q_df = pd.DataFrame.from_dict(Q, orient='index')
    Q_df = Q_df.rename(columns={Q_df.columns[2]: 'up'})
    Q_df = Q_df.rename(columns={Q_df.columns[5]: 'down'})
    Q_df["States"] = Q_df.index
    Q_df = Q_df.reset_index(drop=True)
    # preprocess col to get seperate values
    Q_df["States"] = Q_df["States"].str.replace(" ", "")
    Q_df["States"] = Q_df["States"].str.replace("[", "")
    Q_df["States"] = Q_df["States"].str.replace("]", "")
    cols = ["player_y", "enemy_y", "ball_x", "ball_y"]
    Q_df[cols] = Q_df["States"].str.split(",", expand=True)
    Q_df[cols] = Q_df[cols].apply(pd.to_numeric, errors='coerce')
    # combine up and down to plot both
    Q_df["combinedQ"] = Q_df.iloc[:,2] - Q_df.iloc[:,5] 
    print(Q_df.head(5))
    print(Q_df.describe(include='all'))
    
    plt.rcParams.update({'font.size': 5})
    # plot heatmap for ball position
    fig = plt.figure()
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)
    ax7 = fig.add_subplot(247)
    ax8 = fig.add_subplot(248)
    ax1.hist2d(Q_df["ball_x"], Q_df["up"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax1.set_title('Ball-X and Up Histogram')
    ax1.set_xlabel('ball_x')
    ax1.set_ylabel('up')
    ax2.hist2d(Q_df["ball_x"], Q_df["down"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax2.set_title('Ball-X and Down Histogram')
    ax2.set_xlabel('ball_x')
    ax2.set_ylabel('down')
    ax3.hist2d(Q_df["ball_y"], Q_df["up"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax3.set_title('Ball-Y and Up Histogram')
    ax3.set_xlabel('ball_y')
    ax3.set_ylabel('up')
    ax4.hist2d(Q_df["ball_y"], Q_df["down"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax4.set_title('Ball-Y and Down Histogram')
    ax4.set_xlabel('ball_y')
    ax4.set_ylabel('down')
    ax5.hist2d(Q_df["player_y"], Q_df["up"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax5.set_title('Player-Y and Up Histogram')
    ax5.set_xlabel('player_y')
    ax5.set_ylabel('up')
    ax6.hist2d(Q_df["player_y"], Q_df["down"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax6.set_title('Player-Y and Down Histogram')
    ax6.set_xlabel('player_y')
    ax6.set_ylabel('down')
    ax7.hist2d(Q_df["enemy_y"], Q_df["up"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax7.set_title('enemy-Y and Up Histogram')
    ax7.set_xlabel('enemy_y')
    ax7.set_ylabel('up')
    ax8.hist2d(Q_df["enemy_y"], Q_df["down"], norm=matplotlib.colors.LogNorm(), bins=(20, 100))
    ax8.set_title('enemy-Y and Down Histogram')
    ax8.set_xlabel('enemy_y')
    ax8.set_ylabel('down')
    plt.show()

    corrMatrix = Q_df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()
    

        

