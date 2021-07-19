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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from itertools import count
from PIL import Image
from atariari.benchmark.wrapper import AtariARIWrapper

from rtpt import RTPT

import dqn.dqn_logger
import dqn.utils as utils

cfg, _ = utils.get_config()

# define policy network
class policy_net(nn.Module):
    def __init__(self, nS, nH, nA): # nS: state space size, nH: n. of neurons in hidden layer, nA: size action space
        super(policy_net, self).__init__()
        self.h = nn.Linear(nS, nH)
        self.out = nn.Linear(nH, nA)

    # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x


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
# raw_features contains player x, y, ball x, y, oldplayer x, y, oldball x, y, 
# features are processed stuff for policy
def preprocess_raw_features(env_info, last_raw_features=None):
    features = []
    norm_factor = 250
    # extract raw features
    labels = env_info["labels"]
    player_x = labels["player_x"].astype(np.int16)
    player_y = labels["player_y"].astype(np.int16)
    enemy_x = labels["enemy_x"].astype(np.int16)
    enemy_y = labels["enemy_y"].astype(np.int16)
    ball_x = labels["ball_x"].astype(np.int16)
    ball_y = labels["ball_y"].astype(np.int16)
    # set new raw_features
    raw_features = last_raw_features
    if raw_features is None:
        raw_features = [player_x, player_y, ball_x, ball_y, enemy_x, enemy_y
            ,np.int16(0), np.int16(0), np.int16(0), np.int16(0), np.int16(0), np.int16(0)]
        features.append(0)
    else:
        # move up old values in list
        raw_features = np.roll(raw_features, 6)
        raw_features[0] = player_y
        raw_features[1] = player_y
        raw_features[2] = ball_x
        raw_features[3] = ball_y  
        raw_features[4] = enemy_x
        raw_features[5] = enemy_y 
        # calc target point and put distance in features
        target_y = get_target_x(raw_features[6], ball_x, raw_features[7], ball_y, player_x) 
        features.append((target_y - player_y)/ norm_factor)
    # append other distances
    features.append((player_x - ball_x)/ norm_factor)# distance x ball and player
    features.append((player_y - ball_y)/ norm_factor)# distance y ball and player
    features.append((player_x - enemy_y)/ norm_factor) # distance x player and enemy
    features.append((player_y - enemy_y)/ norm_factor) # distance y player and enemy
    # euclidean distance between old and new ball coordinates to represent current speed per frame
    features.append(math.sqrt((ball_x - raw_features[8])**2 + (ball_y - raw_features[9])**2) / 25) 
    return raw_features, features

# helper function to get features
def get_features(env, action=1, last_raw_features=None):
    if action == 1:
        action = 2
    elif action == 2:
        action = 5
    obs, reward, done, info = env.step(action)
    raw_features, features = preprocess_raw_features(info, last_raw_features)
    return raw_features, features, reward, done

# function to select action by given features
# 0: "NOOP",
# 1: "FIRE",
# 2: "UP",
# 3: "RIGHT",
# 4: "LEFT",
# 5: "DOWN",
def select_action(n_actions, features, policy, global_step, logger):
    # calculate probabilities of taking each action
    probs = policy(torch.tensor(features).unsqueeze(0).float())
    # sample an action from that set of probs
    sampler = Categorical(probs)
    action = sampler.sample()
    # return action
    return action

# train main function
def train():
    print('Experiment name:', cfg.exp_name)
    print('Loading', cfg.env_name)
    env = AtariARIWrapper(gym.make(cfg.env_name))
    _ = env.reset()
    _, _, done, info = env.step(1)

    # instantiate the policy: #input_shape #hidden_node_size #action_size
    policy = policy_net(6, 32, 3) #env.action_space.n
    # create an optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.train.learning_rate)

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
            # init lists
            rewards = []
            actions = []
            states  = []
            # reset env
            obs = env.reset()
            raw_features, features, _, _ = get_features(env)
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
                action = select_action(n_actions, features, policy, global_step, logger)
                #Taking the action and getting the reward and outcome raw_features
                raw_features, features, reward, done = get_features(env, action, raw_features)
                #buffer features actions and reward
                states.append(features)
                actions.append(action)
                rewards.append(reward)
                # for logging
                if reward > 0:
                    pos_reward_count += 1
                if reward < 0:
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
            # optimize after done
            # preprocess rewards
            rewards = np.array(rewards)
            # calculate rewards to go for less variance
            R = torch.tensor([np.sum(rewards[i:] * (cfg.train.gamma ** np.array(range(i, len(rewards))))) for i in range(len(rewards))])
            # or uncomment following line for normal rewards
            #R = torch.sum(torch.tensor(rewards))

            # preprocess states and actions
            states = torch.tensor(states).float()
            actions = torch.tensor(actions)

            # calculate gradient
            probs = policy(states)
            sampler = Categorical(probs)
            log_probs = -sampler.log_prob(actions)   # "-" because it was built to work with gradient descent, but we are using gradient ascent
            pseudo_loss = torch.sum(log_probs * R) # loss that when differentiated with autograd gives the gradient of J(θ)
            # update policy weights
            optimizer.zero_grad()
            pseudo_loss.backward()
            optimizer.step()
            # log episode
            if i_episode % 20 == 0:
                logger.log_episode(episode_steps, pos_reward_count, neg_reward_count, i_episode, global_step)
                None
            # iterate to next episode
            i_episode += 1
            # checkpoint saver
            if i_episode % cfg.train.save_every == 0:
                #TODO: saver?
                None
            rtpt.step()
    # now eval
    # TODO: still buggy
    if cfg.mode == "eval":
        print("Evaluating...")
        # Initialize the environment and state
        obs = env.reset()
        raw_features, features, _, _ = get_features(env)
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
            action = select_action(n_actions, features, policy, global_step, logger)
            #Taking the action and getting the reward and outcome state
            raw_features, features, reward, done = get_features(env, action, raw_features)
            # for logging
            if reward > 0:
                pos_reward_count += 1
            if reward < 0:
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
