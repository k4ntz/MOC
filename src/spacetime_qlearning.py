import joblib
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.distributions import Categorical
from rtpt import RTPT
from engine.utils import get_config
from model import get_model
from utils import Checkpointer
from solver import get_optimizers
from PIL import Image
from torchvision import transforms

# load config
cfg, task = get_config()

# init env stuff
cfg.device_ids = [0]
env_name = cfg.gamelist[0]
env = gym.make(env_name)
env.reset()
n_actions = env.action_space.n

# get models
# TODO: make dynamic feature length
spacetime_model = get_model(cfg)
i_episode = 0
# move to cuda when possible
use_cuda = 'cuda' in cfg.device
if use_cuda:
    spacetime_model = spacetime_model.to('cuda:0')
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,
                            load_time_consistency=cfg.load_time_consistency, add_flow=cfg.add_flow)
optimizer_fg, optimizer_bg = get_optimizers(cfg, spacetime_model)
if cfg.resume:
    checkpoint = checkpointer.load_last(cfg.resume_ckpt, spacetime_model, optimizer_fg, optimizer_bg, cfg.device)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step'] + 1

space = spacetime_model.space
z_classifier_path = f"classifiers/{cfg.exp_name}_z_what_classifier.joblib.pkl"
z_classifier = joblib.load(z_classifier_path)
# x is the image on device as a Tensor, z_classifier accepts the latents,
# only_z_what control if only the z_what latent should be used (see docstring)
transformation = transforms.ToTensor()


# helper function to filter extracted scene from spacetime
def clean_scene(scene):
    empty_keys = []
    for key, val in scene.items():
        for i, z_where in reversed(list(enumerate(val))):
            if z_where[3] < -0.75:
                scene[key].pop(i)
        if len(val) == 0:
            empty_keys.append(key)
    for key in empty_keys:
        scene.pop(key)
    scene_list = []
    for el in [1, 2, 3]:
        if el in scene:
            scene_list.append(scene[el][0][2:])
        else:
            scene_list.append([0, 0]) # object not found
    return scene_list


# helper function to discretize
def discretize(state):
    mulstate = state * 20
    intstate = tuple(list(mulstate.astype(int)))
    return intstate


# helper function to get scene
def get_scene(observation, space):
    img = Image.fromarray(observation[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
    #x = torch.moveaxis(torch.tensor(np.array(img)), 2, 0)
    x = transformation(img)
    if use_cuda:
        x = x.cuda()
    scene = space.scene_description(x, z_classifier=z_classifier,
                                    only_z_what=True)  # class:[(w, h, x, y)]
    scene_list = clean_scene(scene)
    return scene_list


#Initializing the Q-table as dictionary and parameters
Q = {}
gamma =  0.97
eps_start = 1.0
eps_end = 0.01
eps_decay = 100000
learning_rate = 0.05
# calc len of all possible states = all possible position combinations
state_len = env.observation_space.shape[0] * 2 \
        + env.observation_space.shape[1] \
        + env.observation_space.shape[0]


# function to select action by given state
def select_action(state, global_step):
    if state is not None:
        sample = random.random()
        eps_threshold = eps_start
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * global_step / eps_decay)
        if sample > eps_threshold:
            return np.argmax(Q[state])
    return random.randrange(n_actions)


max_episode = 50000
# episode loop
running_reward = -999
# training loop
rtpt = RTPT(name_initials='DV/QT/TR', experiment_name=cfg.exp_name + "-QL",
                max_iterations=max_episode)
rtpt.start()
while i_episode < max_episode:
    state, ep_reward = env.reset(), 0
    action = np.random.randint(n_actions)
    state = tuple([0,0,0,0,0,0])
    if not state in Q.keys():
        # entry doesnt exists, fill entry with 0
        Q[state] = np.zeros(n_actions)
    # env step loop
    for t in range(1, 10000):  # Don't infinite loop while learning
        # select action
        action = select_action(state, t)
        # do action and observe
        observation, reward, done, info = env.step(action)
        ep_reward += reward
        scene_list = get_scene(observation, space)
        # flatten scene list
        object_list = np.asarray([item for sublist in scene_list for item in sublist])
        next_state = discretize(object_list)
        # update qtable
        if not next_state in Q.keys():
            # entry doesnt exists, fill entry with 0
            Q[next_state] = np.zeros(n_actions)
        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * \
                np.max(Q[next_state]) - Q[state][action])
        # finish step
        print('Episode: {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tTable-Len: {}\tSteps: {}       '.format(
            i_episode, ep_reward, running_reward, len(Q), t), end="\r")
        state = next_state
        if done:
            break
    # replace first running reward with last reward for loaded models
    if running_reward == -999:
        running_reward = ep_reward
    else:
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    # finish episode
    i_episode += 1
    rtpt.step()
