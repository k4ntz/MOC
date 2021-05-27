# deep q learning on pong (and later on tennis)
# inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# call with python dq_learning.py --config configs/atari_ball_joint_v1.yaml resume True device 'cpu'

import sys
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import PIL
import cv2

from rtpt import RTPT
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from dqn.dqn import DQN
import dqn.dqn_saver as saver

import argparse

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# space stuff
import os.path as osp

from model import get_model
from engine.utils import get_config
from utils import Checkpointer
from solver import get_optimizers


cfg, task = get_config()

print('Experiment name:', cfg.exp_name)
print('Dataset:', cfg.dataset)
print('Model name:', cfg.model)
print('Resume:', cfg.resume)
if cfg.resume:
    print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'last checkpoint')
print('Using device:', cfg.device)
if 'cuda' in cfg.device:
    print('Using parallel:', cfg.parallel)
if cfg.parallel:
    print('Device ids:', cfg.device_ids)

model = get_model(cfg)
model = model.to(cfg.device)

if len(cfg.gamelist) >= 10:
    print("Using SPACE Model on every game")
    suffix = 'all'
elif len(cfg.gamelist) == 1:
    suffix = cfg.gamelist[0]
    print(f"Using SPACE Model on {suffix}")
elif len(cfg.gamelist) == 2:
    suffix = cfg.gamelist[0] + "_" + cfg.gamelist[1]
    print(f"Using SPACE Model on {suffix}")
else:
    print("Can't train")
    exit(1)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, suffix, cfg.exp_name), max_num=cfg.train.max_ckpt)
use_cpu = 'cpu' in cfg.device
if cfg.resume:
    checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, None, None, use_cpu=cfg.device)
if cfg.parallel:
    model = nn.DataParallel(model, device_ids=cfg.device_ids)


# init env
env = gym.make('Pong-v0')
env.reset()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

### replay memory stuff
Transition = namedtuple('Transition',
                        ('z_where_state', 'z_what_state', 'action', 'next_z_where_state', 'next_z_what_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

### PREPROCESSING

# preprocessing flags
black_bg = getattr(cfg, "train").black_background
dilation = getattr(cfg, "train").dilation

def get_screen():
    # TODO: insert preprocessing from space!
    screen = env.render(mode='rgb_array')
    pil_img = Image.fromarray(screen).resize((128, 128), PIL.Image.BILINEAR)
    # convert image to opencv
    opencv_img = np.asarray(pil_img).copy()
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    if black_bg:
        # get most dominant color
        colors, count = np.unique(opencv_img.reshape(-1,opencv_img.shape[-1]), axis=0, return_counts=True)
        most_dominant_color = colors[count.argmax()]
        # create the mask and use it to change the colors
        bounds_size = 20
        lower = most_dominant_color - [bounds_size, bounds_size, bounds_size]
        upper = most_dominant_color + [bounds_size, bounds_size, bounds_size]
        mask = cv2.inRange(opencv_img, lower, upper)
        opencv_img[mask != 0] = [0,0,0]
    # dilation 
    if dilation:
        kernel = np.ones((3,3), np.uint8)
        opencv_img = cv2.dilate(opencv_img, kernel, iterations=1)
    # convert to tensor
    image_t = torch.from_numpy(opencv_img / 255).permute(2, 0, 1).float()
    return image_t.unsqueeze(0)

def get_z_stuff(model):
    image = get_screen()
    # TODO: treat global_step in a more elegant way
    loss, log = model(image, global_step=100000000)
    # (B, N, 4), (B, N, 1), (B, N, D)
    z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
    return z_where, z_what

env.reset()

### TRAINING


# some hyperparameters

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

liveplot = False

SAVE_EVERY = 5

i_episode = 0
exp_name = "DQ-Learning-Pong-v0"

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

# load if available
if saver.check_loading_model(exp_name):
    # folder and file exists, so load and return
    model_path = saver.model_name(exp_name) 
    print("Loading {}".format(model_path))
    checkpoint = torch.load(model_path)
    policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
    target_net.load_state_dict(checkpoint['target_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    memory = checkpoint['memory']
    i_episode = checkpoint['episode']
else:
    print("No prior checkpoints exists, starting fresh")

steps_done = 0

def select_action(z_where_state, z_what_state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(z_where_state, z_what_state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

# function to plot live while training
def plot_screen(episode, step):
    plt.figure(3)
    plt.clf()
    plt.title('Training - Episode: ' + str(episode) + " - Step: " + str(step))
    plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
    plt.plot()
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# function to plot episode durations
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    tuple1 = tuple(map(lambda s: s is not None,batch.next_z_where_state))
    tuple2 = tuple(map(lambda s: s is not None,batch.next_z_what_state))
    tuple_mask = tuple1 and tuple2
    non_final_mask = torch.tensor(tuple_mask, device=device, dtype=torch.bool)

    non_final_next_z_where_states = torch.cat([s for s in batch.next_z_where_state
                                                if s is not None])
    non_final_next_z_what_states = torch.cat([s for s in batch.next_z_what_state
                                                if s is not None])
    z_where_state_batch = torch.cat(batch.z_where_state)
    z_what_state_batch = torch.cat(batch.z_what_state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(z_where_state_batch, z_what_state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_z_where_states, non_final_next_z_what_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# training loop
num_episodes = 1000

# games wins loses score
wins = 0
loses = 0
last_game = "None"

rtpt = RTPT(name_initials='DV', experiment_name=exp_name,
                max_iterations=num_episodes)
rtpt.start()
while i_episode < num_episodes:
    # reward logger
    pos_reward_count = 0
    neg_reward_count = 0
    # timer stuff
    start = time.perf_counter()
    episode_start = start
    # Initialize the environment and state
    env.reset()
    current_z_where, current_z_what = get_z_stuff(model)
    last_z_where, last_z_what = current_z_where, current_z_what
    z_where_state, z_what_state = current_z_where - last_z_where, current_z_what - last_z_what
    for t in count():
        # timer stuff
        end = time.perf_counter()
        episode_time = end - episode_start
        start = end
        if liveplot:
            plot_screen(i_episode+1, t+1)
        # Select and perform an action
        action = select_action(z_where_state, z_what_state)
        _, reward, done, _ = env.step(action.item())
        if reward > 0:
            pos_reward_count += 1
        if reward < 0:
            neg_reward_count += 1
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_z_where, last_z_what = current_z_where, current_z_what
        current_z_where, current_z_what = get_z_stuff(model)
        if not done:
            next_z_where_state, next_z_what_state = current_z_where - last_z_where, current_z_what - last_z_what
        else:
            next_state = None

        # Store the transition in memory
        memory.push(z_where_state, z_what_state, action, next_z_where_state, next_z_what_state, reward)

        # Move to the next state
        z_where_state, z_what_state = next_z_where_state, next_z_what_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # timer stuff
        end = time.perf_counter()
        step_time = end - start

        if (t) % 10 == 0: # print every 100 steps
            start = time.perf_counter()
            end = time.perf_counter()
            print(
                'exp: {}, episode: {}, step: {}, +reward: {:.2f}, -reward: {:.2f}, s-time: {:.4f}s, e-time: {:.4f}s, w: {}, l:{}, last_game:{}        '.format(
                    exp_name, i_episode + 1, t + 1, pos_reward_count, neg_reward_count,
                    step_time, episode_time, wins, loses, last_game), end="\r")
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    # update wins and loses
    if pos_reward_count >= neg_reward_count:
        wins += 1
        last_game = "Win"
    else:
        loses += 1
        last_game = "Lose"
    # checkpoint saver
    if i_episode % SAVE_EVERY == 0:
        saver.save_models(exp_name, policy_net, target_net, optimizer, memory, i_episode)
    # iterate to next episode
    i_episode += 1
    rtpt.step()

print('Complete')
env.render()
env.close()
