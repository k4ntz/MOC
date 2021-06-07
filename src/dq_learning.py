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
import os

from rtpt import RTPT
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from dqn.dqn import DQN
from dqn.dqn import DuelCNN
import dqn.dqn_saver as saver
import dqn.dqn_logger

import argparse

# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# space stuff
import os.path as osp

from model import get_model
from engine.utils import get_config
from utils import Checkpointer
from solver import get_optimizers


cfg, task = get_config()

# if gpu is to be used
device = cfg.device

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
#if cfg.parallel:
#    model = nn.DataParallel(model, device_ids=cfg.device_ids)


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
                        ('state', 'action', 'next_state', 'reward', 'done'))


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

######## PREPROCESSING ########

i_episode = 0
global_step = 0
video_every = 1

# preprocessing flags
black_bg = getattr(cfg, "train").black_background
dilation = getattr(cfg, "train").dilation

def get_screen():
    screen = env.render(mode='rgb_array')
    pil_img = Image.fromarray(screen).resize((128, 128), PIL.Image.BILINEAR)
    # convert image to opencv
    opencv_img = np.asarray(pil_img).copy()
    # fill video buffer
    if i_episode % video_every == 0:
        logger.fill_video_buffer(opencv_img)
    # convert color
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
    image_t = torch.from_numpy(opencv_img / 255).permute(2, 0, 1).float().to(device)
    return image_t.unsqueeze(0)

# helper function to normalize tensor values to 0-1
def normalize_tensors(t):
    dim1 = t.shape[0]
    dim2 = t.shape[1]
    dim3 = t.shape[2]
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    t = t.view(dim1, dim2, dim3)
    return t

boxes_len = 16

# helper function to process a single frame of z stuff
def process_z_stuff(z_where, z_pres_prob, z_what):
    z_stuff = torch.zeros_like(torch.rand((5, 4)), device=device)
    z_where = z_where.to(device)
    z_pres_prob = z_pres_prob.to(device)
    z_what = z_what.to(device)
    # clean up
    torch.cuda.empty_cache()
    ## nullize all z whats with z pres < 0.5
    z_pres = (z_pres_prob.detach().cpu().squeeze() > 0.5)
    z_what_pres = z_what[z_pres]
    ## same with z where 
    z_where_pres = z_where[z_pres]
    # get coordinates
    coord_x = torch.FloatTensor([i % boxes_len for i, x in enumerate(z_pres) if x]).to(device)
    coord_y = torch.FloatTensor([math.floor(i / boxes_len) for i, x in enumerate(z_pres) if x]).to(device)
    # normalize z where centers to [0:1], add coordinates to its center values and normalize again
    z_where_pres[:, 2] = (((z_where_pres[:, 2] + 1.0) / 2.0) + coord_x) / boxes_len
    z_where_pres[:, 3] = (((z_where_pres[:, 3] + 1.0) / 2.0) + coord_y) / boxes_len
    # combine z what pres with z where tensors
    #z_combined = torch.cat((z_where_pres, z_what_pres), 1)
    #first_dim = min(z_combined.shape[0], z_stuff.shape[0])
    first_dim = min(z_where_pres.shape[0], z_stuff.shape[0])
    z_stuff[:first_dim] = z_where_pres[:first_dim]
    z_stuff = z_stuff.unsqueeze(0).cpu()
    return z_stuff


# use SPACE model
def get_z_stuff(image):
    # TODO: treat global_step in a more elegant way
    with torch.no_grad():
        loss, log = model(image, global_step=100000000)
        # (B, N, 4), (B, N, 1), (B, N, D)
        z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
        return process_z_stuff(z_where[0], z_pres_prob[0], z_what[0])
    return None

env.reset()

######## TRAINING ########

# some hyperparameters

BATCH_SIZE = 128
GAMMA = 0.97
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 100000
# TARGET_UPDATE = 1000
lr = 0.00025

USE_SPACE = False

liveplot = False
DEBUG = False

SAVE_EVERY = 5

num_episodes = 1000

skip_frames = 1

MEMORY_SIZE = 50000
MEMORY_MIN_SIZE = 25000

# for debugging nn stuff
if DEBUG:
    EPS_START = EPS_END
    BATCH_SIZE = 12
    MEMORY_MIN_SIZE = BATCH_SIZE


exp_name = "DQ-Learning-Pong-v7-cnn"

# init tensorboard
log_path = os.getcwd() + "/dqn/logs/"
log_name = exp_name
log_steps = 500
logger = dqn.dqn_logger.DQN_Logger(log_path, exp_name)

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = None
target_net = None
if USE_SPACE:
    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
else:
    policy_net = DuelCNN(128, 128, n_actions).to(device)
    target_net = DuelCNN(128, 128, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=lr)
memory = ReplayMemory(MEMORY_SIZE)

total_max_q = 0
total_loss = 0

# load if available
if saver.check_loading_model(exp_name):
    # folder and file exists, so load and return
    model_path = saver.model_name(exp_name) 
    print("Loading {}".format(model_path))
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
    target_net.load_state_dict(checkpoint['target_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    memory = checkpoint['memory']
    i_episode = checkpoint['episode']
    global_step = checkpoint['global_step']
    total_max_q = checkpoint['total_max_q']
    total_loss = checkpoint['total_loss']
else:
    print("No prior checkpoints exists, starting fresh")

# helper function to get state, whether to use SPACE or not
def get_state():
    if USE_SPACE:
        return get_z_stuff(get_screen())
    else:
        screen = env.render(mode='rgb_array')
        pil_img = Image.fromarray(screen).resize((128, 128), PIL.Image.BILINEAR)
        # convert image to opencv
        opencv_img = np.asarray(pil_img).copy()
        # fill video buffer
        if i_episode % video_every == 0:
            logger.fill_video_buffer(opencv_img)
        # convert color
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(opencv_img / 255).float().to(device)


def select_action(state):
    sample = random.random()
    eps_threshold = EPS_START
    if global_step > MEMORY_MIN_SIZE:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * (global_step - MEMORY_MIN_SIZE) / EPS_DECAY)
    # log eps_treshold
    if global_step % log_steps == 0:
        logger.log_eps(eps_threshold, global_step)
    if sample > eps_threshold or liveplot:
        with torch.no_grad():

            state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []

def optimize_model():
    # logging variables
    global total_max_q
    global total_loss


    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    zip(*transitions) unzips the transitions into
    Transition(*) creates new named tuple
    batch.state - tuple of all the states (each state is a tensor)
    batch.next_state - tuple of all the next states (each state is a tensor)
    batch.reward - tuple of all the rewards (each reward is a float)
    batch.action - tuple of all the actions (each action is an int)    
    """
    batch = Transition(*zip(*transitions))
    
    # Convert them to tensors
    state = torch.from_numpy(np.array(batch.state)).float().to(device)
    next_state = torch.from_numpy(np.array(batch.next_state)).float().to(device)
    action = torch.cat(batch.action, dim=0).to(device).squeeze(1)
    reward = torch.cat(batch.reward, dim=0).to(device)
    done = torch.tensor(batch.done, dtype=torch.float, device=device)

    # Make predictions
    state_q_values = policy_net(state)
    next_states_q_values = policy_net(next_state)
    next_states_target_q_values = target_net(next_state)
    # Find selected action's q_value
    selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    # Get indice of the max value of next_states_q_values
    # Use that indice to get a q_value from next_states_target_q_values
    # We use greedy for policy So it called off-policy
    next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
    # Use Bellman function to find expected q value
    expected_q_value = reward + GAMMA * next_states_target_q_value * (1 - done)
    
    # Calc loss with expected_q_value and q_value
    loss = F.mse_loss(selected_q_value, expected_q_value.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # log loss and max q
    max_q = torch.max(state_q_values).item()
    total_max_q += max_q
    with torch.no_grad():
        total_loss += loss
    # log metrics for last log_steps and reset
    if global_step % log_steps == 0:
        logger.log_max_q(total_max_q/log_steps, global_step)
        total_max_q = 0
        logger.log_loss(total_loss/log_steps, global_step)
        total_loss = 0

### plot stuff 

# function to plot live while training
def plot_screen(episode, step):
    plt.figure(3)
    plt.clf()
    plt.title('Training - Episode: ' + str(episode) + " - Step: " + str(step))
    plt.imshow(env.render(mode='rgb_array'),
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
        

### training loop

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
    episode_time = 0
    # logger stuff
    episode_steps = 0
    # Initialize the environment and state
    env.reset()
    # get z stuff for init
    state = get_state()
    state = np.stack((state, state, state, state))
    # last done action
    action_item = None
    action = None
    for t in count():
        # TODO: REMOVE
        #debugs = time.perf_counter()
        global_step += 1
        # timer stuff
        end = time.perf_counter()
        episode_time = end - episode_start
        start = end
        if liveplot:
            plot_screen(i_episode+1, t+1)
        # If skipped frames are over, select action
        if action_item is None or global_step % skip_frames == 0:
            action = select_action(state)
            action_item = action.item()
        # perform action
        _, reward, done, _ = env.step(action_item)
        
        if reward > 0:
            pos_reward_count += 1
        if reward < 0:
            neg_reward_count += 1
        reward = torch.tensor([reward], device=device)

        # if skip frames are over, observe new state
        next_state = get_state()
        next_state = np.stack((next_state, state[0], state[1], state[2]))
        # Store the transition in memory
        memory.push(state, action, next_state, reward, done)
        # Move to the next state
        state = next_state
        # TODO: REMOVE
        #print(time.perf_counter() - debugs)

        # Perform one step of the optimization (on the policy network)
        if len(memory) > MEMORY_MIN_SIZE:
            optimize_model()
        elif global_step % log_steps == 0:
            logger.log_loss(0, global_step)
            logger.log_max_q(0, global_step)

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
            episode_steps = t + 1
            #plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if len(memory) > MEMORY_MIN_SIZE:
        target_net.load_state_dict(policy_net.state_dict())
    # update wins and loses
    if pos_reward_count >= neg_reward_count:
        wins += 1
        last_game = "Win"
    else:
        loses += 1
        last_game = "Lose"
    # log episode
    logger.log_episode(episode_steps, pos_reward_count, neg_reward_count, i_episode, global_step)
    # if video step, create video
    if i_episode % video_every == 0:
        logger.save_video(exp_name)
    # iterate to next episode
    i_episode += 1
    # checkpoint saver
    if i_episode % SAVE_EVERY == 0:
        saver.save_models(exp_name, policy_net, target_net, optimizer, memory, i_episode, global_step, total_max_q, total_loss)
    rtpt.step()

print('Complete')
env.render()
env.close()
