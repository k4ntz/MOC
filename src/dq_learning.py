# deep q learning on pong (and later on tennis)
# inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# call with:
# python dq_learning.py --config dqn/configs/DQ-Learning-Tennis-v11r.yaml --space-config configs/atari_ball_joint_v1.yaml resume True device 'cpu'

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
from pytorch_grad_cam import GradCAM
import PIL
import cv2
import os

from rtpt import RTPT
import time

import torch
import torch.nn as nn
import torchvision.transforms as T

#from mushroom_rl.algorithms.value import DQN, DoubleDQN, CategoricalDQN
#from mushroom_rl.algorithms.value import DuelingDQN
#from mushroom_rl.approximators.parametric import TorchApproximator
#from mushroom_rl.core import Core
#from mushroom_rl.environments import Atari
#from mushroom_rl.policy import EpsGreedy
#from mushroom_rl.utils.parameters import Parameter, ExponentialParameter
#from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory as PriorityReplay

import dqn.dqn_saver as saver
import dqn.dqn_logger
import dqn.dqn_agent as dqn_agent
import dqn.utils as utils

import argparse

# space stuff
import os.path as osp

from model import get_model
from utils import Checkpointer
from solver import get_optimizers

cfg, space_cfg = utils.get_config()

# if gpu is to be used
device = cfg.device

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

print('Experiment name:', cfg.exp_name)
print('Resume:', space_cfg.resume)
if space_cfg.resume:
    print('Checkpoint:', space_cfg.resume_ckpt if space_cfg.resume_ckpt else 'last checkpoint')
print('Using device:', cfg.device)
if 'cuda' in cfg.device:
    print('Using parallel:', cfg.parallel)
if cfg.parallel:
    print('Device ids:', cfg.device_ids)

model = get_model(space_cfg)
model = model.to(cfg.device)

if len(space_cfg.gamelist) == 1:
    suffix = space_cfg.gamelist[0]
    print(f"Using SPACE Model on {suffix}")
elif len(space_cfg.gamelist) == 2:
    suffix = space_cfg.gamelist[0] + "_" + cfg.gamelist[1]
    print(f"Using SPACE Model on {suffix}")
else:
    print("Can't train")
    exit(1)
checkpointer = Checkpointer(osp.join(space_cfg.checkpointdir, suffix, space_cfg.exp_name), max_num=4)
use_cpu = 'cpu' in cfg.device

checkpoint = checkpointer.load_last(space_cfg.resume_ckpt, model, None, None, use_cpu=cfg.device)
#if cfg.parallel:
#    model = nn.DataParallel(model, device_ids=cfg.device_ids)

# init env
env = gym.make(cfg.env_name)
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
video_every = cfg.video_steps

# preprocessing flags
black_bg = getattr(cfg, "train").black_background
dilation = getattr(cfg, "train").dilation

def get_screen():
    screen = env.render(mode='rgb_array')
    pil_img = Image.fromarray(screen).resize((128, 128), PIL.Image.BILINEAR)
    # convert image to opencv
    opencv_img = np.asarray(pil_img).copy()
    # fill video buffer
    if i_episode % video_every == 0 or cfg.mode == "eval":
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


env.reset()

######## TRAINING ########

# some hyperparameters which are modified here 

BATCH_SIZE = cfg.train.batch_size
EPS_START = cfg.train.eps_start
EPS_END = cfg.train.eps_end

USE_SPACE = cfg.use_space

liveplot = cfg.liveplot
DEBUG = cfg.debug

print('Use Space:', USE_SPACE)
print('Liveplot:', liveplot)
print('Debug Mode:', DEBUG)

SAVE_EVERY = cfg.train.save_every
if not USE_SPACE:
    SAVE_EVERY = SAVE_EVERY * 5

num_episodes = cfg.train.num_episodes

skip_frames = cfg.train.skip_frames

MEMORY_SIZE = cfg.train.memory_size
MEMORY_MIN_SIZE = cfg.train.memory_min_size

# for debugging nn stuff
if DEBUG:
    EPS_START = EPS_END
    BATCH_SIZE = 12
    MEMORY_MIN_SIZE = BATCH_SIZE


exp_name = cfg.exp_name

# init tensorboard
log_path = os.getcwd() + cfg.logdir
log_name = exp_name
log_steps = cfg.train.log_steps
logger = dqn.dqn_logger.DQN_Logger(log_path, exp_name)

# Get number of actions from gym action space
n_actions = env.action_space.n
memory = ReplayMemory(MEMORY_SIZE)

# init agent
agent = dqn_agent.Agent(
    BATCH_SIZE,
    EPS_START,
    EPS_END,
    n_actions,
    MEMORY_MIN_SIZE,
    logger, 
    cfg
)

total_max_q = 0
total_loss = 0

# load if available
if saver.check_loading_model(exp_name):
    # folder and file exists, so load and return
    model_path = saver.model_name(exp_name) 
    print("Loading {}".format(model_path))
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    agent.policy_net.load_state_dict(checkpoint['policy_model_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    memory = checkpoint['memory']
    i_episode = checkpoint['episode']
    global_step = checkpoint['global_step']
    total_max_q = checkpoint['total_max_q']
    total_loss = checkpoint['total_loss']
else:
    print("No prior checkpoints exists, starting fresh")

if cfg.parallel:
    agent.policy_net = nn.DataParallel(agent.policy_net, device_ids=cfg.device_ids)
    agent.target_net = nn.DataParallel(agent.target_net, device_ids=cfg.device_ids)

# helper function to get state, whether to use SPACE or not
def get_state(old_state = None):
    state = None
    if USE_SPACE:
        state = utils.get_z_stuff(get_screen(), model, cfg, i_episode, logger)
    else:
        screen = env.render(mode='rgb_array')
        pil_img = Image.fromarray(screen).resize((128, 128), PIL.Image.BILINEAR)
        # convert image to opencv
        opencv_img = np.asarray(pil_img).copy()
        # fill video buffer
        if i_episode % video_every == 0:
            logger.fill_video_buffer(opencv_img)
        # convert color
        opencv_img = cv2.resize(opencv_img, (64,64), interpolation = cv2.INTER_AREA)
        opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2GRAY)
        state = torch.from_numpy(opencv_img / 255).float()
    ### stack states to have 4 frames
    if cfg.train.reshape_input:
        # stack at last dimension, cause shape should be
        # (batchsize, 36 (zstuff), 16, 16, 4 (stacked dimension))
        if old_state is None:
            state = torch.cat((state, state, state, state), 3)
        else:
            #add newest to front and deselect last state
            old_state_t = torch.from_numpy(old_state).float()
            state = torch.cat((state.detach().cpu(), old_state_t), 3)[:,:,:,:4]
        return state.detach().cpu().numpy()
    else:
        state = state.detach().cpu()
        if old_state is None:
            state = np.stack((state, state, state, state))
        else:
            state = np.stack((state, old_state[0], old_state[1], old_state[2]))
    return state

episode_durations = []


### plot stuff 

gradcam_img = None
# function to plot live while training
def plot_screen(episode, step):
    plt.figure(3)
    plt.clf()
    plt.title('Training - Episode: ' + str(episode) + " - Step: " + str(step))
    plt.imshow(env.render(mode='rgb_array'),
           interpolation='none')
    if gradcam_img is not None:
        plt.figure(4)
        plt.clf()
        plt.title('Training - Episode: ' + str(episode) + " - Step: " + str(step))
        plt.imshow(gradcam_img)
    plt.plot()
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
##################################################
# train mode
if cfg.mode == "train":
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
                action = agent.select_action(state, global_step, logger)
                action_item = action.item()
            # perform action
            _, reward, done, _ = env.step(action_item)

            if reward > 0:
                pos_reward_count += 1
            if reward < 0:
                neg_reward_count += 1
            reward = torch.tensor([reward], device=device)

            # if skip frames are over, observe new state
            next_state = get_state(state)
            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)
            # Move to the next state
            state = next_state
            # TODO: REMOVE
            #print(time.perf_counter() - debugs)

            # Perform one step of the optimization (on the policy network)
            if len(memory) > MEMORY_MIN_SIZE:
                total_max_q, total_loss = agent.optimize_model(memory, total_max_q, total_loss, logger, global_step)
            elif global_step % log_steps == 0:
                logger.log_loss(0, global_step)
                logger.log_max_q(0, global_step)

            # timer stuff
            end = time.perf_counter()
            step_time = end - start

            #TODO: Remove
            #if t % 60 == 0:
            #    logger.save_video(exp_name)

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
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
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
            saver.save_models(exp_name, agent.policy_net, agent.target_net, agent.optimizer, memory, i_episode, global_step, total_max_q, total_loss)
        rtpt.step()
##################################################
# eval mode
elif cfg.mode == "eval":
    print("Evaluating...")
    exp_name = exp_name + "-eval"
    pos_reward_count = 0
    neg_reward_count = 0
    # timer stuff
    start = time.perf_counter()
    episode_start = start
    episode_time = 0
    # Initialize the environment and state
    env.reset()
    # get z stuff for init
    state = get_state()
    # last done action
    action_item = None
    action = None
    # init gradcam
    target_layer = agent.policy_net.conv3
    print(target_layer)
    grad_cam = GradCAM(model=agent.policy_net, target_layer=target_layer, use_cuda=('cuda' in cfg.device))
    # single episode for eval
    for t in count():
        global_step += 1
        # timer stuff
        end = time.perf_counter()
        episode_time = end - episode_start
        start = end
        if liveplot:
            plot_screen("eval", t+1)
        # If skipped frames are over, select action
        if action_item is None or global_step % skip_frames == 0:
            action = agent.select_action(state, global_step, logger)
            action_item = action.item()
            # convert state to tensor and then get grad cam
            state_t = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            #print(state_t.shape)
            gradcam_img = grad_cam(input_tensor=state_t, target_category=action_item)[0, :]
            #print(gradcam_img.shape)
        # perform action
        _, reward, done, _ = env.step(action_item)
        if reward > 0:
            pos_reward_count += 1
        if reward < 0:
            neg_reward_count += 1
        # reward = torch.tensor([reward], device=device)
        state = get_state(state)
        # timer stuff
        end = time.perf_counter()
        step_time = end - start
        #TODO: Remove
        #if t % 60 == 0:
        #    logger.save_video(exp_name)
        if (t) % 10 == 0: # print every 100 steps
            start = time.perf_counter()
            end = time.perf_counter()
            print(
                'exp: {}, step: {}, +reward: {:.2f}, -reward: {:.2f}, s-time: {:.4f}s, e-time: {:.4f}s        '.format(
                    exp_name, t + 1, pos_reward_count, neg_reward_count,
                    step_time, episode_time), end="\r")
        if done:
            break
    # create video
    logger.save_video(exp_name)

print('Complete')
env.render()
env.close()
