import gym
import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import rl_utils

from atariari.benchmark.wrapper import AtariARIWrapper
from rtpt import RTPT
from engine.utils import get_config
from model import get_model
from src.spacetime_reinforce import LEARNING_RATE
from utils import Checkpointer
from solver import get_optimizers
from PIL import Image
from torchvision import transforms


# load config
cfg, task = get_config()
torch.manual_seed(cfg.seed)
print('Seed:', torch.initial_seed())

USE_ATARIARI = (cfg.device == "cpu")
print("Using AtariAri:", USE_ATARIARI)

# lambda for loading and saving qtable
PATH_TO_OUTPUTS = os.getcwd() + "/rl_checkpoints/"
model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_policy_model.pth"

# init env stuff
cfg.device_ids = [0]
env_name = cfg.gamelist[0]
env = gym.make(env_name)
if USE_ATARIARI:
    env = AtariARIWrapper(env)
obs = env.reset()
obs, reward, done, info = env.step(1)
n_actions = env.action_space.n
#Getting the state space
print("Action Space {}".format(env.action_space))
print("State {}".format(info))


# replay memory of dqn
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
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


# model
class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        self.affine1 = nn.Linear(n_inputs, 128)
        self.affine2 = nn.Linear(128, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        return self.affine2(x)


BATCH_SIZE = 128
LEARNING_RATE = 0.0001
GAMMA = 0.97
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

i_episode = 0
features = rl_utils.convert_to_state(info)
policy_net = DQN(len(features), n_actions)
target_net = DQN(len(features), n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(10000)

steps_done = 0
i_episode = 0


# save model helper function
def save_model(training_name, policy, target, steps, episode, optimizer):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'policy': policy.state_dict(),
            'target': target.state_dict(),
            'steps': steps,
            'episode': episode,
            'optimizer': optimizer.state_dict()
            }, model_path)


# load model
def load_model(model_path, policy, target, optimizer=None):
    print("{} does exist, loading ... ".format(model_path))
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint['policy'])
    target.load_state_dict(checkpoint['target'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    i_episode = checkpoint['episode']
    steps = checkpoint['steps']
    return policy, optimizer, target, steps, i_episode


# load if exists
model_path = model_name(cfg.exp_name)
if os.path.isfile(model_path):
    policy_net, target_net, optimizer, steps_done, i_episode = load_model(model_path, policy_net, target_net, optimizer)


def select_action(state):
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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)



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
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

max_episode = 50000
# episode loop
running_reward = -999
# training loop
rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                max_iterations=max_episode)
rtpt.start()
while i_episode < max_episode:
    # Initialize the environment and state
    env.reset()
    ep_reward = 0
    state = rl_utils.convert_to_state(info)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        info, reward, done, _, _ = env.step(action.item())
        ep_reward += reward
        reward = torch.tensor([reward])
        # Observe new state
        next_state = rl_utils.convert_to_state(info)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the policy network)
        optimize_model()
        print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tSteps: {}       '.format(
                i_episode, ep_reward, running_reward, t), end="\r")
        if done:
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    # replace first running reward with last reward for loaded models
    if running_reward == -999:
        running_reward = ep_reward
    else:
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    # save if necessary
    if (i_episode + 1) % 100 == 0:
        save_model(cfg.exp_name, policy_net, target_net, steps_done, i_episode + 1, optimizer)
    # finish episode
    i_episode += 1
    rtpt.step()

