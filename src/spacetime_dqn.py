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
import dqn.dqn_logger as Logger

import rl_utils

from atariari.benchmark.wrapper import AtariARIWrapper
from rtpt import RTPT
from engine.utils import get_config
from PIL import Image


# load config
cfg, task = get_config()
torch.manual_seed(cfg.seed)
print('Seed:', torch.initial_seed())

USE_ATARIARI = (cfg.device == "cpu")
print("Using AtariAri:", USE_ATARIARI)

# lambda for loading and saving qtable
PATH_TO_OUTPUTS = os.getcwd() + "/rl_checkpoints/"
model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_dqn_model.pth"

# init env stuff
cfg.device_ids = [0]
env_name = cfg.gamelist[0]
print("Env Name:", env_name)
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


# model
class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(DQN, self).__init__()
        self.affine1 = nn.Linear(n_inputs, 128)
        self.affine2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        return self.affine2(x)


BATCH_SIZE = 128
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1
MEM_MIN_SIZE = 5000
MEM_MAX_SIZE = 10000

i_episode = 0
features = rl_utils.convert_to_state(cfg, info)
policy_net = DQN(len(features) * 2, n_actions)
target_net = DQN(len(features) * 2, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEM_MAX_SIZE)

logger = Logger.DQN_Logger("rl_logs/", cfg.exp_name + "_dqn")

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
    policy_net, optimizer, target_net, steps_done, i_episode = load_model(model_path, policy_net, target_net, optimizer)
print("Current steps done:", steps_done)

eps_threshold = EPS_START

def select_action(state):
    global steps_done
    global eps_threshold
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            selected_action = torch.argmax(policy_net(state)).unsqueeze(0)
            return selected_action
    else:
        selected_action = torch.tensor([random.randrange(n_actions)], dtype=torch.long)
        return selected_action


def optimize_model():
    if len(memory) < MEM_MIN_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_batch = torch.stack(batch.state)
    next_state_batch = torch.stack(batch.next_state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)
    done_batch = torch.tensor(batch.done).float()

    # Make predictions
    state_q_values = policy_net(state_batch)
    next_states_q_values = policy_net(next_state_batch)
    next_states_target_q_values = target_net(next_state_batch)
    # Find selected action's q_value
    selected_q_value = state_q_values.gather(1, action_batch)
    # Get indice of the max value of next_states_q_values
    # Use that indice to get a q_value from next_states_target_q_values
    # We use greedy for policy So it called off-policy
    next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1))
    # Use Bellman function to find expected q value
    expected_q_value = reward_batch + GAMMA * next_states_target_q_value * (1 - done_batch).unsqueeze(1)

    # Calc loss with expected_q_value and q_value
    loss = F.mse_loss(selected_q_value, expected_q_value.detach())
    logger.log_loss(loss, steps_done)
    optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    optimizer.step()


max_episode = 50000
# episode loop
running_reward = -999
# training loop
rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                max_iterations=max_episode)
rtpt.start()
if i_episode < max_episode:
    while i_episode < max_episode:
        # Initialize the environment and state
        env.reset()
        ep_reward = 0
        s_state = rl_utils.convert_to_state(cfg, info)
        s_state = torch.tensor(s_state, dtype=torch.float) 
        # state stacking to have current and previous state at once
        state = torch.cat((s_state, s_state), 0)
        for t in count():
            # Select and perform an action
            action = select_action(state)
            observation, reward, done, info = env.step(action.item())
            ep_reward += reward
            reward = torch.tensor([reward])
            # Observe new state
            s_next_state = rl_utils.convert_to_state(cfg, info)
            # convert next state to torch
            s_next_state = torch.tensor(s_next_state, dtype=torch.float)
            # concat to stacking tensor
            next_state = torch.cat((s_state, s_next_state), 0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)
            # Move to the next state
            state = next_state
            s_state = s_next_state
            # Perform one step of the optimization (on the policy network)
            optimize_model()
            print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\t Eps: {:.2f}\tSteps: {}       '.format(
                    i_episode, ep_reward, running_reward, eps_threshold, t), end="\r")
            if done:
                logger.log_episode(t, ep_reward, 0, i_episode, steps_done)
                logger.log_eps(eps_threshold, steps_done)
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
else:
    print("Eval mode")
    print("NOT IMPLEMENTED!")

