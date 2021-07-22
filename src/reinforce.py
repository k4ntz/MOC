import argparse
import gym
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from itertools import count
from torch.distributions import Categorical
from atariari.benchmark.wrapper import AtariARIWrapper

from rtpt import RTPT

import xrl.utils as xutils
import dqn.utils as utils

cfg, _ = utils.get_config()

PATH_TO_OUTPUTS = os.getcwd() + "/xrl/checkpoints/"
if not os.path.exists(PATH_TO_OUTPUTS):
    os.makedirs(PATH_TO_OUTPUTS)

model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_model.pth"

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m


def finish_episode(policy, optimizer, eps):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + cfg.train.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    return policy, optimizer


def train():
    print('Experiment name:', cfg.exp_name)
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)
    env = AtariARIWrapper(gym.make(cfg.env_name))
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.train.learning_rate)    #lr 1e-2
    eps = np.finfo(np.float32).eps.item()
    running_reward = -21
    # training loop
    for i_episode in cfg.train.num_episodes:
        # init env
        _, ep_reward = env.reset(), 0
        _, _, done, info = env.step(1)
        raw_features, features, _, _ = xutils.do_step(env)
        # env loop
        for t in range(1, 10000):  # Don't infinite loop while learning
            action, m = select_action(features, policy)
            policy.saved_log_probs.append(m.log_prob(action))
            raw_features, features, reward, done = xutils.do_step(env, action, raw_features)
            if cfg.liveplot:
                xutils.plot_screen(env, i_episode, t)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        # finish episode and optimize nn
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        policy, optimizer = finish_episode(policy, optimizer, eps)
        if i_episode % cfg.train.log_steps == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))


if __name__ == '__main__':
    if cfg.mode == "train":
        train()
    elif cfg.mode == "eval":
        None