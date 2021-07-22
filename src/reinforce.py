import argparse
import gym
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
        self.affine1 = nn.Linear(6, 128)
        self.affine2 = nn.Linear(128, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        return F.softmax(x, dim=1)


def select_action(features, policy):
    probs = policy(torch.tensor(features).unsqueeze(0).float())
    m = Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    return action.item(), log_prob


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

# save model helper function
def save_policy(training_name, policy, episode, optimizer):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'policy': policy.state_dict(),
            'episode': episode,
            'optimizer': optimizer.state_dict()
            }, model_path)


def train():
    print('Experiment name:', cfg.exp_name)
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)
    env = AtariARIWrapper(gym.make(cfg.env_name))
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.train.learning_rate) 
    eps = np.finfo(np.float32).eps.item()
    i_episode = 1
    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        print("{} does exist, loading ... ".format(model_path))
        checkpoint = torch.load(model_path)
        policy.load_state_dict(checkpoint['policy'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        i_episode = checkpoint['episode']
    print('Episodes:', cfg.train.num_episodes)
    print('Gamma:', cfg.train.gamma)
    print('Learning rate:', cfg.train.learning_rate)
    running_reward = None
    reward_buffer = 0
    # training loop
    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                    max_iterations=cfg.train.num_episodes)
    rtpt.start()
    while i_episode <= cfg.train.num_episodes:
        # init env
        _, ep_reward = env.reset(), 0
        _, _, done, _ = env.step(1)
        raw_features, features, _, _ = xutils.do_step(env)
        # env loop
        for t in range(1, 10000):  # Don't infinite loop while learning
            action, log_prob = select_action(features, policy)
            policy.saved_log_probs.append(log_prob)
            raw_features, features, reward, done = xutils.do_step(env, action, raw_features)
            if cfg.liveplot:
                xutils.plot_screen(env, i_episode, t)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        # finish episode and optimize nn
        # replace first running reward with last reward for loaded models
        if running_reward is None:
            running_reward = ep_reward
        else:
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        reward_buffer += ep_reward
        policy, optimizer = finish_episode(policy, optimizer, eps)
        print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}'.format(
            i_episode, ep_reward, running_reward), end="\r")
        if i_episode % cfg.train.log_steps == 0:
            avg_r = reward_buffer / cfg.train.log_steps
            writer.add_scalar('Train/Avg reward', avg_r, i_episode)
            reward_buffer = 0
        if i_episode % cfg.train.save_every == 0:
            save_policy(cfg.exp_name, policy, i_episode + 1, optimizer)
        i_episode += 1
        rtpt.step()


# eval function 
def eval():
    print('Experiment name:', cfg.exp_name)
    print('Evaluating Mode')
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)
    env = AtariARIWrapper(gym.make(cfg.env_name))
    policy = Policy()
    i_episode = 1
    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        print("{} does exist, loading ... ".format(model_path))
        checkpoint = torch.load(model_path)
        policy.load_state_dict(checkpoint['policy'])
        i_episode = checkpoint['episode']
    # init env
    _, ep_reward = env.reset(), 0
    _, _, done, _ = env.step(1)
    raw_features, features, _, _ = xutils.do_step(env)
    # env loop
    for t in range(1, 10000):  # Don't infinite loop while learning
        action, _ = select_action(features, policy)
        raw_features, features, reward, done = xutils.do_step(env, action, raw_features)
        if cfg.liveplot:
            xutils.plot_screen(env, i_episode, t)
        ep_reward += reward
        if done:
            break
    print('Episode {}\tReward: {:.2f}}'.format(
        i_episode, ep_reward), end="\r")


if __name__ == '__main__':
    if cfg.mode == "train":
        train()
    elif cfg.mode == "eval":
        eval()