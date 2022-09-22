import numpy as np
import gym
import os

import torch
import rl_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from atariari.benchmark.wrapper import AtariARIWrapper

from rtpt import RTPT
from engine.utils import get_config

# load config
cfg, task = get_config()
torch.manual_seed(cfg.seed)
print('Seed:', torch.initial_seed())

USE_ATARIARI = (cfg.device == "cpu")
print("Using AtariAri:", USE_ATARIARI)
relevant_atariari_labels = {"pong": ["player", "enemy", "ball"], "boxing": ["enemy", "player"]}

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

print("Loading space...")
space, transformation, sc, z_classifier = rl_utils.load_space(cfg)

# model
class Policy(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_inputs, 128)
        self.affine2 = nn.Linear(128, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

# init policy and optimizer
LEARNING_RATE = 0.0001
GAMMA = 0.97
EPS = np.finfo(np.float32).eps.item()
i_episode = 0
features = rl_utils.convert_to_state(cfg, info)
policy = Policy(len(features), n_actions)
optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)

# select action function
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


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


# load model
def load_model(model_path, policy, optimizer=None):
    print("{} does exist, loading ... ".format(model_path))
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint['policy'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    i_episode = checkpoint['episode']
    return policy, optimizer, i_episode


# load if exists
model_path = model_name(cfg.exp_name)
if os.path.isfile(model_path):
    policy, optimizer, i_episode = load_model(model_path, policy, optimizer)


# env = Atari(env_name)
max_episode = 50000
# episode loop
running_reward = -999
# training loop
rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                max_iterations=max_episode)
rtpt.start()
if i_episode < max_episode:
    while i_episode < max_episode:
        state, ep_reward = env.reset(), 0
        action = np.random.randint(n_actions)
        # env step loop
        for t in range(1, 10000):  # Don't infinite loop while learning
            observation, reward, done, info = env.step(action)
            ep_reward += reward
            policy.rewards.append(reward)
            features = rl_utils.convert_to_state(cfg, info)
            action = select_action(np.asarray(features))
            print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tSteps: {}       '.format(
                i_episode, ep_reward, running_reward, t), end="\r")
            if done:
                break
        # replace first running reward with last reward for loaded models
        if running_reward == -999:
            running_reward = ep_reward
        else:
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # finish up episode and optimize
        R = 0
        policy_loss = []
        returns = []
        for r in policy.rewards[::-1]:
            R = r + GAMMA * R            # gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + EPS)
        for log_prob, R in zip(policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        del policy.rewards[:]
        del policy.saved_log_probs[:]
        # save if necessary
        if (i_episode + 1) % 100 == 0:
            save_policy(cfg.exp_name, policy, i_episode + 1, optimizer)
        # finish episode
        i_episode += 1
        rtpt.step()
else:
    print("Eval Mode")
    print("NOT IMPLEMENTED")