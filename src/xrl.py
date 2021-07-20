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
from numpy.lib.function_base import select
import pandas as pd
import seaborn as sn
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from itertools import count
from PIL import Image
from atariari.benchmark.wrapper import AtariARIWrapper

from rtpt import RTPT

import dqn.dqn_logger
import dqn.utils as utils

cfg, _ = utils.get_config()

PATH_TO_OUTPUTS = os.getcwd() + "/xrl/checkpoints/"
if not os.path.exists(PATH_TO_OUTPUTS):
    os.makedirs(PATH_TO_OUTPUTS)

model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_model.pth"


# define policy network
class policy_net(nn.Module):
    def __init__(self, input, hidden, actions): 
        super(policy_net, self).__init__()
        self.h = nn.Linear(input, hidden)
        self.out = nn.Linear(hidden, actions)

    # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
    def forward(self, x):
        x = F.relu(self.h(x))
        x = F.softmax(self.out(x), dim=1)
        return x


def init_weights(m):
    # nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
    # nn.Conv2d bias is of shape [16] i.e. # number of filters
    
    # nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
    # nn.Linear bias is of shape [32] i.e. # number of output features
    
    if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)


# function to create random agents of given count
def return_random_agents(num_agents):
    agents = []
    for _ in range(num_agents):
        agent = policy_net(6, 32, 3)
        for param in agent.parameters():
            param.requires_grad = False
        init_weights(agent)
        agents.append(agent) 
    return agents


# function to plot live while training
def plot_screen(env, episode, step, second_img=None):
    plt.figure(3)
    plt.clf()
    plt.title('Episode: ' + str(episode) + " - Step: " + str(step))
    plt.imshow(env.render(mode='rgb_array'),
           interpolation='none')
    if second_img is not None:
        plt.figure(2)
        plt.clf()
        plt.title('X - Episode: ' + str(episode) + " - Step: " + str(step))
        plt.imshow(second_img)
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
    features.append(0) 
    # not needed, bc target pos y is already calculated
    # features.append((player_y - ball_y)/ norm_factor)# distance y ball and player
    features.append((ball_x - enemy_x)/ norm_factor) # distance x ball and enemy
    features.append((ball_y - enemy_y)/ norm_factor) # distance y ball and enemy
    # euclidean distance between old and new ball coordinates to represent current speed per frame
    features.append(math.sqrt((ball_x - raw_features[8])**2 + (ball_y - raw_features[9])**2) / 25) 
    return raw_features, features


# helper function to get features
def do_step(env, action=1, last_raw_features=None):
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
def select_action(features, policy):
    # calculate probabilities of taking each action
    probs = policy(torch.tensor(features).unsqueeze(0).float())
    # sample an action from that set of probs
    sampler = Categorical(probs)
    action = sampler.sample()
    # return action
    return action


# function to run list of agents in given env
def run_agents(env, agents):
    start = time.perf_counter()
    reward_agents = []
    _ = env.reset()
    _, _, done, info = env.step(1)
    for agent in agents:
        agent.eval()
        raw_features, features, _, _ = do_step(env)
        r = 0
        for t in count():
            action = select_action(features, agent)
            raw_features, features, reward, done = do_step(env, action, raw_features)
            r = r + reward
            if(done):
                break
        reward_agents.append(r)
    return reward_agents


# returns average score of given agent when it runs n times
def return_average_score(env, agent, runs):
    score = 0.
    for i in range(runs):
        score += run_agents(env, [agent])[0]
    avg_score = score/runs
    return avg_score


# gets avg score of every agent running n runs 
def run_agents_n_times(agents, runs):
    avg_score = []
    env = AtariARIWrapper(gym.make(cfg.env_name))
    for agent in tqdm(agents):
        avg_score.append(return_average_score(env, agent, runs))
    return avg_score


# function to mutate given agent to child agent
def mutate(agent):
    child_agent = copy.deepcopy(agent)
    mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
    for param in child_agent.parameters():
        if(len(param.shape)==4): #weights of Conv2D
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    for i2 in range(param.shape[2]):
                        for i3 in range(param.shape[3]):
                            param[i0][i1][i2][i3]+= mutation_power * np.random.randn()
        elif(len(param.shape)==2): #weights of linear layer
            for i0 in range(param.shape[0]):
                for i1 in range(param.shape[1]):
                    param[i0][i1]+= mutation_power * np.random.randn()
        elif(len(param.shape)==1): #biases of linear layer or conv layer
            for i0 in range(param.shape[0]):
                param[i0]+=mutation_power * np.random.randn()
    return child_agent


# function to add elite to childs 
def add_elite(agents, sorted_parent_indexes, elite_index=None, only_consider_top_n=10):
    candidate_elite_index = sorted_parent_indexes[:only_consider_top_n]
    if(elite_index is not None):
        candidate_elite_index = np.append(candidate_elite_index,[elite_index]) 
    top_score = None
    top_elite_index = None
    env = AtariARIWrapper(gym.make(cfg.env_name))
    for i in candidate_elite_index:
        score = return_average_score(env, agents[i], runs=5)
        print("Score for elite i ", i, " is ", score)
        if(top_score is None):
            top_score = score
            top_elite_index = i
        elif(score > top_score):
            top_score = score
            top_elite_index = i
    print("Elite selected with index ",top_elite_index, " and score", top_score)
    child_agent = copy.deepcopy(agents[top_elite_index])
    return child_agent


# function to create and return children from given parent agents
def return_children(agents, sorted_parent_indexes, elite_index):  
    children_agents = []
    #first take selected parents from sorted_parent_indexes and generate N-1 children
    for i in range(len(agents) - 1):
        selected_agent_index = sorted_parent_indexes[np.random.randint(len(sorted_parent_indexes))]
        children_agents.append(mutate(agents[selected_agent_index]))
    #now add one elite
    elite_child = add_elite(agents, sorted_parent_indexes, elite_index)
    children_agents.append(elite_child)
    elite_index = len(children_agents) - 1 #it is the last one
    return children_agents, elite_index


 # Compute softmax values for each sets of scores in x
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# save model helper function
def save_model(training_name, policy, episode, global_step):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'policy_state_dict': policy.state_dict(),
            'episode': episode,
            'global_step': global_step
            }, model_path)


# train main function
def train():
    print('Experiment name:', cfg.exp_name)

    LIVEPLOT = cfg.liveplot
    DEBUG = cfg.debug
    print('Liveplot:', LIVEPLOT)
    print('Debug Mode:', DEBUG)

    generations = cfg.train.num_episodes
    print('Generations:', generations)

    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize N number of agents
    num_agents = 500
    print('Number of agents:', num_agents)
    agents = return_random_agents(num_agents)

    # How many top agents to consider as parents
    top_limit = 20
    print('Number of top agents:', top_limit)

    # runs per generation
    n_gen_runs = 3
    print('Number of runs per generation:', n_gen_runs)

    elite_index = None

    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                    max_iterations=generations)
    rtpt.start()
    for generation in range(generations):
        print("Starting generation", generation)
        # return rewards of agents
        rewards = run_agents_n_times(agents, n_gen_runs) #return average of 3 runs
 
        # sort by rewards
        # reverses and gives top values (argsort sorts by ascending by default) https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
        sorted_parent_indexes = np.argsort(rewards)[::-1][:top_limit] 
        print("")
        print("")
        
        top_rewards = []
        for best_parent in sorted_parent_indexes:
            top_rewards.append(rewards[best_parent])
       
        print("Generation ", generation, " | Mean rewards: ", np.mean(rewards), " | Mean of top 5: ",np.mean(top_rewards[:5]))
        #print(rewards)
        print("Top ",top_limit," scores", sorted_parent_indexes)
        print("Rewards for top: ",top_rewards)
        
        # setup an empty list for containing children agents
        children_agents, elite_index = return_children(agents, sorted_parent_indexes, elite_index)
 
        # kill all agents, and replace them with their children
        agents = children_agents
        rtpt.step()


if __name__ == '__main__':
    if cfg.mode == "train":
        train()
