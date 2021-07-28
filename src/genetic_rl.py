# q learning with given shallow representation of atari game

import gym
import numpy as np
import os
import pandas as pd
import seaborn as sn
import copy
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from joblib import Parallel, delayed

from itertools import count
from PIL import Image
from atariari.benchmark.wrapper import AtariARIWrapper
from captum.attr import IntegratedGradients

from rtpt import RTPT

import xrl.utils as xutils
import dqn.utils as utils
import dqn.dqn_logger as vlogger

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
    reward_agents = []
    _ = env.reset()
    _, _, done, info = env.step(1)
    for agent in agents:
        agent.eval()
        raw_features, features, _, _ = xutils.do_step(env)
        r = 0
        for t in count():
            action = select_action(features, agent)
            raw_features, features, reward, done = xutils.do_step(env, action, raw_features)
            r = r + reward
            if(done):
                break
        reward_agents.append(r)
    return reward_agents


# returns average score of given agent when it runs n times
def return_average_score(agent, runs):
    score = 0.
    env = AtariARIWrapper(gym.make(cfg.env_name))
    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                    max_iterations=runs)
    rtpt.start()
    for i in range(runs):
        score += run_agents(env, [agent])[0]
        rtpt.step()
    avg_score = score/runs
    return avg_score


# gets avg score of every agent running n runs 
def run_agents_n_times(agents, runs):
    avg_score = []
    agents = tqdm(agents)
    cpu_cores = min(multiprocessing.cpu_count(), 100)
    avg_score = Parallel(n_jobs=cpu_cores)(delayed(return_average_score)(agent, runs) for agent in agents)
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
    candidate_elite_index = tqdm(candidate_elite_index)
    cpu_cores = min(100, max(multiprocessing.cpu_count(), only_consider_top_n))
    scores = Parallel(n_jobs=cpu_cores)(delayed(return_average_score)(agents[i], runs=5) for i in candidate_elite_index)
    for i, score in enumerate(scores):
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
def save_agents(training_name, agents, generation):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'agents': agents,
            'generation': generation
            }, model_path)


# train main function
def train():
    print('Experiment name:', cfg.exp_name)

    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    generations = cfg.train.num_episodes
    print('Generations:', generations)

    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize N number of agents
    num_agents = 500
    print('Number of agents:', num_agents)
    agents = return_random_agents(num_agents)
    generation = 0

    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        print("{} does exist, loading ... ".format(model_path))
        checkpoint = torch.load(model_path)
        agents = checkpoint['agents']
        generation = checkpoint['generation']

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
    while generation < generations:
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

        #log stuff
        writer.add_scalar('Train/Mean rewards', np.mean(rewards), generation)
        writer.add_scalar('Train/Mean of top 5', np.mean(top_rewards[:5]), generation)
        # save generation
        generation += 1
        save_agents(cfg.exp_name, agents, generation)
        # make rtpt step
        rtpt.step()


# function to eval best agent of last generation
def play_agent():
    print('Experiment name:', cfg.exp_name)
    print('Evaluating Mode')
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)
    # initialize N number of agents
    num_agents = 500
    print('Number of agents:', num_agents)
    agents = return_random_agents(num_agents)
    generation = 0

    # load if exists
    model_path = model_name(cfg.exp_name)
    if os.path.isfile(model_path):
        print("{} does exist, loading ... ".format(model_path))
        checkpoint = torch.load(model_path)
        agents = checkpoint['agents']
        generation = checkpoint['generation']

    elite_index = 269 # SET FOR ELITE INDEX FROM LOADED GENERATION
    elite_agent = agents[elite_index]

    # play with elite agent
    env = AtariARIWrapper(gym.make(cfg.env_name))
    logger = vlogger.DQN_Logger(os.getcwd() + cfg.logdir, cfg.exp_name, vfolder="/xrl/video/", size=(480,480))
    ig = IntegratedGradients(elite_agent)
    _ = env.reset()
    _, _, done, info = env.step(1)
    elite_agent.eval()
    raw_features, features, _, _ = xutils.do_step(env)
    r = 0
    for t in count():
        action = select_action(features, elite_agent)
        raw_features, features, reward, done = xutils.do_step(env, action, raw_features)
        if cfg.liveplot or cfg.make_video:
            img = xutils.plot_integrated_gradient_img(ig, cfg.exp_name, features, action, env, cfg.liveplot)
            logger.fill_video_buffer(img)
            print('Generation {}\tReward: {:.2f}\t Step: {:.2f}'.format(
                generation, r, t), end="\r")
        r = r + reward
        if(done):
            break
    print("Elite agent with index {} - final reward: {}".format(elite_index, r))
    if cfg.liveplot or cfg.make_video:
        logger.save_video(cfg.exp_name)



if __name__ == '__main__':
    if cfg.mode == "train":
        train()
    elif cfg.mode == "eval":
        play_agent()
