import numpy as np
import gym
import os
import math
import random
import pickle
import bz2
import rl_utils

from atariari.benchmark.wrapper import AtariARIWrapper
from rtpt import RTPT
from engine.utils import get_config
from tqdm import tqdm

# load config
cfg, task = get_config()
use_cuda = 'cuda' in cfg.device

USE_ATARIARI = (cfg.device == "cpu")
print("Using AtariAri:", USE_ATARIARI)
relevant_atariari_labels = {"pong": ["player", "enemy", "ball"], "boxing": ["enemy", "player"]}

# lambda for loading and saving qtable
PATH_TO_OUTPUTS = os.getcwd() + "/rl_checkpoints/"
model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + ".pbz2"

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

DISCRETIZE_FACTOR = 10
print("Discretization Factor:", DISCRETIZE_FACTOR)

# helper function to discretize
def discretize(state):
    div_state = [i/DISCRETIZE_FACTOR for i in state]
    intstate = [int(i) for i in div_state]
    return intstate


#Initializing the Q-table as dictionary and parameters
Q = {}
gamma =  0.97
eps_start = 1.0
eps_end = 0.01
eps_decay = 10000
learning_rate = 0.005


# function to save qtable
def save_qtable(training_name, q_table, i_episode):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    checkpoint_path = model_name(training_name)
    print("Saving {}".format(checkpoint_path))
    saving_dict = {}
    saving_dict["q_table"] = q_table
    saving_dict["i_episode"] = i_episode
    # create bz2 file
    qfile = bz2.BZ2File(checkpoint_path,'w')
    pickle.dump(saving_dict, qfile)
    qfile.close()


# check checks if model with given name exists,
# and loads it
def load_qtable(training_name):
    if not os.path.exists(PATH_TO_OUTPUTS):
        print("{} does not exist".format(PATH_TO_OUTPUTS))
        return None, None
    checkpoint_path = model_name(training_name)
    if not os.path.isfile(checkpoint_path):
        print("{} does not exist".format(checkpoint_path))
        return None, None
    # load bz2 file
    qfile = bz2.BZ2File(checkpoint_path,'r')
    loading_dict = pickle.load(qfile)
    qfile.close()
    print("Successfully loaded")
    print("Loaded episode:", loading_dict["i_episode"])
    print("Q-Table Length:", len(loading_dict["q_table"]))
    return loading_dict["q_table"], loading_dict["i_episode"]


# function to select action by given state
def select_action(state, episode):
    eps_threshold = eps_start
    eps_threshold = eps_end + (eps_start - eps_end) * \
        math.exp(-1. * episode / eps_decay)
    if state is not None and state in Q.keys():
        sample = random.random()
        if sample > eps_threshold:
            return np.argmax(Q[state]), eps_threshold
    return random.randrange(n_actions), eps_threshold


i_episode = 0
exp_name = cfg.exp_name + "-qlearning"
# check if q table loading is not null
tmp_Q, tmp_i_episode = load_qtable(exp_name)
if tmp_Q is not None:
    Q = tmp_Q
    i_episode = tmp_i_episode

max_episode = 100000
# episode loop
running_reward = -999
if i_episode < max_episode:
    # training loop
    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name + "-QL",
                    max_iterations=max_episode)
    rtpt.start()
    while i_episode < max_episode:
        state, ep_reward = env.reset(), 0
        action = np.random.randint(n_actions)
        state = str([0,0,0,0,0,0])
        if not state in Q.keys():
            # entry doesnt exists, fill entry with 0
            Q[state] = np.zeros(n_actions)
        # env step loop
        for t in range(1, 10000):  # Don't infinite loop while learning
            # select action
            action, eps_t = select_action(state, i_episode)
            # do action and observe
            observation, reward, done, info = env.step(action)
            ep_reward += reward
            next_state = None
            # when atariari
            if USE_ATARIARI:
                next_state = rl_utils.convert_to_state(cfg, info)
            # when spacetime
            else:
                # use spacetime to get scene_list
                _, next_state = rl_utils.get_scene(cfg, observation, space, z_classifier, sc, transformation, use_cuda)
                # flatten scene list and discretize
            # update qtable
            next_state = str(discretize(next_state))
            if not next_state in Q.keys():
                # entry doesnt exists, fill entry with 0
                Q[next_state] = np.zeros(n_actions)
            Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * \
                    np.max(Q[next_state]) - Q[state][action])
            # finish step
            print('Episode: {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tEps Treshold: {:.2f}\tTable-Len: {}\tSteps: {}       '.format(
                i_episode, ep_reward, running_reward, eps_t, len(Q), t), end="\r")
            state = next_state
            if done:
                break
        # replace first running reward with last reward for loaded models
        if running_reward == -999:
            running_reward = ep_reward
        else:
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # finish episode
        i_episode += 1
        # checkpoint saver
        if i_episode % 100 == 0:
            save_qtable(exp_name, Q, i_episode)
        rtpt.step()
else:
    runs = 15
    print("Eval mode")
    for run in tqdm(range(runs)):
        state, ep_reward = env.reset(), 0
        action = np.random.randint(n_actions)
        state = str([0,0,0,0,0,0])
        # env step loop
        for t in range(1, 10000):  # Don't infinite loop while learning
            # select action
            action, eps_t = select_action(state, i_episode)
            # do action and observe
            observation, reward, done, info = env.step(action)
            ep_reward += reward
            # when atariari
            if False:
                state = rl_utils.convert_to_state(cfg, info)
            # when spacetime
            else:
                # use spacetime to get scene_list
                _, state = rl_utils.get_scene(cfg, observation, space, z_classifier, sc, transformation, use_cuda)
            # finish step
            state = str(discretize(state))
            print('Episode: {}\tLast reward: {:.2f}\tEps Treshold: {:.2f}\tTable-Len: {}\tSteps: {}       '.format(
                i_episode, ep_reward, eps_t, len(Q), t), end="\r")
            if done:
                break
        print("Final Reward:", ep_reward)