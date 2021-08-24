# own mini version of dreamer for processed features
import gym
import os
import torch
import threading

from atariari.benchmark.wrapper import AtariARIWrapper
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim import Adam
from itertools import count
from rtpt import RTPT
from time import sleep, time

import xrl.utils as xutils

import reinforce

from xrl.minidreamer.dataset import Dataset
from xrl.minidreamer.model import WorldPredictor

# world pred parameter
batch = 50
L = 50 #seq len world training
history_size = 400
lr_pred = 1e-3

# policy parameter
lr_policy = 1e-2

# misc parameter
adam_eps = 1e-5
decay = 1e-6


# helper function to save all models
def save(world_model, policy, global_step, episode, optimizer):
    return None


# helper function to load all models
def load(cfg):
    return None


def train(cfg):
    ### HYPERPARAMETERS ###
    print('Experiment name:', cfg.exp_name)
    print('Env name:', cfg.env_name)
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)


    # params for saving loading
    save_path = os.getcwd() + "/xrl/checkpoints/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = save_path + cfg.exp_name + "_model.pth"

    # device to use for training
    device = cfg.device

    # logger
    writer = SummaryWriter(os.getcwd() + cfg.logdir + cfg.exp_name)

    # params for game
    env = AtariARIWrapper(gym.make(cfg.env_name))
    n_actions = env.action_space.n
    _, features, _, _ = xutils.do_step(env)
    # init policy net
    
    ### MODELS ###
    predictor = WorldPredictor(len(features)).to(device)
    policy = reinforce.get_network(cfg, len(features), cfg.train.hidden_layer_size, n_actions).to(device)
    optim_predictor = Adam(predictor.parameters(), lr=lr_pred, eps=adam_eps, weight_decay=decay)
    optim_policy = Adam(policy.parameters(), lr=lr_policy, eps=adam_eps, weight_decay=decay)

    history = []
    steps_done = [0]  
    i_episode = 1

    #TODO: load save
       
    # inner function to give it access to shared variables
    # loop to interactively fill history with newest episodes
    def gather_episode():
        with torch.no_grad():
            rtpt_s = RTPT(name_initials='DV', experiment_name=cfg.exp_name + "_interactive",
                    max_iterations=cfg.train.num_episodes)
            rtpt_s.start()
            while steps_done[0] < cfg.train.max_steps:
                obs = env.reset()
                raw_features, features, _, _ = xutils.do_step(env)
                episode = []
                done = False
                r_sum = 0
                t = 0
                while t < 50000:
                    a = policy(features)
                    # create episode entry with last state, action, state and reward
                    episode.append(features)
                    episode.append(a)
                    raw_features, features, reward, done = xutils.do_step(env, a, raw_features)
                    # append reward to features to have both
                    episode.append(features)
                    episode.append(reward)
                    r_sum += reward
                    #print("Step: {}, Reward: {}".format(t, rew), end="\r")
                    steps_done[0] += 1
                    if done:
                        break
                    t += 1
                writer.add_scalar('Train/Reward', r_sum, steps_done[0])
                history.append(episode)
                for _ in range(len(history) - history_size):
                    history.pop(0)
                rtpt_s.step()

    #start gathering episode thread
    t = threading.Thread(target=gather_episode)
    t.start()

    print("Dataset init")
    while len(history) < 50:
        # check every second if first history entry is inside
        print("Wainting until history large enough (current size: {}, needed: {})".format(len(history), 50), end="\r")
        sleep(5.0)
    print("done")

    ### DATASET ###
    ds = Dataset(history, seq_len=L, history_size=history_size)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True
    )

    rtpt = RTPT(name_initials='DV', experiment_name=cfg.exp_name,
                    max_iterations=cfg.train.num_episodes)
    rtpt.start()
    while steps_done[0] < cfg.train.max_steps:
        l_pred = 0
        l_policy = 0
        len_h = 0
        pbar = tqdm(loader)
        for ls, a, s, r in pbar:
            ls = ls.to(device)
            a = a.to(device)
            s = s.to(device)
            r = r.to(device)

            # TRAIN PREDICTOR
            
            #TODO: implement training loop for predictor

            # TRAIN POLICY

            #TODO: implement training loop for policy with given REINFORCE stuff

            # set logging variables
            l_pred = 0
            l_policy = 0
            len_h = len(history)
            # display
            pbar.set_postfix(
                l_pred = l_pred,
                l_policy = l_policy,
                len_h = len_h,
                i_episode = i_episode
            )

        # log all
        writer.add_scalar('Train/Loss World Model', l_pred, i_episode)
        writer.add_scalar('Train/Loss Actor', l_policy, i_episode)

        # end episode
        i_episode += 1
        rtpt.step()