# helper to log with tensorboard

import torch
import os
from torch.utils.tensorboard import SummaryWriter


class DQN_Logger:
    def __init__(self, logpath, logname):
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        self.writer = SummaryWriter(logpath + logname)

    def log_episode(self, global_wins, global_loses, episode_time, pos_reward, neg_reward, episode, global_step):
        if self.writer == None:
            print("Logger is not initialized...")
            return
        self.writer.add_scalar('Train/global_wins', global_wins, global_step)
        self.writer.add_scalar('Train/global_loses', global_loses, global_step)
        self.writer.add_scalar('Train/episode_time', episode_time, global_step)
        self.writer.add_scalar('Train/pos_reward', pos_reward, global_step)
        self.writer.add_scalar('Train/neg_reward', neg_reward, global_step)
        self.writer.add_scalar('Train/reward', pos_reward - neg_reward, global_step)
        self.writer.add_scalar('Train/pos_reward_episode', pos_reward, episode)
        self.writer.add_scalar('Train/neg_reward_episode', neg_reward, episode)
        self.writer.add_scalar('Train/reward_episode', pos_reward - neg_reward, episode)

    def log_eps(self, eps, global_step):
        if self.writer == None:
            print("Logger is not initialized...")
            return
        self.writer.add_scalar('Train/epsilon', eps, global_step)

    
    def log_loss(self, loss, global_step):
        if self.writer == None:
            print("Logger is not initialized...")
            return
        self.writer.add_scalar('Train/loss', loss, global_step)
