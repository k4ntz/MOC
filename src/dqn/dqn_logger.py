# helper to log with tensorboard

import torch
import os
from torch.utils.tensorboard import SummaryWriter


class DQN_Logger:
    def __init__(self, logpath, logname):
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        self.writer = SummaryWriter(logpath + logname)

    def log_episode(self, episode_steps, pos_reward, neg_reward, episode, global_step):
        if self.writer == None:
            print("Logger is not initialized...")
            return
        self.writer.add_scalar('Train/episode_steps', episode_steps, episode)
        self.writer.add_scalar('Train/reward', pos_reward - neg_reward, global_step)
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

    def log_max_q(self, max_q, global_step):
        if self.writer == None:
            print("Logger is not initialized...")
            return
        self.writer.add_scalar('Train/max q value', max_q, global_step)
