# helper to log with tensorboard

import torch
import os
import cv2
from torch.utils.tensorboard import SummaryWriter

FOLDER_TO_VIDEO = "/dqn/video/"
PATH_TO_VIDEO = os.getcwd() + FOLDER_TO_VIDEO

class DQN_Logger:
    def __init__(self, logpath, logname):
        if not os.path.exists(logpath):
            os.makedirs(logpath)
        self.writer = SummaryWriter(logpath + logname)
        self.video_buffer = []

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

    def fill_video_buffer(self, image, fps=30):
        self.video_buffer.append(image)

    def save_video(self, episode, fps=25.0):
        if not os.path.exists(PATH_TO_VIDEO):
            os.makedirs(PATH_TO_VIDEO)
        if len(self.video_buffer) > fps:
            file_path = PATH_TO_VIDEO + "episode_" + str(episode) + ".avi"
            # do video saving stuff
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            writer = cv2.VideoWriter(file_path, fourcc, fps, (128, 128))
            # fill buffer of video writer
            for frame in self.video_buffer:
                writer.write(frame)
            writer.release() 
            self.video_buffer = []
        else:
            print("Warning: Trying to write log video without enough frames!")
