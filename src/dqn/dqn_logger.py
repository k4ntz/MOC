# helper to log with tensorboard

import torch
import os
import cv2
import numpy as np

from torch.utils.tensorboard import SummaryWriter

#from utils import draw_bounding_boxes

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

    # helper function to draw bounding box over z wheres 
    def draw_bounding_box(self, boxes_batch, indices):
        last_frame = self.video_buffer.pop()
        bb = (boxes_batch[0][:,:4] * (128, 128, 128, 128)).round().astype(int)
        # blue = dqn, green = ball, red = enemy, random = black
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (0,0,0)]
        thickness = 2
        for single_bb, color_i in zip(bb, indices):
            start = (single_bb[2], single_bb[0])
            end = (single_bb[3], single_bb[1])
            last_frame = cv2.rectangle(last_frame, start, end, color[color_i], thickness)
        self.video_buffer.append(last_frame)

    def save_video(self, model_name, fps=25.0):
        if not os.path.exists(PATH_TO_VIDEO):
            os.makedirs(PATH_TO_VIDEO)
        if len(self.video_buffer) > fps:
            file_path = PATH_TO_VIDEO + model_name + ".avi"
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
