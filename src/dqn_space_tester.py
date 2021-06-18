# call with python space_converter_train.py --config configs/atari_ball_joint_v1.yaml resume True device 'cpu'

import os
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data_utils
import numpy as np
import pandas as pd

import dqn.utils as utils

from rtpt import RTPT
from tqdm import tqdm

from model import get_model
from utils import Checkpointer
from solver import get_optimizers
from eval.ap import convert_to_boxes
from dataset import get_dataset, get_dataloader

# space stuff
import os.path as osp

cfg, space_cfg = utils.get_config()

# if gpu is to be used
device = cfg.device


# helper function to get labels
def get_labels(table, i, cfg):
    start_index = i * cfg.train.batch_size
    end_index = (i + 1) * cfg.train.batch_size
    i_list = np.arange(start_index, end_index, 1).tolist()
    # cut out all except from pos of player enemy and ball
    # normalize image coordinates between 0 and 1
    cut_table = (table.iloc[i_list, 1:7]/200).fillna(0)
    table_t = torch.from_numpy(cut_table.to_numpy()).float().to(device)
    return table_t


# main training function 
def train(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Resume:', space_cfg.resume)
    if space_cfg.resume:
        print('Checkpoint:', space_cfg.resume_ckpt if space_cfg.resume_ckpt else 'last checkpoint')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)

    model = get_model(space_cfg)
    model = model.to(cfg.device)

    if len(space_cfg.gamelist) == 1:
        suffix = space_cfg.gamelist[0]
        print(f"Using SPACE Model on {suffix}")
    elif len(space_cfg.gamelist) == 2:
        suffix = space_cfg.gamelist[0] + "_" + cfg.gamelist[1]
        print(f"Using SPACE Model on {suffix}")
    else:
        print("Can't train")
        exit(1)
    checkpointer = Checkpointer(osp.join(space_cfg.checkpointdir, suffix, space_cfg.exp_name), max_num=4)
    use_cpu = 'cpu' in cfg.device

    checkpoint = checkpointer.load_last(space_cfg.resume_ckpt, model, None, None, use_cpu=cfg.device)
    # parallelize on multiple devices
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)
    
    # load data and cut to size of table 
    trainloader = get_dataloader(cfg, 'train')
    len_table = len(train_table) - ((len(train_table) % cfg.train.batch_size) * 2)
    len_data = round(len_table/cfg.train.batch_size)

    print('Start training')
    rtpt = RTPT(name_initials='DV', experiment_name=exp_name,
                max_iterations=max_epoch)
    rtpt.start()
    for epoch in range(start_epoch, max_epoch):
        start = time.perf_counter()
        with tqdm(total=len_data) as pbar:
            for i, data in tqdm(enumerate(trainloader)):
                end = time.perf_counter()
                data_time = end - start
                start = end
                imgs = data
                imgs = imgs.to(cfg.device)
                labels = get_labels(train_table, i, cfg)
                
                #TODO: get z stuff to compare with

                # end step
                end = time.perf_counter()
                batch_time = end - start
                if (global_step) % cfg.train.print_every == 0:
                    start = time.perf_counter()
                    end = time.perf_counter()
                    print(
                        'exp: {}, epoch: {}, iter: {}/{}, global_step: {}, loss: {:.2f}, batch time: {:.4f}s, log time: {:.4f}s'.format(
                            exp_name, epoch + 1, i + 1, len(trainloader), global_step, 0.00,
                            batch_time, end - start))
                    print(outputs[0])
                    print(labels[0])
                start = time.perf_counter()
                global_step += 1
                pbar.update(1)
                if i + cfg.train.batch_size >= len_data:
                    break
        rtpt.step()


if __name__ == '__main__':
    train(cfg)