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

from rtpt import RTPT
from tqdm import tqdm

from model import get_model
from engine.utils import get_config
from utils import Checkpointer
from solver import get_optimizers
from eval.ap import convert_to_boxes
from dataset import get_dataset, get_dataloader
from dqn.test.test_nn import SPACE_Classifier

# space stuff
import os.path as osp


cfg, task = get_config()

# if gpu is to be used
device = cfg.device

PATH_TO_OUTPUTS = os.getcwd() + "/dqn/test/checkpoints/"

# helper function to save model
def save_model(model_name, model, optimizer, global_step):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
            }, PATH_TO_OUTPUTS + model_name)

# helper function to load model
def load_model(model_name, model, optimizer):
    model_path = PATH_TO_OUTPUTS + model_name
    if not os.path.isfile(model_path):
        print("{} does not exist".format(model_path))
        return False
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = checkpoint['global_step']
    return model, optimizer, global_step

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
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)

    if len(cfg.gamelist) >= 10:
        print("Using SPACE Model on every game")
        suffix = 'all'
    elif len(cfg.gamelist) == 1:
        suffix = cfg.gamelist[0]
        print(f"Using SPACE Model on {suffix}")
    elif len(cfg.gamelist) == 2:
        suffix = cfg.gamelist[0] + "_" + cfg.gamelist[1]
        print(f"Using SPACE Model on {suffix}")
    else:
        print("Can't train")
        exit(1)

    start_epoch = 0
    max_epoch = 10000
    start_iter = 0
    max_iter = 1000000000
    global_step = 0

    folder = "train"
    game_name = "Pong-v0" #cfg.gamelist[0]
    train_table = pd.read_csv(f"../data/ATARI/{game_name}/{folder}_labels.csv")
    exp_name = cfg.exp_name + "_tester"

    model_name = exp_name + ".pth"
    full_path = PATH_TO_OUTPUTS + model_name
   
    model = SPACE_Classifier()
    model = model.to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if os.path.exists(PATH_TO_OUTPUTS):
        if os.path.isfile(full_path):
            print("Loading " + full_path)
            model, optimizer, global_step = load_model(model_name, model, optimizer)
    else:
        os.makedirs(PATH_TO_OUTPUTS)


    # parallelize on multiple devices
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)

    trainloader = get_dataloader(cfg, 'train')    

    print('Start training')
    rtpt = RTPT(name_initials='DV', experiment_name=exp_name,
                max_iterations=max_iter)

    for epoch in range(start_epoch, max_epoch):
        start = time.perf_counter()
        with tqdm(total=len(train_table)) as pbar:
            for i, data in tqdm(enumerate(trainloader)):
                end = time.perf_counter()
                data_time = end - start
                start = end
                imgs = data
                imgs = imgs.to(cfg.device)
                labels = get_labels(train_table, i, cfg)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # end step
                end = time.perf_counter()
                batch_time = end - start
                if (global_step) % cfg.train.print_every == 0:
                    start = time.perf_counter()
                    end = time.perf_counter()
                    print(
                        'exp: {}, epoch: {}, iter: {}/{}, global_step: {}, loss: {:.2f}, batch time: {:.4f}s, log time: {:.4f}s'.format(
                            exp_name, epoch + 1, i + 1, len(trainloader), global_step, loss.mean(),
                            batch_time, end - start))
                if (global_step) % cfg.train.save_every == 0:
                    save_model(model_name, model, optimizer, global_step)
                    print('Saving checkpoint takes {:.4f}s.'.format(time.perf_counter() - start))
                start = time.perf_counter()
                global_step += 1
                pbar.update(1)
                if (i+1) * 12 > len(train_table):
                    break
        rtpt.step()


if __name__ == '__main__':
    train(cfg)