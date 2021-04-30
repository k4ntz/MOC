from engine.utils import get_config
from model import get_model
from utils import Checkpointer
import os.path as osp
import torch
from tqdm import tqdm
from termcolor import colored
import pandas as pd


folder = "test"


folder_sizes = {"train": 50000, "test": 5000, "validation": 5000}
nb_images = folder_sizes[folder]

cfg, task = get_config()

model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
use_cpu = 'cpu' in cfg.device
if cfg.resume_ckpt:
    checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, use_cpu=use_cpu)

import ipdb; ipdb.set_trace()
