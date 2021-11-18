import matplotlib.pyplot as plt
import sys
import numpy as np
from engine.utils import get_config
from engine.train import train
from engine.eval import eval
from engine.show import show
from model import get_model
from vis import get_vislogger
from dataset import get_dataset, get_dataloader
from augmentation import Checkpointer, open_image, show_image, save_image, \
    corners_to_wh, colors, image_pca
import os
import os.path as osp
from torch import nn
from torch.utils.data import Subset, DataLoader
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import crop
import torch
from eval.ap import read_boxes, convert_to_boxes, compute_ap, compute_counts
from tqdm import tqdm

cfg, task = get_config()

model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
if cfg.resume_ckpt:
    checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)

img_path = f"../data/ATARI/MsPacman-v0/train/00001.jpg"
image = open_image(img_path).to(cfg.device)

# TODO: treat global_step in a more elegant way
loss, log = model(image, global_step=100000000)

# (B, N, 4), (B, N, 1), (B, N, D)
z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
# (B, N, 4), (B, N), (B, N)
z_where = z_where.detach().cpu()

z_pres_prob = z_pres_prob.detach().cpu().squeeze()
z_pres = z_pres_prob > 0.5

z_what_pres = z_what[z_pres.unsqueeze(0)]

boxes_batch = convert_to_boxes(z_where, z_pres.unsqueeze(0), z_pres_prob.unsqueeze(0))


image = (image[0] * 255).round().to(torch.uint8) # for draw_bounding_boxes
bb = (boxes_batch[0][:,:-1] * 128).round()
bb[:,[0, 1, 2, 3]] = bb[:,[2, 0, 3, 1]] # swapping xmin <-> ymax ... etc
image_pca(image, torch.tensor(bb), z_what_pres, False)
