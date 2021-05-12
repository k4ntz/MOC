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
from utils import Checkpointer, open_image, show_image, save_image, \
    corners_to_wh, draw_bounding_boxes, get_labels, place_labels
import os
import os.path as osp
from torch import nn
from torch.utils.data import Subset, DataLoader
from utils import draw_bounding_boxes
from torchvision.transforms.functional import crop
import torch
from eval.ap import read_boxes, convert_to_boxes, compute_ap, compute_counts
from tqdm import tqdm
from termcolor import colored
import pandas as pd


folder = "validation"

action = ["visu", "extract"][0]

folder_sizes = {"train": 50000, "test": 5000, "validation": 5000}
nb_images = folder_sizes[folder]

cfg, task = get_config()


EXTRACT_IMAGES = False
USE_FULL_SIZE = True

model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
use_cpu = 'cpu' in cfg.device
if cfg.resume_ckpt:
    checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, use_cpu=use_cpu)


table = pd.read_csv(f"../aiml_atari_data/rgb/{cfg.exp_name}/{folder}_labels.csv")
# print(colored("Please change number of images !", "red"))
all_z_what = []
all_labels = []
for i in tqdm(range(nb_images)):
    # img_path = f"../data/ATARI/SpaceInvaders-v0/train/{i:05}.jpg"
    # img_path = f"../data/ATARI/MsPacman-v0/train/{i:05}.jpg"
    img_path = f"../aiml_atari_data/space_like/{cfg.exp_name}/{folder}/{i:05}.png"
    img_path_fs = f"../aiml_atari_data/rgb/{cfg.exp_name}/{folder}/{i:05}.png"
    image = open_image(img_path).to(cfg.device)
    image_fs = open_image(img_path_fs).to(cfg.device)


    # TODO: treat global_step in a more elegant way
    loss, log = model(image, global_step=100000000)
    # (B, N, 4), (B, N, 1), (B, N, D)
    z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
    # (B, N, 4), (B, N), (B, N)
    z_where = z_where.detach().cpu()

    z_pres_prob = z_pres_prob.detach().cpu().squeeze()
    z_pres = z_pres_prob > 0.5
    # import ipdb; ipdb.set_trace()


    z_what_pres = z_what[z_pres.unsqueeze(0)]

    boxes_batch = convert_to_boxes(z_where, z_pres.unsqueeze(0), z_pres_prob.unsqueeze(0))
    # boxes_batch, zwhats = boxes_and_what(z_where, z_pres.unsqueeze(0), z_pres_prob.unsqueeze(0), z_what)


    labels = get_labels(table.iloc[[i]], boxes_batch)
    if action == "visu":
        image = place_labels(labels, boxes_batch, image_fs[0])
        image = draw_bounding_boxes(image, boxes_batch, labels)
        show_image(image)
        # show_image(image_fs[0])
    assert z_what_pres.shape[0] == labels.shape[0]
    all_z_what.append(z_what_pres.detach().cpu())
    all_labels.append(labels.detach().cpu())


# all_z_what = torch.cat(all_z_what)
# all_labels = torch.cat(all_labels)

if action == "extract":
    torch.save(all_z_what, f"labeled/z_what_{folder}.pt")
    torch.save(all_labels, f"labeled/labels_{folder}.pt")

# import ipdb; ipdb.set_trace()
    # image = (image[0] * 255).round().to(torch.uint8)  # for draw_bounding_boxes
    # image_fs = (image_fs[0] * 255).round().to(torch.uint8)  # for draw_bounding_boxes

    # image_fs = place_labels(labels, boxes_batch, image_fs)



    # if USE_FULL_SIZE:
    #     show_image(draw_bounding_boxes(image_fs, boxes_batch))
    # else:
    #     show_image(draw_bounding_boxes(image, boxes_batch))

    # exit()
    # if EXTRACT_IMAGES:
    #     for j, bbox in enumerate(torch.tensor(bb)):
    #         top, left, height, width = corners_to_wh(bbox)
    #         cropped = crop(image.to("cpu"), top, left, height, width)
    #         # show_image(cropped)
    #         save_image(cropped, img_path, j)
