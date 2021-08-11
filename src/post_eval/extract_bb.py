import os, sys
_curent_dir = os.getcwd()
for _cd in [_curent_dir, _curent_dir + "/post_eval"]:
    if _cd not in sys.path:
        sys.path.append(_cd)

import os.path as osp
import matplotlib.pyplot as plt
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
from torch import nn
from torch.utils.data import Subset, DataLoader
from utils import draw_bounding_boxes
from torchvision.transforms.functional import crop
import torch
from eval.ap import read_boxes, convert_to_boxes, compute_ap, compute_counts
from tqdm import tqdm
from termcolor import colored
import pandas as pd
from PIL import Image
import PIL

folder = "validation"

action = ["visu", "extract"][1]

folder_sizes = {"train": 50000, "test": 5000, "validation": 50}
nb_images = folder_sizes[folder]
cfg, task = get_config()

TIME_CONSISTENCY = True
EXTRACT_IMAGES = False
USE_FULL_SIZE = True

model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name),
                            max_num=cfg.train.max_ckpt, load_time_consistency=TIME_CONSISTENCY)
use_cpu = 'cpu' in cfg.device
if cfg.resume:
    checkpoint = checkpointer.load_last('', model, None, None, use_cpu=use_cpu)

table = pd.read_csv(f"../aiml_atari_data/rgb/{cfg.gamelist[0]}/{folder}_labels.csv")
# print(colored("Please change number of images !", "red"))
all_z_what = []
all_labels = []
print(table)


def process_image(log, image_rgb, idx):
    # (B, N, 4), (B, N, 1), (B, N, D)
    z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
    # (B, N, 4), (B, N), (B, N)
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu().squeeze()
    z_pres = z_pres_prob > 0.5
    z_what_pres = z_what[z_pres.unsqueeze(0)]
    boxes_batch = np.array(convert_to_boxes(z_where, z_pres.unsqueeze(0), z_pres_prob.unsqueeze(0))).squeeze()
    labels = get_labels(table.iloc[[idx]], boxes_batch)
    if action == "visu":
        visu = place_labels(labels, boxes_batch, image_rgb)
        visu = draw_bounding_boxes(visu, boxes_batch, labels)
        show_image(visu)
        exit()
        # show_image(image_fs[0])
    assert z_what_pres.shape[0] == labels.shape[0]
    all_z_what.append(z_what_pres.detach().cpu())
    all_labels.append(labels.detach().cpu())


def img_path_to_tensor(path):
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
    image = np.array(pil_img)
    return torch.from_numpy(image / 255).permute(2, 0, 1).float()


for i in tqdm(range(0, nb_images, 4 if TIME_CONSISTENCY else 1)):
    if TIME_CONSISTENCY:
        fn = [f"../aiml_atari_data/space_like/{cfg.gamelist[0]}/{folder}/{i + j:05}.png" for j in range(4)]
        image = torch.stack([img_path_to_tensor(f) for f in fn]).to(cfg.device).unsqueeze(0)
    else:
        img_path = f"../aiml_atari_data/space_like/{cfg.gamelist[0]}/{folder}/{i:05}.png"
        image = open_image(img_path).to(cfg.device)
    img_path_fs = f"../aiml_atari_data/rgb/{cfg.gamelist[0]}/{folder}/{i:05}.png"
    image_fs = open_image(img_path_fs).to(cfg.device)

    # TODO: treat global_step in a more elegant way
    with torch.no_grad():
        loss, space_log = model(image, global_step=100000000)
    if TIME_CONSISTENCY:
        for j in range(4):
            process_image(space_log['space_log'][j], image_fs, i + j)
    else:
        process_image(space_log, image_fs, i)

# all_z_what = torch.cat(all_z_what)
# all_labels = torch.cat(all_labels)

if action == "extract":
    if not os.path.exists(f"labeled/{cfg.exp_name}"):
        os.makedirs(f"labeled/{cfg.exp_name}")
    torch.save(all_z_what, f"labeled/{cfg.exp_name}/z_what_{folder}.pt")
    torch.save(all_labels, f"labeled/{cfg.exp_name}/labels_{folder}.pt")
    print(f"Extracted z_whats and saved it in labeled/{cfg.exp_name}/")

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
