import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import scipy
 
from PIL import Image
import cv2 as cv
import itertools

folder_sizes = {"train": 50, "test": 5000, "validation": 5000}
parser = argparse.ArgumentParser(
    description='Create the dataset for a specific game of ATARI Gym')
parser.add_argument('-g', '--game', type=str, help='An atari game',
                    # default='SpaceInvaders')
                    # default='Pong')
                    # default='Tennis')
                    default='MsPacman')
parser.add_argument('-d', '--draw_images', type=str, help='Should we render the flow?',
                    # default='SpaceInvaders')
                    # default='Pong')
                    # default='Tennis')
                    default='')
parser.add_argument('-f', '--folder', type=str, choices=folder_sizes.keys(),
                    required=True,
                    help='folder to write to: train, test or validation')
args = parser.parse_args()
some = vars(args)
for flow_folder in ['median_delta', 'median_delta_vis']:
    if not os.path.exists(f'../aiml_atari_data/{flow_folder}/{args.game}-v0/{args.folder}/'):
        os.makedirs(f'../aiml_atari_data/{flow_folder}/{args.game}-v0/{args.folder}/')
pbar = tqdm(total=folder_sizes[args.folder])
TRAIL = 4
# all_img = np.stack([cv.imread(f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{t_idx:05}.png') for t_idx in range(0, 50000, 10)])
# cv.imwrite(f'../aiml_atari_data/median_delta/{args.game}-v0/{args.folder}/the_median.png', np.median(all_img, axis=0))
pbar = tqdm(total=folder_sizes[args.folder])

def ring_kernel(lst):
    start = np.array([[lst[0]]])
    for v in lst[1:]:
        start = np.pad(start, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=v)
    return start

for idx in range(folder_sizes[args.folder]):
    trail = [(idx - i) % folder_sizes[args.folder] for i in range(TRAIL)]
    trail = np.stack([np.array(
        Image.open(f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{t_idx:05}.png').convert('RGB')) for
                      t_idx in trail])
    frame = trail[0]
    median_delta = np.abs(frame - np.median(trail, axis=0))
    median_delta = np.max(median_delta, axis=-1)
    # np.save(f'../aiml_atari_data/median_delta/{args.game}-v0/{args.folder}/{idx:05}', median_delta)
    pbar.update(1)
    if args.draw_images:
        median_delta += abs(np.random.randn(*median_delta.shape)) * 0.0001
        flow = median_delta / median_delta.max()
        from skimage.segmentation import felzenszwalb
        from skimage.morphology import (disk, square)
        from skimage.morphology import (erosion, dilation, opening, closing, white_tophat, skeletonize)
        from skimage.filters import rank

        output = cv.imread(f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{idx:05}.png')
        flow = (flow > 0.1) * 255
        flow = closing(flow, square(3))
        for i in range(2):
            vis_flow = (4 * flow / 255 + 1) / 5
            output_flow = output * vis_flow[..., None]
            output_flow = ndimage.zoom(output_flow, (3, 3, 1), order=1)
            cv.imwrite(f'../aiml_atari_data/median_delta_vis/{args.game}-v0/{args.folder}/median_{idx:05}_{i}.png',
                       output_flow)
            flow = rank.mean(flow, disk(2))
            flow = (flow > 200) * 255
        flow = median_delta / median_delta.max()
        grid_width = 4
        avg_pool = nn.AvgPool2d(grid_width + 2, grid_width, padding=1, count_include_pad=False)
        grid_flow = avg_pool(torch.tensor(flow).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        grid_flow_idx = (grid_flow > grid_flow.mean()).nonzero()
        new_grid_flow = np.zeros_like(grid_flow)
        for x, y in itertools.product(range(0, 32, 4), range(0, 32, 4)):
            x_slice = slice(max(0, x - 2), x + 3)
            y_slice = slice(max(0, y - 2), y + 3)
            neighborhood = grid_flow[x_slice, y_slice]
            local_max = neighborhood.max()
            new_grid_flow[x_slice, y_slice] = neighborhood * (neighborhood >= local_max)
        max_pool = nn.MaxPool2d(2, 2)
        grid_flow = max_pool(torch.tensor(new_grid_flow).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        grid_flow = grid_flow > grid_flow.mean()
        grid_flow = grid_flow.repeat(8, axis=0).repeat(8, axis=1)
        grid_flow = (4 * grid_flow + 1) / 3
        output_avg = output * grid_flow[..., None]
        output_avg = ndimage.zoom(output_avg, (3, 3, 1), order=1)
        cv.imwrite(f'../aiml_atari_data/median_delta_vis/{args.game}-v0/{args.folder}/avg_{idx:05}.png', output_avg)
        flow = median_delta / median_delta.max()
        flow = np.pad(flow, pad_width=((3, 4), (3, 4)), mode='constant',
                           constant_values=0)
        grid_width = 8
        avg_pool = nn.AvgPool2d(grid_width, 1, padding=0, count_include_pad=False)
        grid_flow = avg_pool(torch.tensor(flow).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        result_flow = np.zeros_like(flow)
        base_pad = 10
        grid_flow = np.pad(grid_flow, pad_width=((base_pad, base_pad), (base_pad, base_pad)), mode='constant',
                           constant_values=0)
        while grid_flow.max() > 0.07:
            x, y = np.unravel_index(np.argmax(grid_flow), grid_flow.shape)
            result_flow[x - base_pad, y - base_pad] = 1
            x_slice = slice(x - 7, x + 8)
            y_slice = slice(y - 7, y + 8)
            grid_flow[x_slice, y_slice] = grid_flow[x_slice, y_slice] * ring_kernel([0, 0.1, 0.1] + [0.5] * 5)
        cv.imwrite(f'../aiml_atari_data/median_delta_vis/{args.game}-v0/{args.folder}/sel_{idx:05}-r.png', result_flow * 255)
        max_pool = nn.MaxPool2d(8, 8)
        grid_flow = max_pool(torch.tensor(result_flow).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        grid_flow = (4 * grid_flow + 1) / 3
        grid_flow = grid_flow.repeat(8, axis=0).repeat(8, axis=1)
        output_avg = output * grid_flow[..., None]
        output_avg = ndimage.zoom(output_avg, (3, 3, 1), order=1)
        cv.imwrite(f'../aiml_atari_data/median_delta_vis/{args.game}-v0/{args.folder}/sel_{idx:05}.png', output_avg)
