import numpy as np
import cv2 as cv
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import scipy
from scipy import ndimage

folder_sizes = {"train": 50000, "test": 5000, "validation": 5000}
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
for flow_folder in ['flow', 'frame_flow', 'bgr_flow']:
    if not os.path.exists(f'../aiml_atari_data/{flow_folder}/{args.game}-v0/{args.folder}/'):
        os.makedirs(f'../aiml_atari_data/{flow_folder}/{args.game}-v0/{args.folder}/')
pbar = tqdm(total=folder_sizes[args.folder])
for idx in range(folder_sizes[args.folder] - 1):
    # Note we are computing flow in reverse, as such that flow[x, y] != 0 iff the pixel was moved into
    # that position from [x + flow[x, y][0], y - flow[x, y][1]]
    frame1 = cv.imread(f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{idx + 1:05}.png')
    hsv = np.zeros_like(frame1)
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    frame2 = cv.imread(f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{idx:05}.png')
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    # TODO: Parameters
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.6, 5, 15, 30, 5, 1.5, 0)
    np.save(f'../aiml_atari_data/flow/{args.game}-v0/{args.folder}/{idx + 1:05}', flow)
    pbar.update(1)
    if args.draw_images:
        flow = np.load(f'../aiml_atari_data/flow/{args.game}-v0/{args.folder}/{idx + 1:05}.npy')
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        print(f'Rendering {idx + 1:05}.png')
        cv.imwrite(f'../aiml_atari_data/bgr_flow/{args.game}-v0/{args.folder}/{idx + 1:05}.png', bgr)
        flow = (flow * flow).sum(axis=2)
        flow = flow / flow.max()
        vis_flow = (100 * flow + 1) / 3
        output = cv.imread(f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{idx:05}.png')
        grid_width = 128 // 16
        avg_pool = nn.AvgPool2d(grid_width + 2, grid_width, padding=1, count_include_pad=False)
        grid_flow = avg_pool(torch.tensor(flow).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        grid_flow = grid_flow > grid_flow.mean()
        grid_flow = grid_flow.repeat(8, axis=0).repeat(8, axis=1)
        output_avg = output * grid_flow[..., None]
        output_avg = ndimage.zoom(output_avg, (3, 3, 1), order=1)
        output = output * vis_flow[..., None]
        output = ndimage.zoom(output, (3, 3, 1), order=1)
        cv.imwrite(f'../aiml_atari_data/frame_flow/{args.game}-v0/{args.folder}/{idx + 1:05}.png', output)
        cv.imwrite(f'../aiml_atari_data/frame_flow/{args.game}-v0/{args.folder}/{idx + 1:05}_avg.png', output_avg)
