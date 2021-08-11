import numpy as np
import cv2 as cv
import os
import argparse

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

for idx in range(folder_sizes[args.folder] - 1):
    path = f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{idx:05}.png'
    print(path)
    frame1 = cv.imread(path)
    hsv = np.zeros_like(frame1)
    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    frame2 = cv.imread(f'../aiml_atari_data/space_like/{args.game}-v0/{args.folder}/{idx + 1:05}.png')
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    hsv[..., 1] = 255
    flow = cv.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    np.save(f'../aiml_atari_data/flow/{args.game}-v0/{args.folder}/{idx + 1:05}', flow)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    if args.draw_images:
        print(f'Rendering {idx + 1:05}.png')
        cv.imwrite(f'../aiml_atari_data/frame_flow/{args.game}-v0/{args.folder}/{idx + 1:05}.png', frame2)
        cv.imwrite(f'../aiml_atari_data/bgr_flow/{args.game}-v0/{args.folder}/{idx + 1:05}.png', bgr)
