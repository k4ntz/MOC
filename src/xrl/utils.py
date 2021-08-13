import os
import math
import torch
import cv2
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from argparse import ArgumentParser
from xrl.xrl_config import cfg

######################
######## INIT ########
######################

# function to get config
def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )

    parser.add_argument(
        '--space-config-file',
        type=str,
        default='configs/atari_ball_joint_v1.yaml',
        metavar='FILE',
        help='Path to SPACE config file'
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')

    # Seed
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return cfg

######################
###### PLOTTING ######
######################

plt.interactive(False)

# function to plot live while training
def plot_screen(env, episode, step, second_img=None):
    plt.figure(3)
    plt.title('Episode: ' + str(episode) + " - Step: " + str(step))
    plt.imshow(env.render(mode='rgb_array'),
           interpolation='none')
    if second_img is not None:
        plt.figure(2)
        plt.clf()
        plt.title('X - Episode: ' + str(episode) + " - Step: " + str(step))
        plt.imshow(second_img)
    plt.plot()
    plt.pause(0.0001)  # pause a bit so that plots are updated


# function to get integrated gradients
def get_integrated_gradients(ig, input, target_class):
    # get attributions and print
    attributions, approximation_error = ig.attribute(torch.tensor(input).unsqueeze(0).float(), 
        target=target_class, return_convergence_delta=True)
    #print(attributions)
    attr = attributions[0].cpu().detach().numpy()
    #print(attr_df)
    return attr


# function to get feature titles
def get_feature_titles():
    feature_titles = []
    for i in range(0, 3):
        feature_titles.append(str("obj" +  str(i) + " vel"))
        for j in range(0, 3):
            if j > i:
                feature_titles.append(str("x obj" + str(j) + " - obj" + str(i)))
                feature_titles.append(str("y obj" + str(j) + " - obj" + str(i)))
        for j in range(0, 3):
            if i != j:
                feature_titles.append(str("target y obj" + str(j) + " - obj" + str(i)))
                feature_titles.append(str("target x obj" + str(j) + " - obj" + str(i)))
    return feature_titles


# helper function to get integrated gradients of given features as plotable image
def plot_integrated_gradient_img(ig, exp_name, input, feature_titles, target_class, env, plot):
    attr = get_integrated_gradients(ig, input, target_class)
    attr_df = pd.DataFrame({"Values": attr},
                  index=feature_titles)
    #print(attr_df)
    env_img = env.render(mode='rgb_array')
    # plot both next to each other
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(env_img)
    sn.heatmap(attr_df, ax=ax2, vmin=-0.2, vmax=1)
    ax1.set_title(exp_name)
    fig.tight_layout()
    # convert fig to cv2 img
    # put pixel buffer in numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(mat, (480, 480), interpolation = cv2.INTER_AREA)
    if plot:
        plt.draw()
        plt.pause(0.0001)
    # clean up
    fig.clf()
    plt.close(fig)
    return resized


# helper function to plot igs of each feature over whole episode
def plot_igs(ig_sum, plot_titles):
    for i, igs in enumerate(ig_sum.T):
        plt.plot(igs)
        plt.xlabel("Steps")
        plt.ylabel("Integrated Gradient Value")
        if plot_titles is not None:
            plt.title(plot_titles[i])
        plt.show()
        plt.hist(igs, bins=20)
        plt.xlabel("Integrated Gradient Value")
        if plot_titles is not None:
            plt.title(plot_titles[i])
        plt.show()

###############################
##### PROCESSING FEATURES #####
###############################


# function to get raw features and order them by 
def get_raw_features(env_info, last_raw_features=None):
    # extract raw features
    labels = env_info["labels"]
    player = [labels["player_x"].astype(np.int16), 
            labels["player_y"].astype(np.int16)]
    enemy = [labels["enemy_x"].astype(np.int16), 
            labels["enemy_y"].astype(np.int16)]
    ball = [labels["ball_x"].astype(np.int16), 
            labels["ball_y"].astype(np.int16)]
    # set new raw_features
    raw_features = last_raw_features
    if raw_features is None:
        raw_features = [player, enemy, ball, None, None, None]
    else:
        raw_features = np.roll(raw_features, 3)
        raw_features[0] = player
        raw_features[1] = enemy
        raw_features[2] = ball
    return raw_features


# helper function to calc linear equation
def get_lineq_param(obj1, obj2):
    x = obj1
    y = obj2
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


# helper function to convert env info into custom list
# raw_features contains player x, y, ball x, y, oldplayer x, y, oldball x, y, 
# features are processed stuff for policy
def preprocess_raw_features(env_info, last_raw_features=None):
    features = []
    raw_features = get_raw_features(env_info, last_raw_features)
    for i in range(0, 3):
        obj1, obj1_past = raw_features[i], raw_features[i + 3]
        # when object has moved and has history
        if obj1_past is not None and not (obj1[0] == obj1_past[0] and obj1[1] == obj1_past[1]):
            # append velocity of itself
            features.append(math.sqrt((obj1_past[0] - obj1[0])**2 + (obj1_past[1] - obj1[1])**2))
        else:
            features.append(0)
        for j in range(0, 3):
            # apped all manhattan distances to all other objects
            # which are not already calculated
            if j > i:
                obj2 = raw_features[j]
                # append coord distances
                features.append(obj2[0] - obj1[0]) # append x dist
                features.append(obj2[1] - obj1[1]) # append y dist
        for j in range(0, 3):
            # calculate movement paths of all other objects
            # and calculate distance to its x and y intersection
            if i != j:
                obj2, obj2_past = raw_features[j], raw_features[j + 3]
                # if other object has moved
                if obj2_past is not None and not (obj2[0] == obj2_past[0] and obj2[1] == obj2_past[1]):
                    # append trajectory cutting points
                    m, c = get_lineq_param(obj2, obj2_past)
                    # now calc target pos
                    # y = mx + c substracted from its y pos
                    features.append(np.int16(m * obj1[0] + c) - obj1[1])
                    # x = (y - c)/m substracted from its x pos
                    features.append(np.int16((obj1[1] - c) / m)  - obj1[0])
                else:
                    features.append(0)
                    features.append(0)
    return raw_features, features


# helper function to get features
def do_step(env, action=1, last_raw_features=None):
    obs, reward, done, info = env.step(action)
    raw_features, features = preprocess_raw_features(info, last_raw_features)
    return raw_features, features, reward, done