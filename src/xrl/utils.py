import math
import torch
import cv2

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    attr_df = pd.DataFrame({"Values": attr},
                  index=["Target Y - Player Y", "Player X - Ball X", "0", "Ball X - Enemy X", "Ball Y - Enemy Y", "Ball Speed"])
    #print(attr_df)
    return attr_df


# helper function to get integrated gradients of given features as plotable image
def plot_integrated_gradient_img(ig, exp_name, input, target_class, env, plot):
    attr_df = get_integrated_gradients(ig, input, target_class)
    #print(attr_df)
    env_img = env.render(mode='rgb_array')
    # plot both next to each other
    plt.clf()
    plt.cla()
    plt.close()
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(env_img)
    sn.heatmap(attr_df, ax=ax2, vmin=-0.2, vmax=1)
    plt.title(exp_name)
    plt.tight_layout()
    # convert fig to cv2 img
    # put pixel buffer in numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(mat, (480, 480), interpolation = cv2.INTER_AREA)
    if plot:
        plt.plot()
        plt.pause(0.0001)
    return resized


# helper function to calc linear equation
def get_target_x(x1, x2, y1, y2, player_x):
    x = [x1, x2]
    y = [y1, y2]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # now calc target pos
    # y = mx + c
    return np.int16(m * player_x + c)


# helper function to convert env info into custom list
# raw_features contains player x, y, ball x, y, oldplayer x, y, oldball x, y, 
# features are processed stuff for policy
def preprocess_raw_features(env_info, last_raw_features=None):
    features = []
    norm_factor = 250
    # extract raw features
    labels = env_info["labels"]
    player_x = labels["player_x"].astype(np.int16)
    player_y = labels["player_y"].astype(np.int16)
    enemy_x = labels["enemy_x"].astype(np.int16)
    enemy_y = labels["enemy_y"].astype(np.int16)
    ball_x = labels["ball_x"].astype(np.int16)
    ball_y = labels["ball_y"].astype(np.int16)
    # set new raw_features
    raw_features = last_raw_features
    if raw_features is None:
        raw_features = [player_x, player_y, ball_x, ball_y, enemy_x, enemy_y
            ,np.int16(0), np.int16(0), np.int16(0), np.int16(0), np.int16(0), np.int16(0)]
        features.append(0)
    else:
        # move up old values in list
        raw_features = np.roll(raw_features, 6)
        raw_features[0] = player_y
        raw_features[1] = player_y
        raw_features[2] = ball_x
        raw_features[3] = ball_y  
        raw_features[4] = enemy_x
        raw_features[5] = enemy_y 
        # calc target point and put distance in features
        target_y = get_target_x(raw_features[6], ball_x, raw_features[7], ball_y, player_x) 
        features.append((target_y - player_y)/ norm_factor)
    # append other distances
    features.append((player_x - ball_x)/ norm_factor)# distance x ball and player
    features.append(0) 
    # not needed, bc target pos y is already calculated
    # features.append((player_y - ball_y)/ norm_factor)# distance y ball and player
    features.append((ball_x - enemy_x)/ norm_factor) # distance x ball and enemy
    features.append((ball_y - enemy_y)/ norm_factor) # distance y ball and enemy
    # euclidean distance between old and new ball coordinates to represent current speed per frame
    features.append(math.sqrt((ball_x - raw_features[8])**2 + (ball_y - raw_features[9])**2) / 25) 
    return raw_features, features


# helper function to get features
def do_step(env, action=1, last_raw_features=None):
    if action == 1:
        action = 2
    elif action == 2:
        action = 5
    obs, reward, done, info = env.step(action)
    raw_features, features = preprocess_raw_features(info, last_raw_features)
    return raw_features, features, reward, done