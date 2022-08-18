import joblib
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import gym
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from rtpt import RTPT
from engine.utils import get_config
from model import get_model
from utils import Checkpointer
from solver import get_optimizers
from PIL import Image
from torchvision import transforms

# load config
cfg, task = get_config()

# stuff to save and load policy model
PATH_TO_OUTPUTS = os.getcwd() + "/classifiers/"
if not os.path.exists(PATH_TO_OUTPUTS):
    os.makedirs(PATH_TO_OUTPUTS)
# lambda name function to create policy model file path
model_name = lambda training_name : PATH_TO_OUTPUTS + training_name + "_policy_model.pth"


def clean_scene(scene):
    empty_keys = []
    for key, val in scene.items():
        for i, z_where in reversed(list(enumerate(val))):
            if z_where[3] < -0.75:
                scene[key].pop(i)
        if len(val) == 0:
            empty_keys.append(key)
    for key in empty_keys:
        scene.pop(key)
    scene_list = []
    for el in [1, 2, 3]:
        if el in scene:
            scene_list.append(scene[el][0][2:])
        else:
            scene_list.append([0, 0]) # object not found
    return scene_list

cfg.device_ids = [0]

# model
class Policy(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(n_inputs, 128)
        self.affine2 = nn.Linear(128, n_actions)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

# create env
env_name = cfg.gamelist[0]
env = gym.make(env_name)
env.reset()
nb_action = env.action_space.n

# get models
# TODO: make dynamic feature length
spacetime_model = get_model(cfg)
policy = Policy(6, nb_action)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
i_episode = 0

# move to cuda when possible
use_cuda = 'cuda' in cfg.device
if use_cuda:
    spacetime_model = spacetime_model.to('cuda:0')

checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,
                            load_time_consistency=cfg.load_time_consistency, add_flow=cfg.add_flow)
optimizer_fg, optimizer_bg = get_optimizers(cfg, spacetime_model)

if cfg.resume:
    checkpoint = checkpointer.load_last(cfg.resume_ckpt, spacetime_model, optimizer_fg, optimizer_bg, cfg.device)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step'] + 1



space = spacetime_model.space
z_classifier_path = f"classifiers/{cfg.exp_name}_z_what_classifier.joblib.pkl"
z_classifier = joblib.load(z_classifier_path)
# x is the image on device as a Tensor, z_classifier accepts the latents,
# only_z_what control if only the z_what latent should be used (see docstring)

transformation = transforms.ToTensor()

# select action function
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


# save model helper function
def save_policy(training_name, policy, episode, optimizer):
    if not os.path.exists(PATH_TO_OUTPUTS):
        os.makedirs(PATH_TO_OUTPUTS)
    model_path = model_name(training_name)
    print("Saving {}".format(model_path))
    torch.save({
            'policy': policy.state_dict(),
            'episode': episode,
            'optimizer': optimizer.state_dict()
            }, model_path)


# load model
def load_model(model_path, policy, optimizer=None):
    print("{} does exist, loading ... ".format(model_path))
    checkpoint = torch.load(model_path)
    policy.load_state_dict(checkpoint['policy'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    i_episode = checkpoint['episode']
    return policy, optimizer, i_episode

# load if exists
model_path = model_name(cfg.exp_name)
if os.path.isfile(model_path):
    policy, optimizer, i_episode = load_model(model_path, policy, optimizer)


# helper function to get scene
def get_scene(observation, space):
    img = Image.fromarray(observation[:, :, ::-1], 'RGB').resize((128, 128), Image.ANTIALIAS)
    #x = torch.moveaxis(torch.tensor(np.array(img)), 2, 0)
    x = transformation(img)
    if use_cuda:
        x = x.cuda()
    scene = space.scene_description(x, z_classifier=z_classifier,
                                    only_z_what=True)  # class:[(w, h, x, y)]
    scene_list = clean_scene(scene)
    return scene_list


# env = Atari(env_name)
max_episode = 50000
# episode loop
running_reward = None
# training loop
rtpt = RTPT(name_initials='DV/QT/TR', experiment_name=cfg.exp_name,
                max_iterations=cfg.train.num_episodes)
rtpt.start()
while i_episode < max_episode:
    state, ep_reward = env.reset(), 0
    action = np.random.randint(nb_action)
    # env step loop
    for t in range(1, 10000):  # Don't infinite loop while learning
        observation, reward, done, info = env.step(action)
        ep_reward += reward
        policy.rewards.append(reward)
        scene_list = get_scene(observation, space)
        # flatten scene list
        scene_list = np.asarray([item for sublist in scene_list for item in sublist])
        action = select_action(scene_list)
        print('Episode {}\tLast reward: {:.2f}\tRunning reward: {:.2f}\tSteps: {}       '.format(
            i_episode, ep_reward, running_reward, t), end="\r")
        if done:
            break
    # replace first running reward with last reward for loaded models
    if running_reward is None:
        running_reward = ep_reward
    else:
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    # finish up episode and optimize
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R            # gamma
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    # save if necessary
    if (i_episode + 1) % 100 == 0:
        save_policy(cfg.exp_name, policy, i_episode + 1, optimizer)
    # finish episode
    i_episode += 1
    rtpt.step()
