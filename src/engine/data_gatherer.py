import argparse
import sys
import gym
from gym import wrappers, logger
from rtpt import RTPT
import matplotlib.pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import multiprocessing as mp

# Create RTPT object
rtpt = RTPT(name_initials='TR', experiment_name='DataGather', max_iterations=10)

# Start the RTPT tracking
rtpt.start()


class GymRenderer:
    def __init__(self, env, title="video"):
        self.env = env
        self.record = record
        self.video_rec = VideoRecorder(env, path=f"videos/{title}.mp4")

    def render(self):
        self.video_rec.capture_frame()

    def close_recorder(self):
        self.video_rec.close()
        self.video_rec.enabled = False


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def gather_agent(cfg, agent_id):
    args_env_id = 'Pong-v0'
    env = gym.make(args_env_id)

    agent = RandomAgent(env.action_space)

    episode_count = 10
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        renderer = GymRenderer(env, title=f'{args_env_id}_ep{agent_id * episode_count + i}')
        step = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            renderer.render()
            # plt.imshow(env.render('rgb_array'))
            # plt.savefig(f"{out_dir}{args_env_id}_ep{agent_id*episode_count+i}_st{step}.png")
            step += 1
            if done:
                break
        renderer.close_recorder()
        rtpt.step(subtitle=f"ag{agent_id} step={i}/{episode_count}")

    env.close()


NUM_PROCESSES = 16

def gather(cfg):
    for i in range(NUM_PROCESSES):
        proc = mp.Process(target=gather_agent, args=(cfg, env))
        proc.start()
