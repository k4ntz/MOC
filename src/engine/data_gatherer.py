import argparse
import sys
import gym
from gym import wrappers, logger
from rtpt import RTPT
import matplotlib.pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import multiprocessing as mp

episode_count = 1000
# Create RTPT object
rtpt = RTPT(name_initials='TRo', experiment_name='DataGatherer', max_iterations=episode_count)

# Start the RTPT tracking
rtpt.start()


class GymRenderer:
    def __init__(self, env, title="video"):
        self.env = env
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


    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        episode_id = agent_id * episode_count + i
        renderer = GymRenderer(env, title=f'{args_env_id}_ep{episode_id:06}')
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
        if agent_id == 0:
            rtpt.step(subtitle=f"step={i}/{episode_count}")

    env.close()


NUM_PROCESSES = 16


def gather(cfg):
    for agent_id in range(NUM_PROCESSES):
        proc = mp.Process(target=gather_agent, args=(cfg, agent_id))
        proc.start()

    gather_agent(cfg, NUM_PROCESSES)
