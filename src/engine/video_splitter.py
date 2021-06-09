import sys
from rtpt import RTPT
import skvideo.io as skv

total_files = 16000
SEQUENCE_LENGTH = 10
# Create RTPT object
rtpt = RTPT(name_initials='TRo', experiment_name='VideoSplitter', max_iterations=total_files)

rtpt.start()

def split_videos(cfg):
    args_env_id = 'Pong-v0'

    for i in range(total_files):
        vid = skv.vread(f'videos/{args_env_id}/{args_env_id}_ep{episode_id:06}')
        for j in range(0, vid.shape[0] - SEQUENCE_LENGTH, 1):
            sub_vid = vid[j:j + SEQUENCE_LENGTH]
            skv.vwrite(f'data/ATARI/{args_env_id}/train/{args_env_id}_ep{i:06}', sub_vid)
        rtpt.step(subtitle=f"step={i}/{total_files}")



