import sys
from rtpt import RTPT
import skvideo.io as skv
import multiprocessing as mp

total_files = 2000

NUM_PROCESSES = 32

SEQUENCE_LENGTH = 4
# Create RTPT object
rtpt = RTPT(name_initials='TRo', experiment_name='VideoSplitter', max_iterations=total_files / NUM_PROCESSES)

rtpt.start()


def split_videos_job(cfg, job_id):
    args_env_id = 'SpaceInvaders-v0'
    files = total_files // NUM_PROCESSES
    for i in range(files * job_id, files * (job_id + 1)):
        try:
            vid = skv.vread(f'videos/{args_env_id}/{args_env_id}_ep{i:06}.mp4')
            print(i)
            for j in range(0, vid.shape[0] - SEQUENCE_LENGTH, 4):
                sub_vid = vid[j:j + SEQUENCE_LENGTH]
                if j % 12 == 0:
                    skv.vwrite(f'data/ATARI/{args_env_id}/test/{args_env_id}_ep{i:06}_seq{j:03}.mp4', sub_vid)
                elif j % 16 == 0:
                    skv.vwrite(f'data/ATARI/{args_env_id}/validation/{args_env_id}_ep{i:06}_seq{j:03}.mp4', sub_vid)
                else:
                    skv.vwrite(f'data/ATARI/{args_env_id}/train/{args_env_id}_ep{i:06}_seq{j:03}.mp4', sub_vid)
            if job_id == 0:
                rtpt.step(subtitle=f"step={i}/{total_files}")
        except:
            files += 0


def split_videos(cfg):
    for agent_id in range(NUM_PROCESSES):
        proc = mp.Process(target=split_videos_job, args=(cfg, agent_id))
        proc.start()
