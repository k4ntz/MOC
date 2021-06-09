import sys
from rtpt import RTPT
import skvideo.io as skv

total_files = 16000

NUM_PROCESSES = 32

SEQUENCE_LENGTH = 10
# Create RTPT object
rtpt = RTPT(name_initials='TRo', experiment_name='VideoSplitter', max_iterations=total_files / NUM_PROCESSES)

rtpt.start()


def split_videos_job(cfg, job_id):
    args_env_id = 'Pong-v0'
    files = total_files // NUM_PROCESSES
    for i in range(files * job_id, files * (job_id + 1)):
        vid = skv.vread(f'videos/{args_env_id}/{args_env_id}_ep{i:06}.mp4')
        for j in range(0, vid.shape[0] - SEQUENCE_LENGTH, 4):
            sub_vid = vid[j:j + SEQUENCE_LENGTH]
            if j % 3 == 0:
                skv.vwrite(f'data/ATARI/{args_env_id}/test/{args_env_id}_ep{i:06}_seq{j:03}.mp4', sub_vid)
            elif j % 4 == 0:
                skv.vwrite(f'data/ATARI/{args_env_id}/validation/{args_env_id}_ep{i:06}_seq{j:03}.mp4', sub_vid)
            else:
                skv.vwrite(f'data/ATARI/{args_env_id}/train/{args_env_id}_ep{i:06}_seq{j:03}.mp4', sub_vid)
        if job_id == 0:
            rtpt.step(subtitle=f"step={i}/{total_files}")


def split_videos(cfg):
    for agent_id in range(NUM_PROCESSES):
        proc = mp.Process(target=gather_agent, args=(cfg, agent_id))
        proc.start()
