import numpy as np
from PIL import Image
import cv2 as cv
import torch
from .motion_processing import process_motion_to_latents, save_motion
from .mode_util import vector_mode


def save(trail, output_path, visualizations=None, mode=None, space_frame=None):
    """
    Computes the flow from the last 4 images in the stack
    :param trail: [>=10, H, W, 3] array
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    :param mode: Optional[nd.array<H, W>] if the mode is known from other sources than the trail
    :param space_frame: [>=10, H_2, W_2, 3] array resized for SPACE
    """
    if mode is None:
        mode = np.apply_along_axis(lambda x: vector_mode(x), axis=0,
                                   arr=np.moveaxis(trail[:-4], 3, 1).reshape(-1, *trail.shape[1:3]))
        mode = np.moveaxis(mode, 0, 2)
    for i, frame in enumerate(trail[-4:]):
        save_frame(frame, mode, output_path.format(i), visualizations=visualizations, space_frame=space_frame[i])


def save_coinrun(trail, output_path, visualizations=None, space_frame=None):
    frame_modes = [align(frame, prev) for frame, prev in zip(trail[-4:], trail[-5:])]

    for i, (frame, frame_mode) in enumerate(zip(trail[-4:], frame_modes)):
        save_frame(frame, frame_mode, output_path.format(i), visualizations=visualizations, space_frame=space_frame[i])

def save_frame(frame, mode, output_path, visualizations=None, space_frame=None):
    """
    Computes the flow from frame2 to frame1
    This inverse oder is done as such that the vectors are high at the spot in frame2 were the moving object is,
    i.e. inverse movement to map more directly to z_pres, Saved object is [H, W] in [0, 1]
    :param frame: [H, W, 3] array
    :param mode: [H, W, 3] array describing the mode
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    :param space_frame: [H_2, W_2, 3] array resized for SPACE
    """
    mode_delta = np.abs(frame.astype(np.int32) - mode.astype(np.int32))
    mode_delta = np.max(mode_delta, axis=-1)
    # print("Warning:temporary mode modification not yet removed!")
    # mode_delta = np.maximum(np.zeros_like(mode_delta), mode_delta - 60).astype(np.uint8)
    delta_max = mode_delta.max()
    mode_delta = mode_delta / delta_max if delta_max > 0 else mode_delta
    save_motion(frame, mode_delta, output_path)
    for vis in visualizations:
        vis.save_vis(frame, mode_delta, space_frame=space_frame)


def align(frame, ref):
    best_aligned = np.zeros_like(frame)
    best_delta = 1000
    offset = 0
    for x in range(-offset, offset + 1, 1):
        for y in range(-offset, offset + 1, 1):
            align = np.copy(frame)
            sub_ref = ref[y:] if y >= 0 else ref[:y]
            sub_ref = sub_ref[:, x:] if x >= 0 else sub_ref[:, :x]
            ysc = slice(None, -y) if y > 0 else slice(-y, None)
            xsc = slice(None, -x) if x > 0 else slice(-x, None)
            align[ysc, xsc] = sub_ref
            mode_delta = np.abs(frame.astype(np.int32) - align.astype(np.int32))
            delta = np.mean(np.maximum(np.zeros_like(mode_delta), mode_delta - 120))
            if delta < best_delta:
                best_aligned = align
                best_delta = delta

    return best_aligned

