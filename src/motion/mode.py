import numpy as np
from PIL import Image
import cv2 as cv
import torch
from .motion_processing import process_motion_to_latents, save_motion
from .mode_util import vector_mode



def save(trail, output_path, visualizations=None, mode=None):
    """
    Computes the flow from the last 4 images in the stack
    :param trail: [>=10, H, W, 3] array
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    :param mode: Optional[nd.array<H, W>] if the mode is known from other sources than the trail
    """
    if mode is None:
        mode = np.apply_along_axis(lambda x: vector_mode(x), axis=0,
                                   arr=np.moveaxis(trail[:-4], 3, 1).reshape(-1, *trail.shape[1:3]))
        mode = np.moveaxis(mode, 0, 2)
    for i, frame in enumerate(trail[-4:]):
        save_frame(frame, mode, output_path.format(i), visualizations=visualizations, i=i)


def save_frame(frame, mode, output_path, visualizations=None, i=0):
    """
    Computes the flow from frame2 to frame1
    This inverse oder is done as such that the vectors are high at the spot in frame2 were the moving object is,
    i.e. inverse movement to map more directly to z_pres, Saved object is [H, W] in [0, 1]
    :param frame: [H, W, 3] array
    :param mode: [H, W, 3] array describing the mode
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    """
    mode_delta = np.abs(frame - mode)
    mode_delta = np.max(mode_delta, axis=-1)
    delta_max = mode_delta.max()
    mode_delta = mode_delta / delta_max if delta_max > 0 else mode_delta
    save_motion(frame, mode_delta, output_path)
    for vis in visualizations:
        vis.save_vis(frame, mode_delta)
