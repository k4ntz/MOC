import numpy as np
from PIL import Image
import cv2 as cv
import torch
from .motion_processing import process_motion_to_latents, save_motion


def save(trail, output_path, visualizations=None, median=None):
    """
    Computes the flow from the last 4 images in the stack
    :param trail: [>=10, H, W, 3] array
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    :param median: Optional[nd.array<128, 128>] if the median is known from other sources than the trail
    """
    if median is None:
        median = np.median(trail[:-4], axis=0)

    for i, frame in enumerate(trail[-4:]):
        save_frame(frame, median, output_path.format(i), visualizations=visualizations)


def save_frame(frame, median, output_path, visualizations=None):
    """
    Computes the flow from frame2 to frame1
    This inverse order is done as such that the vectors are high at the spot in frame2 were the moving object is,
    i.e. inverse movement to map more directly to z_pres, Saved object is [H, W] in [0, 1]
    :param frame: [H, W, 3] array
    :param median: [H, W, 3] array describing the median
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    """
    median_delta = np.abs(frame - median)
    median_delta = np.max(median_delta, axis=-1)
    delta_max = median_delta.max()
    median_delta = median_delta / delta_max if delta_max > 0 else median_delta
    save_motion(frame, median_delta, output_path)
    for vis in visualizations:
        vis.save_vis(frame, median_delta)
