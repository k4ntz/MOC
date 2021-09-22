import numpy as np
from PIL import Image
import cv2 as cv


def save(trail, output_path, visualizations=None):
    """
    Computes the flow from the last 4 images in the stack
    :param trail: [>=10, H, W, 3] array
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[MotionProcessing] for visualizations
    """
    median = np.median(trail[:-4], axis=0)

    for i, frame in enumerate(trail[:-4]):
        save_frame(frame, median, output_path.format(i), visualizations=visualizations)


def save_frame(frame, median, output_path, visualizations=None):
    """
    Computes the flow from frame2 to frame1
    This inverse oder is done as such that the vectors are high at the spot in frame2 were the moving object is,
    i.e. inverse movement to map more directly to z_pres
    :param frame: [H, W, 3] array
    :param median: [H, W, 3] array describing the median
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[MotionProcessing] for visualizations
    """
    median_delta = np.abs(frame - median)
    median_delta = np.max(median_delta, axis=-1)
    np.save(output_path, median_delta)
    for vis in visualizations:
        vis.save_vis(median_delta)
