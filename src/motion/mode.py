import numpy as np
from PIL import Image
import cv2 as cv


def save(trail, output_path, visualizations=None, mode=None):
    """
    Computes the flow from the last 4 images in the stack
    :param trail: [>=10, H, W, 3] array
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    :param mode: Optional[nd.array<128, 128>] if the mode is known from other sources than the trail
    """
    if mode is None:
        mode = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=trail[:-4])

    for i, frame in enumerate(trail[-4:]):
        save_frame(frame, mode, output_path.format(i), visualizations=visualizations)


def save_frame(frame, mode, output_path, visualizations=None):
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
    np.save(output_path, mode_delta)
    for vis in visualizations:
        vis.save_vis(frame, mode_delta)