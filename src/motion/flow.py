import cv2 as cv
import numpy as np


def last_four_pairs(trail):
    return [(i, *content) for i, content in enumerate(zip(trail[-5:-1], trail[-4:]))]


def save(trail, output_path, visualizations=None):
    """
    Computes the flow from the last 4 images in the stack
    :param trail: [>=10, H, W, 3] array
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    """
    for i, frame1, frame2 in last_four_pairs(trail):
        save_frame(frame1, frame2, output_path.format(i), visualizations=visualizations)


def save_frame(frame1, frame2, output_path, visualizations=None):
    """
    Computes the flow from frame2 to frame1
    This inverse oder is done as such that the vectors are high at the spot in frame2 were the moving object is,
    i.e. inverse movement to map more directly to z_pres
    :param frame1: [H, W, 3] array
    :param frame2: [H, W, 3] array
    :param output_path: a path to the numpy data file to write
    :param visualizations: List[ProcessingVisualization] for visualizations
    """
    if visualizations is None:
        visualizations = []
    frame1_bw = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    frame2_bw = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(frame2_bw, frame1_bw, None, 0.6, 5, 15, 30, 5, 1.5, 0)
    flow = (flow * flow).sum(axis=2)
    flow = flow * 255 / flow.max()
    np.save(output_path, flow)
    for vis in visualizations:
        vis.save_vis(frame2, flow)
