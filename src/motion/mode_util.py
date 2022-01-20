import numpy as np

def vector_mode(arr):
    """
    :param arr: [3*N]
    :return mode [3]
    """
    colors, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
    return colors[counts.argmax()]
