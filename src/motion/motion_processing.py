import numpy as np

def ring_kernel(lst):
    start = np.array([[lst[0]]])
    for v in lst[1:]:
        start = np.pad(start, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=v)
    return start

class MotionProcessing:
    def save_vis(self):
        if self.vis_counter < 100:
            self.vis_counter += 1
            self.make_visualization(data)