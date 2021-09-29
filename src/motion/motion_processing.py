import numpy as np
from skimage.morphology import (disk, square)
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat, skeletonize)
from skimage.filters import rank
import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes as draw_bb
from torchvision.utils import save_image
from PIL import Image
import cv2 as cv
from scipy import ndimage


def ring_kernel(lst):
    start = np.array([[lst[0]]])
    for v in lst[1:]:
        start = np.pad(start, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=v)
    return start


class ProcessingVisualization:
    def __init__(self, vis_path, motion_type, every_n=4, max_vis=20, saturation=4):
        self.vis_counter = 0
        self.max_vis = max_vis
        self.every_n = every_n
        self.vis_path = vis_path
        self.saturation = saturation
        self.motion_type = motion_type

    def save_vis(self, frame, motion):
        if self.vis_counter < self.max_vis * self.every_n and not self.vis_counter % self.every_n:
            self.make_visualization(frame, motion)
        self.vis_counter += 1

    def make_visualization(self, frame, motion):
        pass

    def apply_data(self, frame, data):
        vis_flow = (self.saturation * data + 1) / 5
        frame = frame * vis_flow[..., None]
        return ndimage.zoom(frame, (3, 3, 1), order=1)


class Identity(ProcessingVisualization):
    def make_visualization(self, frame, motion):
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/Identity_{self.vis_counter:04}.png',
                   self.apply_data(frame, motion))
        grid_width = 8
        avg_pool = nn.AvgPool2d(grid_width + 2, grid_width, padding=1, count_include_pad=False)
        grid_flow = avg_pool(torch.tensor(motion).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        grid_flow = grid_flow > grid_flow.mean()
        grid_flow = grid_flow.repeat(8, axis=0).repeat(8, axis=1)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/IdentityGrid_{self.vis_counter:04}.png',
                   self.apply_data(frame, grid_flow))


class ClosingMeanThreshold(ProcessingVisualization):
    def __init__(self, vis_path, motion_type, mean_threshold=200, **kwargs):
        super().__init__(vis_path, motion_type, **kwargs)
        self.mean_threshold = mean_threshold

    def make_visualization(self, frame, motion):
        motion = ((motion > 0.1) * 255).astype(np.uint8)
        motion = closing(motion, square(3))
        for i in range(4):
            cv.imwrite(f'{self.vis_path}/{self.motion_type}/ClosingMeanThresh_{self.vis_counter:04}_{i}.png',
                       self.apply_data(frame, motion / 255))
            motion = rank.mean(motion, disk(2))
            motion = ((motion > self.mean_threshold) * 255).astype(np.uint8)


class OneForEachSuperCell(ProcessingVisualization):
    def __init__(self, vis_path, motion_type, grid_width=4, **kwargs):
        super().__init__(vis_path, motion_type, **kwargs)
        self.grid_width = grid_width

    def make_visualization(self, frame, motion):
        avg_pool = nn.AvgPool2d(self.grid_width + 2, self.grid_width, padding=1, count_include_pad=False)
        grid_flow = avg_pool(torch.tensor(motion).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        new_grid_flow = np.zeros_like(grid_flow)
        for x, y in itertools.product(range(0, 32, 4), range(0, 32, 4)):
            x_slice = slice(max(0, x - 2), x + 3)
            y_slice = slice(max(0, y - 2), y + 3)
            neighborhood = grid_flow[x_slice, y_slice]
            local_max = neighborhood.max()
            new_grid_flow[x_slice, y_slice] = neighborhood * (neighborhood >= local_max)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/OneForEachSuperCell_{self.vis_counter:04}.png',
                   self.apply_data(frame, new_grid_flow))


class Skeletonize(ProcessingVisualization):
    def make_visualization(self, frame, motion):
        motion = motion > 0.1
        motion = skeletonize(motion)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/Skeletonize_{self.vis_counter:04}.png',
                   self.apply_data(frame, motion))


class Erosion(ProcessingVisualization):
    def make_visualization(self, frame, motion):
        motion = (motion > 0.1) * 255
        motion = closing(motion, square(3))
        for i in range(4):
            cv.imwrite(f'{self.vis_path}/{self.motion_type}/Erosion_{self.vis_counter:04}_{i}.png',
                       self.apply_data(frame, motion / 255))
            motion = erosion(motion, square(2))


class BoundingBoxes(ProcessingVisualization):
    def make_visualization(self, frame, bb):
        image = np.array(frame)
        bb_coor = bb.drop([4], axis=1).to_numpy()
        objects = torch.from_numpy(bb_coor)
        objects[:, 2:] += objects[:, :2]
        torch_img = torch.from_numpy(image).permute(2, 0, 1)
        bb_img = draw_bb(torch_img, objects, colors=['red'] * len(objects))
        result = Image.fromarray(bb_img.permute(1, 2, 0).numpy())
        result.save(f'{self.vis_path}/BoundingBox/{self.vis_counter:04}.png')


def unique_color(color):
    """
    Computes a unique value for uint8 array, e.g. for identifying the input color to make variance computation easy
    :param color: nd.array<n>
    """
    return sum([255**i * c for i, c in enumerate(color)])

class FlowBoundingBox(ProcessingVisualization):

    def make_visualization(self, frame, motion):
        motion = motion > motion.mean()
        motion = closing(motion, square(3))
        motion = (motion * 255).astype(np.uint8)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/{self.__class__.__name__}_motion_{self.vis_counter:04}.png',
                   motion)
        canny_output = cv.Canny(motion, 100, 200)  # Parameters irrelevant for our binary case
        contours = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        z_pres = np.zeros((16, 16))
        frame_copy = np.copy(frame)
        for c in contours:
            x, y, w, h = cv.boundingRect(c)
            selection = np.apply_along_axis(unique_color, axis=2, arr=frame_copy[y:y + h, x:x + w])
            if w * h >= 30 and np.var(selection) > 1e-4:
                cv.rectangle(frame_copy, (x, y), (x + w, y + h), (36, 255, 12), 1)
                z_pres[(y + h // 2) // 8, (x + w // 2) // 8] = 1.0
        grid_flow = z_pres.repeat(8, axis=0).repeat(8, axis=1)
        self.saturation = 1
        data = self.apply_data(frame_copy, grid_flow) * 2.5
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/{self.__class__.__name__}_{self.vis_counter:04}.png',
                   data)


class IteratedCentroidSelection(ProcessingVisualization):
    def __init__(self, vis_path, motion_type, grid_width=8,
                 kernel_params=np.linspace(0, 0.5, 8), min_object_flow_threshold=0.07, **kwargs):
        super().__init__(vis_path, motion_type, **kwargs)
        self.grid_width = grid_width
        self.kernel_params = kernel_params
        self.min_object_flow_threshold = min_object_flow_threshold

    def make_visualization(self, frame, motion):
        avg_pool = nn.AvgPool2d(self.grid_width, 1, padding=0, count_include_pad=False)
        grid_flow = avg_pool(torch.tensor(motion).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        result_flow = np.zeros_like(motion)

        base_pad = 2 * len(self.kernel_params)  # Simply has to be higher than slice size, otherwise does not matter
        grid_flow = np.pad(grid_flow, pad_width=((base_pad, base_pad), (base_pad, base_pad)), mode='constant',
                           constant_values=0)
        while grid_flow.max() > self.min_object_flow_threshold:
            x, y = np.unravel_index(np.argmax(grid_flow), grid_flow.shape)
            result_flow[x - base_pad, y - base_pad] = 1
            x_slice = slice(x - len(self.kernel_params) + 1, x + len(self.kernel_params))
            y_slice = slice(y - len(self.kernel_params) + 1, y + len(self.kernel_params))
            grid_flow[x_slice, y_slice] = grid_flow[x_slice, y_slice] * ring_kernel(self.kernel_params)
        max_pool = nn.MaxPool2d(8, 8)
        grid_flow = max_pool(torch.tensor(result_flow).unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy().astype(np.uint8)
        grid_flow = grid_flow.repeat(8, axis=0).repeat(8, axis=1)
        data = self.apply_data(frame, grid_flow)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/IteratedCentroidSelection_{self.vis_counter:04}.png',
                   data)
