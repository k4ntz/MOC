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

color_hist = None
use_color_hist = False


def set_color_hist(img):
    global color_hist
    color_codes = np.apply_along_axis(unique_color, axis=2, arr=img)
    unique, counts = np.unique(color_codes.ravel(), return_counts=True)
    color_hist = dict(zip(unique, counts))
    print(color_hist)
    for c in color_hist:
        print(c, to_inverse_count(c))


def set_special_color_weight(color, count):
    global color_hist
    color_hist[color] = count
    for c in color_hist:
        print(c, to_inverse_count(c))


def to_inverse_count(color):
    return 1 / color_hist.get(color, 1) - 1e-4


def exciting_color_score(img, x, y, w, h):
    if w < 1 or h < 1:
        return 0
    selection = img[y:y + h, x:x + w]
    color_codes = np.apply_along_axis(unique_color, axis=2, arr=selection)
    inverse_counts = np.vectorize(to_inverse_count)(color_codes)
    return np.sum(inverse_counts)


def select(img, x, y, w, h):
    base_score = exciting_color_score(img, x, y, w, h)
    w_ = w // 5 + 1
    h_ = h // 5 + 1
    while w > 8:
        left_cut_score = exciting_color_score(img, x + w_, y, w - w_, h)
        if left_cut_score > base_score:
            x = x + w_
            w -= w_
            base_score = left_cut_score
        else:
            break
    while w > 8:
        right_cut_score = exciting_color_score(img, x, y, w - w_, h)
        if right_cut_score > base_score:
            w -= w_
            base_score = right_cut_score
        else:
            break
    while h > 9:
        left_cut_score = exciting_color_score(img, x, y + h_, w, h - h_)
        if left_cut_score > base_score:
            y = y + h_
            h -= h_
            base_score = left_cut_score
        else:
            break
    while h > 9:
        right_cut_score = exciting_color_score(img, x, y, w, h - h_)
        if right_cut_score > base_score:
            h -= h_
            base_score = right_cut_score
        else:
            break
    return (-1, -1, -1, -1) if base_score < 1e-2 else (x, y, w, h)


def ring_kernel(lst):
    start = np.array([[lst[0]]])
    for v in lst[1:]:
        start = np.pad(start, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=v)
    return start


def save_motion(img, motion, output_path):
    _, _, z_pres, z_where = process_motion_to_latents(img, motion)
    torch.save(resize(motion, size=128),
               output_path.replace(".pt", ".pt"))
    torch.save(resize(motion, size=84),
               output_path.replace(".pt", "_84.pt"))
    torch.save(resize(motion, size=64),
               output_path.replace(".pt", "_64.pt"))
    torch.save(z_pres.float(), output_path.replace(".pt", "_z_pres.pt"))
    torch.save(z_where.float(), output_path.replace(".pt", "_z_where.pt"))


def resize(motion, size):
    return torch.nn.functional.interpolate(torch.from_numpy(motion).float().unsqueeze(0).unsqueeze(0),
                                           size=size).squeeze()


class RectBB():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.is_used = False

    def touches(self, other):
        return (self.x - 1 <= other.x <= self.x_max + 1 or other.x - 1 <= self.x <= other.x_max + 1) \
               and \
               (self.y - 1 <= other.y <= self.y_max + 1 or other.y - 1 <= self.y <= other.y_max + 1)

    def merge(self, other):
        return RectBB(min(self.x, other.x), min(self.y, other.y),
                      max(self.x_max, other.x_max) - min(self.x, other.x),
                      max(self.y_max, other.y_max) - min(self.y, other.y))

    @property
    def x_max(self):
        return self.x + self.w

    @property
    def y_max(self):
        return self.y + self.h

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __repr__(self):
        return f"RectBB{self.x, self.y, self.w, self.h}"


def merge_bbs(bbs):
    bbs_len = len(bbs) + 1
    next_bbs = []
    while len(bbs) != bbs_len:
        next_bbs = []
        for bb in bbs:
            if bb.is_used:
                continue
            cur_bb = bb
            for other_bb in bbs:
                if bb is not other_bb and cur_bb.touches(other_bb):
                    cur_bb = cur_bb.merge(other_bb)
                    other_bb.is_used = True
            next_bbs.append(cur_bb)
        bbs_len = len(bbs)
        bbs = next_bbs[:]
    return next_bbs


def process_motion_to_latents(img, motion, G=16):
    # Idea: Move z_pres instead of overwriting
    # y_delta, x_delta = z_where_y - motion_z_where[y_idx, x_idx][3], z_where_x - motion_z_where[y_idx, x_idx][2]
    # if np.abs(y_delta) > np.abs(x_delta):
    #     y_idx += 1 if y_delta > 0 else -1
    # else:
    #     x_idx += 1 if x_delta > 0 else -1
    # y_idx = max(0, min(y_idx, G - 1))
    # x_idx = max(0, min(x_idx, G - 1))
    #
    """
    Converts the motion as magnitude of flow / delta to the median or mode of the last few images into
    z_pres and z_where, i.e. giving hints where the objects are so SPACE can one the one hand imitate it and
    on the other concentrate on finding sensible z_what representations
    :param motion: (H_img, W_img)
    :param img: (3, H_img, W_img)
    :param G: grid_size, and int
    :return z_pres: (G * G, 1) in (-1, 1) (tanh)
    :return z_where: (G * G, 4), where all grid cells without z_pres == 1 only contain zero
    """
    H, W = motion.shape
    vis_motion = motion > motion.mean()
    vis_motion = (closing(vis_motion, square(3)) * 255).astype(np.uint8)
    # TODO: Investigate Canny function
    # canny_output = cv.Canny(vis_motion, 100, 200)  # Parameters irrelevant for our binary case
    contours = cv.findContours(vis_motion, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    bbs = [RectBB(x, y, w, h) for x, y, w, h in [cv.boundingRect(c) for c in contours]]
    # bbs = merge_bbs(bbs)
    motion_z_pres = torch.zeros((G, G, 1))
    motion_z_where = torch.zeros((G, G, 4))
    to_G_y = H / G
    to_G_x = W / G
    for x, y, w, h in bbs:
        # One could collect background colors and prune non-objects only consisting of those colors.
        # But background might be of same color in other area
        if color_hist:
            x, y, w, h = select(img, x, y, w, h)
            if x < 0:
                continue
        w = min(W - x, w)
        h = min(H - y, h)
        z_where_x = ((x + w / 2) / W) * 2 - 1
        z_where_y = ((y + h / 2) / H) * 2 - 1
        y_idx = int((y + h // 2) // to_G_y)
        x_idx = int((x + w // 2) // to_G_x)
        if motion_z_pres[y_idx, x_idx] == 0.0 or w / W * h / H > motion_z_where[y_idx, x_idx][:2].prod():
            motion_z_pres[y_idx, x_idx] = 1.0
            motion_z_where[y_idx, x_idx] = torch.tensor(
                [w / W, h / H, z_where_x, z_where_y])
    return vis_motion, contours, motion_z_pres.reshape(G * G, -1), motion_z_where.reshape(G * G, -1)


def unique_color(color):
    """
    Computes a unique value for uint8 array, e.g. for identifying the input color to make variance computation easy
    :param color: nd.array<n>
    """
    return sum([256 ** i * c for i, c in enumerate(color)])


class ProcessingVisualization:
    def __init__(self, vis_path, motion_type, every_n=4, max_vis=20, saturation=6, G=16):
        self.vis_counter = 0
        self.max_vis = max_vis
        self.every_n = every_n
        self.vis_path = vis_path
        self.saturation = saturation
        self.motion_type = motion_type
        self.G = G

    def save_vis(self, frame, motion, space_frame=None):
        if self.vis_counter < self.max_vis * self.every_n and not self.vis_counter % self.every_n:
            self.make_visualization(frame, motion, space_frame=space_frame)
        self.vis_counter += 1

    def make_visualization(self, frame, motion, space_frame=None):
        pass

    def apply_data(self, frame, data):
        vis_flow = (self.saturation * data + 1) / 7
        sat_frame = frame + 20
        sat_frame = sat_frame * vis_flow[..., None]
        return ndimage.zoom(sat_frame, (3, 3, 1), order=1)


class Identity(ProcessingVisualization):
    def make_visualization(self, frame, motion, space_frame=None):
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/MotionIdentity_{self.vis_counter:04}.png',
                   motion * 255)
        vis_motion = motion > 0
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/Identity_{self.vis_counter:04}.png',
                   self.apply_data(frame, vis_motion))
        to_G_x = 160 // self.G
        to_G_y = 210 // self.G

        avg_pool = nn.AvgPool2d((to_G_y + 2, to_G_x + 2), (to_G_y, to_G_x), padding=1, count_include_pad=False)
        grid_flow = avg_pool(torch.tensor(motion).float().unsqueeze(0).unsqueeze(0))
        grid_flow = torch.squeeze(grid_flow).numpy()
        grid_flow = grid_flow > grid_flow.mean()
        grid_flow = grid_flow.repeat(to_G_y, axis=0).repeat(to_G_x, axis=1)
        vis_frame = frame[:grid_flow.shape[0], :grid_flow.shape[1]]
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/IdentityGrid_{self.vis_counter:04}.png',
                   self.apply_data(vis_frame, grid_flow))


class ClosingMeanThreshold(ProcessingVisualization):
    def __init__(self, vis_path, motion_type, mean_threshold=200, **kwargs):
        super().__init__(vis_path, motion_type, **kwargs)
        self.mean_threshold = mean_threshold

    def make_visualization(self, frame, motion, space_frame=None):
        vis_motion = ((motion > 0.1) * 255).astype(np.uint8)
        vis_motion = closing(vis_motion, square(3))
        for i in range(4):
            cv.imwrite(f'{self.vis_path}/{self.motion_type}/ClosingMeanThresh_{self.vis_counter:04}_{i}.png',
                       self.apply_data(frame, vis_motion / 255))
            vis_motion = rank.mean(vis_motion, disk(2))
            vis_motion = ((vis_motion > self.mean_threshold) * 255).astype(np.uint8)


class OneForEachSuperCell(ProcessingVisualization):
    def __init__(self, vis_path, motion_type, grid_width=4, **kwargs):
        super().__init__(vis_path, motion_type, **kwargs)
        self.grid_width = grid_width

    def make_visualization(self, frame, motion, space_frame=None):
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
    def make_visualization(self, frame, motion, space_frame=None):
        vis_motion = motion > 0.1
        vis_motion = skeletonize(vis_motion)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/Skeletonize_{self.vis_counter:04}.png',
                   self.apply_data(frame, vis_motion))


class Erosion(ProcessingVisualization):
    def make_visualization(self, frame, motion, space_frame=None):
        vis_motion = (motion > 0.1) * 255
        vis_motion = closing(vis_motion, square(3))
        for i in range(4):
            cv.imwrite(f'{self.vis_path}/{self.motion_type}/Erosion_{self.vis_counter:04}_{i}.png',
                       self.apply_data(frame, vis_motion / 255))
            vis_motion = erosion(vis_motion, square(2))


class BoundingBoxes(ProcessingVisualization):
    def make_visualization(self, frame, motion, space_frame=None):
        image = np.array(frame)
        zoom = 3
        image = ndimage.zoom(image, (zoom, zoom, 1), order=0)
        labels = motion[5]
        bb_coor = motion.drop([4, 5], axis=1).to_numpy()
        objects = torch.from_numpy(bb_coor)
        objects[:, [2, 3]] *= 128 * zoom
        objects[:, [0, 1]] *= 128 * zoom
        torch_img = torch.from_numpy(image).permute(2, 0, 1)
        objects = objects[:, [2, 0, 3, 1]]
        bb_img = draw_bb(torch_img, objects, colors=['red'] * len(objects), labels=labels)
        result = Image.fromarray(ndimage.zoom(bb_img.permute(1, 2, 0).numpy(), (3, 3, 1), order=1))
        result.save(f'{self.vis_path}/{self.vis_counter:04}.png')


class ZWhereZPres(ProcessingVisualization):
    def make_visualization(self, frame, motion, space_frame=None):
        vis_motion, contours, z_pres, z_where = process_motion_to_latents(frame, motion)
        z_where = z_where[z_pres.squeeze() > 0.5]
        image = np.array(frame)
        H, W, C = frame.shape
        z_where_space = torch.clone(z_where)
        z_where[:, 2:] += 1
        z_where[:, 2:] /= 2
        z_where[:, 0] *= W
        z_where[:, 2] *= W
        z_where[:, 1] *= H
        z_where[:, 3] *= H

        z_where_space[:, 2:] += 1
        z_where_space[:, 2:] /= 2
        z_where_space *= 128

        objects = torch.zeros_like(z_where)
        objects_space = torch.zeros_like(z_where)
        cc_bb = torch.tensor([[x, y, x + w, y + h] for x, y, w, h in [cv.boundingRect(c) for c in contours]])
        objects[:, :2] = z_where[:, 2:] - z_where[:, :2] / 2
        objects[:, 2:] = z_where[:, 2:] + z_where[:, :2] / 2

        objects_space[:, :2] = z_where_space[:, 2:] - z_where_space[:, :2] / 2
        objects_space[:, 2:] = z_where_space[:, 2:] + z_where_space[:, :2] / 2

        torch_img = torch.from_numpy(image).permute(2, 0, 1)
        bb_img = draw_bb(torch_img, objects, colors=['red'] * len(objects))
        result = Image.fromarray(bb_img.permute(1, 2, 0).numpy())
        result.save(f'{self.vis_path}/{self.motion_type}/z_where_{self.vis_counter:04}.png')

        torch_img = torch.from_numpy(np.array(space_frame)).permute(2, 0, 1)
        bb_img = draw_bb(torch_img, objects_space, colors=['red'] * len(objects))
        result = Image.fromarray(bb_img.permute(1, 2, 0).numpy())
        result.save(f'{self.vis_path}/{self.motion_type}/space_z_where_{self.vis_counter:04}.png')

        cc_bb_img = draw_bb(torch_img, cc_bb, colors=['orange'] * len(contours))
        result_cc = Image.fromarray(cc_bb_img.permute(1, 2, 0).numpy())
        result_cc.save(f'{self.vis_path}/{self.motion_type}/cc_bb_{self.vis_counter:04}.png')
        to_G_x = W // self.G
        to_G_y = H // self.G
        grid_flow = z_pres.numpy().reshape((self.G, self.G)).repeat(to_G_x, axis=1).repeat(to_G_y, axis=0)
        vis_frame = frame[:grid_flow.shape[0], :grid_flow.shape[1]]
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/vis_motion_{self.vis_counter:04}.png',
                   vis_motion)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/vis_motion_{self.vis_counter:04}.png',
                   vis_motion)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/z_pres_{self.vis_counter:04}.png',
                   self.apply_data(vis_frame, grid_flow))


def unique_color(color):
    """
    Computes a unique value for uint8 array, e.g. for identifying the input color to make variance computation easy
    :param color: nd.array<n>
    """
    return sum([255 ** i * c for i, c in enumerate(color)])


# Deprecated. Only for 128x128
class FlowBoundingBox(ProcessingVisualization):

    def make_visualization(self, frame, motion, space_frame=None):
        vis_motion = motion > motion.mean()
        vis_motion = closing(vis_motion, square(3))
        vis_motion = (vis_motion * 255).astype(np.uint8)
        cv.imwrite(f'{self.vis_path}/{self.motion_type}/{self.__class__.__name__}_motion_{self.vis_counter:04}.png',
                   vis_motion)
        canny_output = cv.Canny(vis_motion, 100, 200)  # Parameters irrelevant for our binary case
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


# Deprecated. Only for 128x128
class IteratedCentroidSelection(ProcessingVisualization):
    def __init__(self, vis_path, motion_type, grid_width=8,
                 kernel_params=np.linspace(0, 0.5, 8), min_object_flow_threshold=0.07, **kwargs):
        super().__init__(vis_path, motion_type, **kwargs)
        self.grid_width = grid_width
        self.kernel_params = kernel_params
        self.min_object_flow_threshold = min_object_flow_threshold

    def make_visualization(self, frame, motion, space_frame=None):
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
