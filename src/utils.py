import torch
from datetime import datetime
import math
import json
import pickle
import os
import os.path as osp
from collections import defaultdict, deque
import numpy as np
from torch import nn
from torch.nn import functional as F
from model.space.utils import spatial_transform
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF
from torchvision.utils import save_image as Simage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 100, 100), (100, 100, 100), (100, 100, 255), (100, 200, 100)] * 10


class Checkpointer:
    def __init__(self, checkpointdir, max_num):
        self.max_num = max_num
        self.checkpointdir = checkpointdir
        if not osp.exists(checkpointdir):
            os.makedirs(checkpointdir)
        self.listfile = osp.join(checkpointdir, 'model_list.pkl')

        if not osp.exists(self.listfile):
            with open(self.listfile, 'wb') as f:
                model_list = []
                pickle.dump(model_list, f)

    def save(self, path: str, model, optimizer_fg, optimizer_bg, epoch, global_step):
        assert path.endswith('.pth')
        os.makedirs(osp.dirname(path), exist_ok=True)

        if isinstance(model, nn.DataParallel):
            model = model.module
        checkpoint = {
            'model': model.state_dict(),
            'optimizer_fg': optimizer_fg.state_dict() if optimizer_fg else None,
            'optimizer_bg': optimizer_bg.state_dict() if optimizer_bg else None,
            'epoch': epoch,
            'global_step': global_step
        }
        with open(path, 'wb') as f:
            torch.save(checkpoint, f)
            print(f'Checkpoint has been saved to "{path}".')

    def save_last(self, model, optimizer_fg, optimizer_bg, epoch, global_step):
        path = osp.join(self.checkpointdir, 'model_{:09}.pth'.format(global_step + 1))

        with open(self.listfile, 'rb+') as f:
            model_list = pickle.load(f)
            if len(model_list) >= self.max_num:
                if osp.exists(model_list[0]):
                    os.remove(model_list[0])
                del model_list[0]
            model_list.append(path)
        with open(self.listfile, 'rb+') as f:
            pickle.dump(model_list, f)
        self.save(path, model, optimizer_fg, optimizer_bg, epoch, global_step)

    def load(self, path, model, optimizer_fg, optimizer_bg, use_cpu=False):
        """
        Return starting epoch and global step
        """

        assert osp.exists(path), f'Checkpoint {path} does not exist.'
        print('Loading checkpoint from {}...'.format(path))
        if not use_cpu:
            if torch.cuda.device_count() == 1:
                checkpoint = torch.load(path, map_location="cuda:0")
            else:
                checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint.pop('model'))
        if optimizer_fg:
            optimizer_fg.load_state_dict(checkpoint.pop('optimizer_fg'))
        if optimizer_bg:
            optimizer_bg.load_state_dict(checkpoint.pop('optimizer_bg'))
        print('Checkpoint loaded.')
        return checkpoint


    def load_last(self, path, model, optimizer_fg, optimizer_bg, use_cpu=False):
        """
        If path is '', we load the last checkpoint
        """

        if path == '':
            with open(self.listfile, 'rb') as f:
                model_list = pickle.load(f)
                if len(model_list) == 0:
                    print('No checkpoint found. Starting from scratch')
                    return None
                else:
                    path = model_list[-1]

        return self.load(path, model, optimizer_fg, optimizer_bg, use_cpu)

    def save_best(self, metric_name, value, checkpoint,  min_is_better):
        metric_file = os.path.join(self.checkpointdir, f'best_{metric_name}.json')
        checkpoint_file = os.path.join(self.checkpointdir, f'best_{metric_name}.pth')

        now = datetime.now()
        log = {
            'name': metric_name,
            'value': float(value),
            'date': now.strftime("%Y-%m-%d %H:%M:%S"),
            'global_step': checkpoint[-1]
        }

        if not os.path.exists(metric_file):
            dump = True
        else:
            with open(metric_file, 'r') as f:
                previous_best = json.load(f)
            if not math.isfinite(log['value']):
                dump = True
            elif (min_is_better and log['value'] < previous_best['value']) or (
                    not min_is_better and log['value'] > previous_best['value']):
                dump = True
            else:
                dump = False
        if dump:
            with open(metric_file, 'w') as f:
                json.dump(log, f)
            self.save(checkpoint_file, *checkpoint)

    def load_best(self, metric_name, model, fg_optimizer, bg_optimizer, use_cpu=False):
        metric_file = os.path.join(self.checkpointdir, f'best_{metric_name}.json')
        checkpoint_file = os.path.join(self.checkpointdir, f'best_{metric_name}.pth')

        assert osp.exists(metric_file), 'Metric file does not exist'
        assert osp.exists(checkpoint_file), 'checkpoint file does not exist'

        return self.load(checkpoint_file, model, fg_optimizer, bg_optimizer, use_cpu)


class SmoothedValue:
    """
    Record the last several values, and return summaries
    """

    def __init__(self, maxsize=20):
        self.values = deque(maxlen=maxsize)
        self.count = 0
        self.sum = 0.0

    def update(self, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.values.append(value)
        self.count += 1
        self.sum += value

    @property
    def median(self):
        return np.median(np.array(self.values))

    @property
    def avg(self):
        return np.mean(self.values)

    @property
    def global_avg(self):
        return self.sum / self.count


class MetricLogger:
    def __init__(self):
        self.values = defaultdict(SmoothedValue)

    def update(self, **kargs):
        for key, value in kargs.items():
            self.values[key].update(value)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, item):
        self.values[key].update(item)


def open_image(path):
    image = Image.open(path)
    image = TF.to_tensor(image).unsqueeze_(0)
    return image


def show_image(image):
    plt.imshow(np.moveaxis(image.detach().cpu().numpy(), (0, 1, 2), (2, 0, 1)))
    plt.show()


def save_image(image, path, number=None):
    path = path.replace("data", "extracted_images")
    folder = "/".join(path.split("/")[:-1])
    os.makedirs(folder, exist_ok=True)
    image_name = path.split("/")[-1]
    name, ext = image_name.split(".")
    if number is not None:
        new_name = name + f"_{number}.png" # saved as png to avoid loosing
    try:
        Simage(image/255, f"{folder}/{new_name}")
    except ValueError:
        print(f"Couldn't save {new_name}, continuing...")


def corners_to_wh(bbox):
    """
    To convert 2 corners to 1 corner + width and height
    xmin, ymin, xmax, ymax -> top, left, height, width
    """
    xmin, ymin, xmax, ymax = bbox.to(torch.int)
    xmin += 1; ymin += 1
    height, width = (ymax - ymin), (xmax - xmin)
    return ymin, xmin, height, width


def image_pca(image, bbox, z_what, row_color=True):
    """
    Draw the image next to a plot of PCA(zwat), reduced to dim=2
    if row_color, then one color is used per row
    """
    pca = PCA(n_components=2)
    z_what_emb = pca.fit_transform(z_what.detach().cpu().numpy())
    fig, (ax1, ax2) = plt.subplots(1, 2)
    selected_colors = np.array(colors[:len(z_what)])/255
    if row_color:
        for b, box in enumerate(bbox):
            row_group = int(box[1] / 10)
            image = draw_bounding_boxes(image.to("cpu"), torch.tensor(box).unsqueeze_(0), colors=[colors[row_group]])
            ax2.scatter(z_what_emb[:, 0][b], z_what_emb[:, 1][b], c=selected_colors[row_group])
        ax1.imshow(np.moveaxis(image.detach().cpu().numpy(), (0, 1, 2), (2, 0, 1)))
    else:
        image = draw_bounding_boxes(image.to("cpu"), torch.tensor(bbox), colors=colors)
        ax1.imshow(np.moveaxis(image.detach().cpu().numpy(), (0, 1, 2), (2, 0, 1)))
        ax2.scatter(z_what_emb[:, 0], z_what_emb[:, 1], c=selected_colors)
    plt.show()
