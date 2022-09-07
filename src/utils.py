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
from model.space.utils import spatial_transform, inverse_spatial_transform
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image as Simage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes as draw_bb
from ast import literal_eval


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 255), (255, 100, 100), (100, 100, 100), (100, 100, 255), (100, 200, 100)] * 100

base_objects_colors = {"sue": (180, 122, 48), "inky": (84, 184, 153),
                       "pinky": (198, 89, 179), "blinky": (200, 72, 72),
                       "pacman": (210, 164, 74), "fruit": (184, 50, 50),
                       "save_fruit": (144, 10, 10),
                       "white_ghost": (214, 214, 214),
                       "blue_ghost": (66, 114, 194), "score0": (200, 50, 200),
                       "life": (70, 210, 70), "life2": (20, 150, 20)
                       }  #

RESTORE_COLORS = True

turned = False


def draw_bounding_boxes(image, boxes_batch, labels=None):
    if len(image.shape) == 4:
        image = image.squeeze(0)
    if "float" in str(image.dtype):
        image = (image * 255).to(torch.uint8)
    _, imW, imH = image.shape
    if not torch.is_tensor(image):
        image = torch.tensor(image)
    boxes_batch = np.array(boxes_batch)
    if boxes_batch.min() < -0.5:
        boxes_batch = (boxes_batch + 1)/2
    # bb = (boxes_batch[0][:, :4] * (imW, imW, imH, imH)).round()
    bb = (np.array(boxes_batch)[:, :4] * (imW, imW, imH, imH)).round()
    bb[:, [0, 1, 2, 3]] = bb[:, [2, 0, 3, 1]]  # swapping xmin <-> ymax ... etc
    if imW == 128 and imH == 128 and RESTORE_COLORS:
        image = torch.tensor(np.flip(image.to("cpu").numpy(), axis=0).copy())
    else:
        image = image.to("cpu")
    colors = ["red"] * 200
    if labels is not None:
        labels = label_names(labels)
        # colors = [base_objects_colors[lab] for lab in labels]
    colors = [(0,255,0)] * 50
    image = draw_bb(image, torch.tensor(bb), colors=colors, width=2)
    return image


class Checkpointer:
    def __init__(self, checkpointdir, max_num, load_time_consistency=False, add_flow=False):
        self.max_num = max_num
        self.load_time_consistency = load_time_consistency
        self.add_flow = add_flow
        self.checkpointdir = checkpointdir
        if not osp.exists(checkpointdir):
            os.makedirs(checkpointdir)
        self.listfile = osp.join(checkpointdir, 'model_list.pkl')
        print(f'Listfile: {self.listfile}')
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

    def load(self, path, model, optimizer_fg=None, optimizer_bg=None, device=None):
        """
        Return starting epoch and global step
        """

        assert osp.exists(path), f'Checkpoint {path} does not exist.'
        checkpoint = torch.load(path, map_location=device)
        checkpoint_model = checkpoint.pop('model')
        # if self.add_flow:
        #     for key in checkpoint_model:
        #         if model.state_dict()[key].shape != checkpoint_model[key].shape:
        #             print(key)
        #             B, C, H, W = checkpoint_model[key].size()
        #             flow_weights = torch.randn((B, 1, H, W)).to(checkpoint_model[key].device)
        #             checkpoint_model[key] = torch.cat((checkpoint_model[key], flow_weights), dim=1)
        try:
            model.load_state_dict(checkpoint_model)
        except RuntimeError as rterr:
            model.space.load_state_dict(checkpoint_model)
        if optimizer_fg and not self.add_flow:
            optimizer_fg.load_state_dict(checkpoint.pop('optimizer_fg'))
        if optimizer_bg and not self.add_flow:
            optimizer_bg.load_state_dict(checkpoint.pop('optimizer_bg'))
        print('Checkpoint loaded.')
        return checkpoint

    def load_last(self, path, model, optimizer_fg=None, optimizer_bg=None, device=None):
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
        if not path.startswith('../'):
            path = '../' + path
        return self.load(path, model, optimizer_fg, optimizer_bg, device)

    def save_best(self, metric_name, value, checkpoint, min_is_better):
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

    def load_best(self, metric_name, model, fg_optimizer, bg_optimizer, device):
        metric_file = os.path.join(self.checkpointdir, f'best_{metric_name}.json')
        checkpoint_file = os.path.join(self.checkpointdir, f'best_{metric_name}.pth')

        assert osp.exists(metric_file), 'Metric file does not exist'
        assert osp.exists(checkpoint_file), 'checkpoint file does not exist'

        return self.load(checkpoint_file, model, fg_optimizer, bg_optimizer, device)


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
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    plt.imshow(np.moveaxis(image, (0, 1, 2), (2, 0, 1)))
    plt.show()


def save_image(image, path, number=None):
    path = path.replace("data", "extracted_images")
    folder = "/".join(path.split("/")[:-1])
    os.makedirs(folder, exist_ok=True)
    image_name = path.split("/")[-1]
    name, ext = image_name.split(".")
    if number is not None:
        new_name = name + f"_{number}.png"  # saved as png to avoid loosing
    try:
        Simage(image / 255, f"{folder}/{new_name}")
    except ValueError:
        print(f"Couldn't save {new_name}, continuing...")


def corners_to_wh(bbox):
    """
    To convert 2 corners to 1 corner + width and height
    xmin, ymin, xmax, ymax -> top, left, height, width
    """
    xmin, ymin, xmax, ymax = bbox.to(torch.int)
    xmin += 1
    ymin += 1
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
    selected_colors = np.array(colors[:len(z_what)]) / 255
    if row_color:
        for b, box in enumerate(bbox):
            row_group = int(box[1] / 10)
            image = draw_bb(image.to("cpu"), torch.tensor(box).unsqueeze_(0), colors=[colors[row_group]])
            ax2.scatter(z_what_emb[:, 0][b], z_what_emb[:, 1][b], c=selected_colors[row_group])
        ax1.imshow(np.moveaxis(image.detach().cpu().numpy(), (0, 1, 2), (2, 0, 1)))
    else:
        image = draw_bb(image.to("cpu"), torch.tensor(bbox), colors=colors)
        ax1.imshow(np.moveaxis(image.detach().cpu().numpy(), (0, 1, 2), (2, 0, 1)))
        ax2.scatter(z_what_emb[:, 0], z_what_emb[:, 1], c=selected_colors)
    plt.show()


mspacman_label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky',
                        "blue_ghost", "white_ghost", "fruit", "save_fruit",
                        "life", "life2", "score"]

spaceinvaders_label_list = ["player", "invader", "missile", "shield",
                            "planet", "green_score", "yellow_score"]


def place_labels(labels, boxes_batch, image):
    # labels = label_names(labels)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if len(image) == 3:
        turned = True
        image = np.moveaxis(image, 0, 2)
    bb = (boxes_batch[0][:, :4] * (210, 210, 160, 160)).round().astype(int)
    for bbx, label in zip(bb, labels):
        image = draw_name(image, label, bbx)
    if turned:
        image = np.moveaxis(image, 2, 0)
    return image


def get_labels(serie, boxes_batch, game, return_form="int"):
    """
    Return starting the label list
    return_form in `str`, `int`
    """
    def extract_data(pd_item):
        return literal_eval(pd_item.item())

    def _get_label_spaceinvaders(serie, boxes_batch, return_form):
        bb = (boxes_batch[0][:,:4] * (210, 210, 160, 160)).round().astype(int)
        # Fill labels list
        potential_obj = ["shields", "invaders", "missiles", "player",
                         "yellow_scores", "green_scores", "planet"]
        pieces = {}
        for pobj in potential_obj:
            pieces[pobj] = extract_data(serie[pobj])
        labels = []
        for bbx in bb:
            min_dist = np.inf
            label = ""
            cor_pos = None
            for name, pos in pieces.items():
                # print(name)
                if len(pos) > 0 and isinstance(pos[0], int):  # single object
                    cur_dist = abs(bbx[0] - pos[0]) + abs(bbx[3] - pos[1])
                    if cur_dist < min_dist:
                        label = name
                        min_dist = cur_dist
                        cor_pos = pos
                else: # multiple objects
                    for spos in pos:
                        cur_dist = abs(bbx[0] - spos[0]) + abs(bbx[2] - spos[1])
                        if cur_dist < min_dist:
                            label = name[:-1]
                            min_dist = cur_dist
                            cor_pos = spos
            labels.append(label)
        if return_form == "str":
            return labels
        elif return_form == "int":
            return torch.LongTensor([spaceinvaders_label_list.index(lab) for lab in labels])

    if "mspacman" in game.lower():
        return _get_label_mspacman(serie, boxes_batch, return_form)
    elif "spaceinvaders" in game.lower():
        return _get_label_spaceinvaders(serie, boxes_batch, return_form)


def _get_label_mspacman(serie, boxes_batch, return_form="labels_n"):
    enemy_list = ['sue', 'inky', 'pinky', 'blinky']
    pieces = {
        "save_fruit": (170, 136),
        "life": (169, 21),
        "life2": (169, 38),
        "score0": (183, 81)
    }
    bb = (boxes_batch[:, :4] * (210, 210, 160, 160)).round().astype(int)
    pieces["pacman"] = (serie['player_y'].item(), serie['player_x'].item())
    pieces["fruit"] = (serie['fruit_y'].item(), serie['fruit_x'].item())
    for en in enemy_list:
        if serie[f'{en}_visible'].item():
            pieces[en] = (serie[f'{en}_y'].item(), serie[f'{en}_x'].item())
        if not serie[f'{en}_blue'].item():
            all_blue = False
    # Fill labels list
    labels = []
    for bbx in bb:
        min_dist = np.inf
        label = ""
        cor_pos = None
        for name, pos in pieces.items():
            cur_dist = abs(bbx[0] - pos[0]) + abs(bbx[3] - pos[1])
            if cur_dist < min_dist:
                label = name
                min_dist = cur_dist
                cor_pos = pos
        if label in enemy_list:
            if serie[f'{label}_blue'].item():
                label = "blue_ghost"
            elif serie[f'{label}_white'].item():
                label = "white_ghost"
        labels.append(label)
    if return_form == "str":
        return labels
    elif return_form == "int":
        return torch.LongTensor([mspacman_label_list.index(lab) for lab in labels])
    else:
        print("\n\nUNCORECT return_form\n\n")


def draw_name(image, name, bbox):
    from PIL import Image, ImageFont, ImageDraw
    if image.max() <= 1:
        image = np.uint8(image * 255)
    img = Image.fromarray(image, 'RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("../fonts/arial.ttf", 10)
    draw.text((bbox[3], bbox[0]), name, (255, 255, 255), font=font)
    return np.array(img)


def mark_point(image_array, x, y, color=(255, 0, 0), size=2, show=True):
    """
    marks a point on the image at the (x,y) position and displays it
    """
    if len(image_array) == 3:
        image_array = np.moveaxis(image_array, 0, 2)
    ran = list(range(-size, size + 1))
    for i in ran:
        for j in ran:
            try:
                image_array[x + i, y + j] = color
            except IndexError:
                continue
    if show:
        plt.imshow(image_array)
        plt.show()


def label_names(labels):
    if torch.is_tensor(labels):
        if np.all([i in [0, 1] for i in labels]):  # one hot
            return [mspacman_label_list[i] for i in labels.argmax(1)]
        else:
            return [mspacman_label_list[i] for i in labels]
    return labels
