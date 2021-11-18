from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from skimage.morphology import (disk, square)
from skimage.morphology import (erosion, dilation, opening, closing, white_tophat, skeletonize)

PLOT_BB = False


def set_plot_bb(plot_bb):
    global PLOT_BB
    PLOT_BB = plot_bb


def augment_dict(obs, info, game):
    if game == "MsPacman":
        return _augment_dict_mspacman(obs, info)
    elif game == "Tennis":
        return _augment_dict_tennis(obs, info)
    elif game == "Carnival":
        return _augment_dict_carnival(obs, info)
    elif game == "SpaceInvaders":
        return _augment_dict_space_invaders(obs, info)
    elif game == "Pong":
        return _augment_dict_pong(obs, info)
    else:
        raise ValueError


def bbs_extend(labels, key: str, stationary=False):
    labels['bbs'].extend([(*bb, "S" if stationary else "M", key) for bb in labels[key]])


def bb_by_color(labels, obs, color, key):
    labels[key] = find_objects(obs, color)
    bbs_extend(labels, key)


def _augment_dict_space_invaders(obs, info):
    labels = info['labels']
    objects_colors = {"player": (50, 132, 50), "space_ship": (151, 25, 122),
                      "enemy": (134, 134, 29), "block": (181, 83, 40),
                      "left_score": (50, 132, 50), "right_score": (162, 134, 56),
                      "bullet": (142, 142, 142)}
    labels['bbs'] = []
    bb_by_color(labels, obs, objects_colors['player'], "player")
    labels['bbs'] = [bb for bb in labels['bbs'] if bb[5] != "player" or bb[0] > 90 and bb[3] > 5]
    bb_by_color(labels, obs, objects_colors['bullet'], "bullet")

    bb_by_color(labels, obs, objects_colors['space_ship'], "space_ship")
    bb_by_color(labels, obs, objects_colors['block'], "block")
    labels['bbs'] += [(10, 4, 10, 60, "S", "left_score"), (10, 84, 10, 60, "S", "right_score")]

    detected_enemies = find_objects(obs, objects_colors['enemy'])
    cur_y = min((bb[0] for bb in detected_enemies), default=0)
    for idx in range(6):
        labels[f"enemy_{idx}"] = [bb for bb in detected_enemies if cur_y + 10 > bb[0] >= cur_y]
        bbs_extend(labels, f"enemy_{idx}")
        cur_y = max((bb[0] for bb in labels[f"enemy_{idx}"]), default=0) + 12

    plot_bounding_boxes(obs, labels["bbs"], objects_colors)
    return labels


def plot_bounding_boxes(obs, bbs, objects_colors):
    if PLOT_BB:
        for bb in bbs:
            mark_bb(obs, bb)


def find_objects(image, color, size=None, tol_s=10, position=None, tol_p=2,
                 min_distance=10):
    """
    image: image to detects objects from
    color: fixed color of the object
    size: presupposed size
    tol_s: tolerance on the size
    position: presupposed position
    tol_p: tolerance on the position
    min_distance: minimal distance between two detected objects
    """
    mask = cv2.inRange(image, np.array(color), np.array(color))
    closed = closing(mask, disk(3))
    closed = closing(closed, square(3))
    contours, _ = cv2.findContours(closed.copy(), 1, 1)
    detected = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if size is not None:
            if not assert_in((h, w), size, tol_s):
                continue
        if position is not None:
            if not assert_in((x, y), position, tol_p):
                continue
        if min_distance is not None:
            too_close = False
            for det in detected:
                if abs(det[0] - y) + abs(det[1] - x) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
        detected.append((y, x, h, w))
    return detected


def _augment_dict_tennis(obs, info):
    labels = info['labels']
    labels.clear()
    objects_colors = {"enemy": [117, 231, 194], "player": [240, 128, 128],
                      "ball": [236, 236, 236], "ball_shadow": [74, 74, 74]}
    objects_sizes = {"enemy": [23, 14], "player": [23, 14],
                     "ball": [2, 2], "ball_shadow": [2, 2]}
    objects_sizes_tol = {"enemy": 2, "player": 2, "ball": 0, "ball_shadow": 0}
    for obj in ['ball', 'player', 'enemy', 'ball_shadow']:
        detected = find_objects(obs, objects_colors[obj],
                                size=objects_sizes[obj], tol_s=5)
        if len(detected) == 0:
            if obj not in ['ball', 'ball_shadow']:
                # print(f"no {obj} detected")
                return False
        elif len(detected) == 1:
            x, y = detected[0][0], detected[0][1]
            labels[f"{obj}_x"] = x
            labels[f"{obj}_y"] = y
            # mark_point(obs, x, y, objects_colors[obj], show=False)
        else:
            print("problem")
            return False
    for obj in ['player', 'enemy']:  # scores
        detected = find_objects(obs, objects_colors[obj],
                                position=(80, 31), tol_p=(50, 3),
                                min_distance=5)
        if not (len(detected) == 1 or len(detected) == 2):
            # print("DEUCE")
            pass
        for i, det in enumerate(detected):
            x, y = det
            labels[f"{obj}_score_{i}_x"] = x
            labels[f"{obj}_score_{i}_y"] = y
    #         mark_point(obs, x, y, (255,0,0), show=False)
    # show_image(obs)
    return labels


def _augment_dict_pong(obs, info):
    labels = info['labels']
    objects_colors = {"enemy": [213, 130, 74], "player": [92, 186, 92],
                      "ball": [236, 236, 236], "background": [144, 72, 17]}
    labels['bbs'] = []
    bb_by_color(labels, obs, objects_colors['player'], "player")
    labels['bbs'] = [bb if bb[5] != "player" or bb[0] > 30 else (*bb[:4], "S", "player_score") for bb in labels['bbs']]
    bb_by_color(labels, obs, objects_colors['enemy'], "enemy")
    labels['bbs'] = [bb if bb[5] != "enemy" or bb[0] > 30 else (*bb[:4], "S", "enemy_score") for bb in labels['bbs']]
    bb_by_color(labels, obs, objects_colors['ball'], "ball")
    labels['bbs'] = [bb for bb in labels['bbs'] if bb[5] != "ball" or bb[3] < 20]
    plot_bounding_boxes(obs, labels["bbs"], objects_colors)
    return labels


def assert_in(observed, target, tol):
    if type(tol) is int:
        tol = (tol, tol)
    return np.all([target[i] + tol[i] > observed[i] > target[i] - tol[i] for i in range(2)])


cou = 0


def _augment_dict_carnival(obs, info):
    global cou
    if 'labels' in info:
        labels = info['labels']
    else:
        labels = {}
        info['labels'] = labels
    base_objects_colors = {"duck": np.array([187, 187, 53]), "flying_duck": np.array([135, 135, 35]),
                           "rabbit": np.array([192, 192, 192]),
                           "refill": np.array([0, 0, 0]), "shooter": np.array([66, 158, 130]),
                           "owl": np.array([214, 92, 92]), "pipes": np.array([0, 160, 0]),
                           "bonus": np.array([190, 75, 75])
                           }
    for label in ['ducks', 'flying_ducks', 'owls', 'rabbits', 'refills', 'shooters']:
        labels[label] = []
    H, W, C = obs.shape

    prior_main_color = (0, 0, 0)
    prior_amount = 0
    brig = np.max(obs, axis=2)
    for y in [54, 75, 96, 192]:
        x = 0
        while x < W:
            colors_around = get_colors_around(obs, x, y)
            if len(colors_around) > 1:
                main_color = list(colors_around.keys())[0] if list(colors_around.keys())[0] != (0, 0, 0) else \
                    list(colors_around.keys())[1]
                current_amount = colors_around[main_color]
                if prior_main_color == main_color and current_amount < prior_amount:
                    obj = [k for k, v in base_objects_colors.items() if tuple(v) == main_color]
                    if not obj:
                        x += 1
                        continue
                    else:
                        obj = obj[0]
                    if main_color == tuple(base_objects_colors['rabbit']) and not tr_color_around(obs, y - 4, x,
                                                                                                  main_color, size=2):
                        obj = 'refill'
                    wings, fly_y, fly_x = found_wings(obs, x - 1, y)
                    if main_color == tuple(base_objects_colors['duck']) and wings:
                        obj = 'flying_duck'
                        labels[f'{obj}s'].append((fly_x, fly_y))
                    else:
                        labels[f'{obj}s'].append((x - 1, y))
                    x += 10
                    prior_main_color = (0, 0, 0)
                    prior_amount = 0
                else:
                    x += 1
                    prior_main_color = main_color
                    prior_amount = current_amount
            else:
                x += 1
        if prior_amount > 10:
            obj = [k for k, v in base_objects_colors.items() if tuple(v) == main_color][0]
            if main_color == tuple(base_objects_colors['rabbit']) and not tr_color_around(obs, y - 4, x, main_color,
                                                                                          size=2):
                obj = 'refill'
            wings, fly_y, fly_x = found_wings(obs, x - 1, y)
            if main_color == tuple(base_objects_colors['duck']) and wings:
                obj = 'flying_duck'
                labels[f'{obj}s'].append((fly_x, fly_y))
            else:
                labels[f'{obj}s'].append((x - 1, y))
    for y in range(104, 185, 6):
        for x in range(0, 160, 4):
            if tr_color_around(obs, y, x, np.array([187, 187, 53])):
                wings, fly_y, fly_x = found_wings(obs, x, y)
                if wings:
                    if not labels['flying_ducks'] or min(
                            [(fly_y - e[1]) ** 2 + (fly_x - e[0]) ** 2 for e in labels['flying_ducks']]) > 60:
                        labels['flying_ducks'].append((fly_x, fly_y))
    # for obj, positions in labels.items():
    #     for x, y in positions:
    #         obs[y-1:y+1, x-1:x+1] = (base_objects_colors[obj[:-1]] * 1.15).astype(int)

    bullets_xs = (brig[203:207] == 157).nonzero()[1]
    labels['bullets_x'] = bullets_xs[-1] if bullets_xs.size else 0
    labels['bonus'] = (brig[29:35] == 214).any()

    return True


def print_obj(brig, y, x, size=(10, 8)):
    with np.printoptions(threshold=np.inf):
        print(brig[y - size[0]:y + size[0] + 1, x - size[1]:x + size[1] + 1])


def print_brig(brig):
    with np.printoptions(threshold=np.inf):
        for i in range(4):
            for j in range(4):
                print(brig[i * 21:i * 21 + 21, j * 16:j * 16 + 16])
            print()


def found_wings(obs, base_x, base_y):
    for y in range(base_y - 6, base_y + 7):
        row = obs[y, max(0, base_x - 8):min(160, base_x + 8)]
        if all(tuple(e) != (187, 187, 53) for e in row):
            continue
        former = (0, 0, 0)
        switches = []
        count = 0
        for e in row:
            if tuple(e) == former:
                count += 1
            else:
                switches.append(count)
                count = 1
                former = tuple(e)
        switches.append(count)
        if len([s for s in switches if s > 1]) >= 5:
            for x in range(base_x - 3, base_x + 4):
                if all(tuple(e) == (187, 187, 53) for e in obs[y, x - 1:x + 2]):
                    return True, y + 5, x
    return False, -1, -1


def _augment_dict_mspacman(obs, info):
    if 'labels' in info:
        labels = info['labels']
    else:
        labels = {}
        info['labels'] = labels
    objects_colors = {
        "sue": (180, 122, 48), "inky": (84, 184, 153),
        "pinky": (198, 89, 179), "blinky": (200, 72, 72),
        "pacman": (210, 164, 74),
        "white_ghost": (214, 214, 214), "blue_ghost": (66, 114, 194)
    }
    labels['bbs'] = []
    for obj_name in objects_colors:
        bb_by_color(labels, obs, objects_colors[obj_name], obj_name)
    fruit_color = (184, 50, 50)
    fruit_bbs = find_objects(obs, fruit_color)
    for bb in fruit_bbs:
        labels['bbs'] += [(*bb, "S" if bb[0] >= 171 else "M", "save_fruit" if bb[0] >= 171 else "fruit")]
    if tr_color_around(obs, 178, 16, (187, 187, 53)):
        labels['bbs'] += [(173, 12, 12, 8, "S", "life1")]
    if tr_color_around(obs, 178, 32, (187, 187, 53)):
        labels['bbs'] += [(173, 28, 12, 8, "S", "life2")]
    labels['bbs'] += [
        (187, 71, 7, 30, "S", "score"),
    ]
    if tr_color_around(obs, 18, 10, (228, 111, 111)):
        labels['bbs'] += [(14, 8, 7, 4, "S", "corner_block")]
    if tr_color_around(obs, 150, 10, (228, 111, 111)):
        labels['bbs'] += [(147, 8, 7, 4, "S", "corner_block")]
    if tr_color_around(obs, 18, 150, (228, 111, 111)):
        labels['bbs'] += [(14, 148, 7, 4, "S", "corner_block")]
    if tr_color_around(obs, 150, 150, (228, 111, 111)):
        labels['bbs'] += [(147, 148, 7, 4, "S", "corner_block")]
    labels['bbs'] = [(*bb[:5], bb[5] if bb[2] > 6 or bb[4] == "S" else "eyes") for bb in labels["bbs"]]
    plot_bounding_boxes(obs, labels["bbs"], objects_colors)
    return labels


def draw_names(obs, info):
    img = Image.fromarray(obs, 'RGB')
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    font = ImageFont.truetype("fonts/arial.ttf", 10)
    for enemy in ['sue', 'inky', 'pinky', 'blinky']:
        if f'enemy_{enemy}_x' in info['labels'].keys():
            x, y = info['labels'][f'enemy_{enemy}_x'], info['labels'][f'enemy_{enemy}_y']
        else:
            x, y = info['labels'][f'{enemy}_x'], info['labels'][f'{enemy}_y']
        draw.text((x, y), enemy, (255, 255, 255), font=font)
    x, y = info['labels'][f'player_x'], info['labels'][f'player_y']
    x_t, y_t = y + 7, x - 9  # x and y in the tensor
    raise ValueError  # TODO


# def pacman_just_ate(image_array, x, y, size=2):
#     """
#     checks if the color is present in the square around the (x, y) point.
#     """
#     if color_around(image_array, x, y, (210, 164, 74), size) and \
#         color_around(image_array, x, y, (177, 67, 80), size):
#         print("Pacman just ate")
#         return True
#     return False

def get_colors_around(image_array, root_x, root_y):
    """
    Counts the colors present in the square around the (x, y) point.
    """
    H, W, C = image_array.shape
    counter = Counter()
    for x in range(max(root_x - 4, 0), min(root_x + 5, W - 1)):
        for y in range(root_y - 4, root_y + 5):
            counter[tuple(image_array[y][x])] += 1
    return {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}


def tr_color_around(image_array, y, x, color, size=2):
    """
    checks if the color is present in the square around the (x, y) point.
    """
    return np.any(points_around(image_array, y, x, color, size))


def only_background(image_array, x, y, size=3):
    """
    checks if only background color is present in the square around the (x, y) point.
    """
    pink_bg = (228, 111, 111)
    blue_bg = (0, 28, 136)
    ran = list(range(-size, size + 1))
    while x + size > image_array.shape[0] or y + size > image_array.shape[1]:
        size -= 1
    points_around = [(image_array[x + i, y + j] == pink_bg).all() or
                     (image_array[x + i, y + j] == blue_bg).all()
                     for i in ran for j in ran]
    # mark_point(image_array, x, y, size)  # to double_check
    return np.all(points_around)


def mark_point(image_array, y, x, color=(255, 0, 0), size=1, show=True, cross=True):
    """
    marks a point on the image at the (x,y) position and displays it
    """
    for i in range(max(0, y - size), min(y + size + 1, 210)):
        for j in range(max(0, x - size), min(x + size + 1, 160)):
            image_array[i, j] = color
    if show:
        plt.imshow(image_array)
        plt.show()


def mark_bb(image_array, bb, color=(255, 0, 0)):
    """
    marks a bounding box on the image
    """
    y, x, h, w, moving, label = bb
    bottom = min(209, y + h)
    right = min(159, x + w)
    image_array[y:bottom, x] = color
    image_array[y:bottom, right] = color
    image_array[y, x:right] = color
    image_array[bottom, x:right] = color


def show_image(image_array, save_path=None, save=False):
    """
    shows the image array using matplotlib.pyplot
    """
    plt.imshow(image_array)
    if save:
        plt.savefig(save_path)
    else:
        plt.show()


# def dict_to_array(info_dict, game):
#     labels = info_dict['labels']
#     array = []
#     if game == "MsPacman":
#         array.append(labels['player_x']); array.append(labels['player_y'])
#         for enemy in ['blinky', 'inky', 'pinky', 'sue']:
#             array.append(labels[f'enemy_{enemy}_x']); array.append(labels[f'enemy_{enemy}_y'])
#         array.append(labels['fruit_x']); array.append(labels['fruit_y'])
#         array.append(labels['player_score']); array.append(labels['ghosts_count'])
#         array.append(labels['player_direction']); array.append(labels['num_lives'])
#         array.append(labels['dots_eaten_count'])
#         return np.array(array, dtype=np.uint16)
#     else:
#         raise ValueError


def enough_color_around(image_array, y, x, color, size=3, threshold=10):
    """
    checks if the color is present in the square of (2*size+1) x (2*size+1)
    around the (x, y) point.
    """
    return np.sum(points_around(image_array, y, x, color, size)) >= threshold


def points_around(image_array, y, x, color, size):
    return [(image_array[i, j] == color).all()
            for i in range(max(0, y - size), min(y + size + 1, 210))
            for j in range(max(0, x - size), min(x + size + 1, 160))]


def load_agent(path):
    from mushroom_rl.utils.parameters import Parameter
    from mushroom_rl.algorithms.agent import Agent
    agent = Agent.load(path)
    epsilon_test = Parameter(value=0.05)
    agent.policy.set_epsilon(epsilon_test)
    agent.policy._predict_params = {}  # mushroom_rl compatibility
    return agent


def put_lives(info_dict):
    info_dict['labels']['lives'] = info_dict['ale.lives']
    return info_dict['labels']


def dict_to_serie(info_dict):
    info_dict['labels']['lives'] = info_dict['ale.lives']
    return pd.Series(info_dict['labels'])
    # return pd.DataFrame(info_dict['labels'], index='columns')
