from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2


def augment_dict(obs, info, game):
    import ipdb; ipdb.set_trace()
    if game == "MsPacman":
        return _augment_dict_mspacman(obs, info)
    elif game == "Tennis":
        return _augment_dict_tennis(obs, info)
    elif game == "SpaceInvaders":
        return _augment_dict_spaceinvaders(obs, info)
    elif game == "Pong":
        return _augment_dict_pong(obs, info)
    else:
        raise ValueError


def find_and_merge(image, color, merge_margin):
    # adapted from   www.onooks.com/how-to-merge-neighboring-bounding-boxes
    # tuplify
    def tup(point):
        return (point[0], point[1])

    # returns true if the two boxes overlap
    def overlap(source, target):
        # unpack points
        tl1, br1 = source
        tl2, br2 = target

        # checks
        if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
            return False
        if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
            return False
        return True

    # returns all overlapping boxes
    def getAllOverlaps(boxes, bounds, index):
        overlaps = []
        for a in range(len(boxes)):
            if a != index:
                x, y, w, h = boxes[a]
                if overlap(bounds, [[x, y], [x + w, y + h]]):
                    overlaps.append(a)
        return overlaps
    # go through the contours and save the box edges
    mask = cv2.inRange(image, np.array(color), np.array(color))
    contours, hierarchy = cv2.findContours(mask.copy(), 1, 1)
    boxes = []  # each element is [[top-left], [bottom-right]]
    hierarchy = hierarchy[0]
    max_area = 5000
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x, y, w, h = cv2.boundingRect(currentContour)
        if currentHierarchy[3] < 0 and w * h < max_area:
            # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
            boxes.append([x, y, w, h])

    # go through the boxes and start merging
    # this is gonna take a long time
    finished = False
    points = [[[0, 0]]]
    while not finished:
        # set end con
        finished = True

        for point in points:
            point = point[0]

        # loop through boxes
        index = len(boxes) - 1
        while index >= 0:
            # add margin
            x, y, w, h = boxes[index]
            tl = [x, y]
            br = [x + h, y + w]
            tl[0] -= merge_margin
            tl[1] -= merge_margin
            br[0] += merge_margin
            br[1] += merge_margin

            # get matching boxes
            overlaps = getAllOverlaps(boxes, [tl, br], index)

            # check if empty
            if len(overlaps) > 0:
                # combine boxes
                # convert to a contour
                con = []
                overlaps.append(index)
                for ind in overlaps:
                    x, y, w, h = boxes[ind]
                    tl, br = [x, y], [x + w, y + h]
                    con.append([tl])
                    con.append([br])
                con = np.array(con)

                # get bounding rect
                x, y, w, h = cv2.boundingRect(con)

                # stop growing
                w -= 1
                h -= 1
                merged = [x, y, w, h]

                # highlights
                points = con

                # remove boxes from list
                overlaps.sort(reverse = True)
                for ind in overlaps:
                    del boxes[ind]
                boxes.append(merged)

                # set flag
                finished = False
                break

            # increment
            index -= 1

    # return [[box[0][1], box[0][0]] for box in boxes]
    return boxes


def find_objects(image, color, size=None, tol_s=10, position=None, tol_p=2,
                 min_distance=None, merge=False, merge_margin=3):
    """
    image: image to detects objects from
    color: fixed color of the object
    size: presuposed size
    tol_s: tolerance on the size
    position: presuposed position
    tol_p: tolerance on the position
    min_distance: minimal distance between two detected objects
    merge: merge objects if distance are too close
    merge_margin: if merge, the minimum margin
    """
    if merge:
        contours = find_and_merge(image, color, merge_margin)
    else:
        mask = cv2.inRange(image, np.array(color), np.array(color))
        cntrs, _ = cv2.findContours(mask.copy(), 1, 1)
        contours = []
        for cnt in cntrs:
            y, x, h, w = cv2.boundingRect(cnt)
            contours.append([y, x, h, w])
    detected = []
    for cnt in contours:
        y, x, h, w = cnt
        if size is not None:
            if not assert_in((w, h), size, tol_s):
                continue
        if position is not None:
            if not assert_in((x, y), position, tol_p):
                continue
        if min_distance is not None:
            too_close = False
            for det in detected:
                if abs(det[0] - x) + abs(det[1] - y) < min_distance:
                    too_close = True
                    break
            if too_close:
                continue
        detected.append((x, y))
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
                import ipdb; ipdb.set_trace()
                return False
        elif len(detected) == 1:
            x, y = detected[0][0], detected[0][1]
            labels[f"{obj}_x"] = x
            labels[f"{obj}_y"] = y
            # mark_point(obs, x, y, objects_colors[obj], show=False)
        else:
            print("problem")
            import ipdb; ipdb.set_trace()
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
    # print(labels)
    objects_colors = {"enemy": [213, 130, 74], "player": [92, 186, 92],
                      "ball": [236, 236, 236], "background": [144, 72, 17]}
    scores = {"score_enemy_0": ((1, 36), (1, 40)),
              "score_enemy_1": ((1, 24), (1, 28)),
              "score_player_0": ((1, 116), (1, 120)),
              "score_player_1": ((1, 104),(1, 108))}
    for obj in ['ball', "player", "enemy"]:
        x, y = labels[f'{obj}_y'] - 13, labels[f'{obj}_x'] - 48
        if obj == 'ball':
            pres = enough_color_around(obs, x, y, objects_colors["background"],
                                       threshold=4)
            if not pres:
                del labels["ball_x"]; del labels["ball_y"]
                continue
        labels[f"{obj}_x"], labels[f"{obj}_y"] = x, y
    for score in scores:
        for potential_pos in scores[score]:
            try:
                x, y = potential_pos
            except:
                import ipdb; ipdb.set_trace()
            if enough_color_around(obs, x, y, objects_colors[score.split("_")[1]],
                                   threshold=4):
                labels[f"{score}_x"] = x
                labels[f"{score}_y"] = y
                # mark_point(obs, x, y, (255, 0, 0))
                break
    return True


def assert_in(observed, target, tol):
    if type(tol) is int:
        tol = (tol, tol)
    return np.all([observed[i] < target[i] + tol[i] and
                   observed[i] > target[i] - tol[i] for i in range(2)])


def _augment_dict_mspacman(obs, info):
    labels = info['labels']
    enemy_list = ['sue', 'inky', 'pinky', 'blinky']
    base_objects_colors = {"sue": (180, 122, 48), "inky": (84, 184, 153),
                           "pinky": (198, 89, 179), "blinky": (200, 72, 72),
                           "pacman": (210, 164, 74), "fruit": (184, 50, 50)
                          } #
    vulnerable_ghost_color = (66, 114, 194)
    wo_vulnerable_ghost_color = (214, 214, 214)
    for enemy in enemy_list:
        labels[f'{enemy}_x'] = labels.pop(f'enemy_{enemy}_x')  # renaming
        labels[f'{enemy}_y'] = labels.pop(f'enemy_{enemy}_y')  # renaming
        labels[f'{enemy}_visible'] = True
        labels[f'{enemy}_blue'] = False
        labels[f'{enemy}_white'] = False
        x, y = labels[f'{enemy}_x'], labels[f'{enemy}_y']
        x_t, y_t = y + 7, x - 9  # x and y in the tensor
        if not enough_color_around(obs, x_t, y_t, base_objects_colors[enemy]):
            if enough_color_around(obs, x_t, y_t, vulnerable_ghost_color): # enemy is blue
                labels[f'{enemy}_blue'] = True
            elif enough_color_around(obs, x_t, y_t, wo_vulnerable_ghost_color): # enemy is white
                labels[f'{enemy}_white'] = True
            else:
                labels[f'{enemy}_visible'] = False
    x, y = labels['player_x'], labels['player_y']
    x_t, y_t = y + 8, x - 9  # x and y in the tensor
    if not color_around(obs, x_t, y_t, base_objects_colors["pacman"]):
            return False
    labels['fruit_visible'] = False
    x, y = labels['fruit_x'], labels['fruit_y']
    x_t, y_t = y + 8, x - 9  # x and y in the tensor
    if color_around(obs, x_t, y_t, base_objects_colors["fruit"]):
        labels['fruit_visible'] = True
    return True


def _augment_dict_spaceinvaders(obs, info):
    labels = info['labels']
    objects = {"planet": (151, 25, 122), # "player": (50, 132, 50),
               "invader":  (134, 134, 29), "shield": (181, 83, 40),
               "missile": (142, 142, 142)}
    scores = {"green_scores":  (50, 132, 50),  "yellow_scores": (162, 134, 56)}
    y_player = labels['player_x']
    x_player = 185
    labels['player'] = [x_player, y_player]
    del labels["player_x"]
    invaders = find_objects(obs, objects['invader'], size=(10, 10), tol_s=5)
    labels['invaders'] = invaders
    planet = find_objects(obs, objects['planet'], size=(7, 7), tol_s=3)
    if planet:
        labels['planet'] = planet[0]
    else:
        labels['planet'] = []
    # for inv in invaders:
    #     mark_point(obs, inv[0], inv[1], color=(255, 255, 0), show=False, size=1)
    for score in scores:
        found = find_objects(obs, scores[score], position=(10, 100), tol_p=(20, 100),
                              merge=True, merge_margin=3)  # scores at the top
        # for found in founds:
        #     mark_point(obs, found[0], found[1], color=(255, 255, 255), show=False, size=1)
        labels[score] = found
    for targ_obj, mind in zip(["shield", "missile"], [26, 7]):
        found = find_objects(obs, objects[targ_obj], min_distance=mind)
        labels[f'{targ_obj}s'] = found
    # print(labels)
    return True



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


def color_around(image_array, x, y, color, size=2):
    """
    checks if the color is present in the square around the (x, y) point.
    """
    ran = list(range(-size, size + 1))
    while x + size > image_array.shape[0] or y + size > image_array.shape[1]:
        size -= 1
    points_around = [(image_array[x + i, y + j] == color).all()
                     for i in ran for j in ran]
    return np.any(points_around)


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


def mark_point(image_array, x, y, color=(255, 0, 0), size=2, show=True, cross=True):
    """
    marks a point on the image at the (x,y) position and displays it
    """
    ran = list(range(-size, size + 1))
    for i in ran:
        for j in ran:
            if not cross or i == j or i == -j:
                image_array[x + i, y + j] = color
    if show:
        plt.imshow(image_array)
        plt.show()


def show_image(image_array):
    """
    shows the image array using matplotlib.pyplot
    """
    plt.imshow(image_array)
    plt.show()


def enough_color_around(image_array, x, y, color, size=3, threshold=10):
    """
    checks if the color is present in the square of (2*size+1) x (2*size+1)
    around the (x, y) point.
    """
    ran = list(range(-size, size + 1))
    points_around = [(image_array[x + i, y + j] == color).all()
                     for i in ran for j in ran]
    return np.sum(points_around) >= threshold


def load_agent(path):
    from mushroom_rl.utils.parameters import Parameter
    from mushroom_rl.algorithms.agent import Agent
    agent = Agent.load(path)
    epsilon_test = Parameter(value=0.05)
    agent.policy.set_epsilon(epsilon_test)
    agent.policy._predict_params = {} # mushroom_rl compatibility
    return agent


def dict_to_serie(info_dict):
    info_dict['labels']['lives'] = info_dict['ale.lives']
    return pd.Series(info_dict['labels'])
    # return pd.DataFrame(info_dict['labels'], index='columns')
