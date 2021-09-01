import torch
import pandas as pd

label_list_pacman = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
                     "white_ghost", "fruit", "save_fruit", "life", "life2", "score0", "no_label"]

label_list_carnival = ["owl", 'rabbit', 'shooter', 'refill', 'bonus', "duck",
                       "flying_duck", "score0", "no_label"]


def filter_relevant_boxes(game, boxes_batch):
    if "MsPacman" in game:
        return [box_bat[box_bat[:, 1] < 105 / 128] for box_bat in boxes_batch]
    elif "Carnival" in game:
        return
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def to_relevant(game, labels_moving):
    """
    Return Labels from line in csv file
    """
    if "MsPacman" in game:
        return labels_moving != label_list_pacman.index("no_label")
    elif "Carnival" in game:
        return labels_moving != label_list_carnival.index("no_label")
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def get_labels(row, game, boxes_batch):
    """
    Return Labels from line in csv file
    """
    if "MsPacman" in game:
        return _get_labels_mspacman(row, boxes_batch)
    elif "Carnival" in game:
        return _get_labels_carnival(row, boxes_batch)
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def get_labels_moving(row, game, boxes_batch):
    """
    Return Labels from line in csv file, but only moving objects
    """
    if "MsPacman" in game:
        return _get_labels_mspacman_moving(row, boxes_batch)
    elif "Carnival" in game:
        return _get_labels_carnival_moving(row, boxes_batch)
    else:
        raise ValueError(f"Game {game} could not be found in labels")


def _get_labels_mspacman(row, boxes_batch):
    entity_list = ["pacman", "fruit", 'sue', 'inky', 'pinky', 'blinky']
    pieces = {
        "save_fruit": (170, 136),
        "life": (169, 21),
        "life2": (169, 38),
        "score0": (183, 81)
    }
    bbs = (boxes_batch[:, :4] * (210, 210, 160, 160)).round().astype(int)
    for en in entity_list:
        if f'{en}_visible' not in row or row[f'{en}_visible'].item():
            pieces[en] = (row[f'{en}_y'].item(), row[f'{en}_x'].item())
    return labels_for_pieces(bbs, row, pieces)


def _get_labels_mspacman_moving(row, boxes_batch):
    entity_list = ['pacman', 'fruit', 'sue', 'inky', 'pinky', 'blinky']
    bbs = (boxes_batch[:, :4] * (210, 210, 160, 160)).round().astype(int)
    pieces = {}
    for en in entity_list:
        if f'{en}_visible' not in row or row[f'{en}_visible'].item():
            pieces[en] = (row[f'{en}_y'].item(), row[f'{en}_x'].item())
    return labels_for_pieces(bbs, row, pieces)


def labels_for_pieces(bbs, row, pieces):
    labels = []
    for bb in bbs:
        label = label_for_bb(bb, row, pieces)
        labels.append(label)
    return torch.LongTensor([label_list_pacman.index(lab) for lab in labels])


# TODO: How to validate no_label distance
def label_for_bb(bb, row, pieces):
    label = min(((name, bb_dist(bb, pos)) for name, pos in pieces.items()), key=lambda tup: tup[1])
    if label[1] > 15:  # dist
        label = ("no_label", 0)
    label = label[0]  # name
    if f'{label}_blue' in row and row[f'{label}_blue'].item():
        label = "blue_ghost"
    elif f'{label}_white' in row and row[f'{label}_white'].item():
        label = "white_ghost"
    return label


def bb_dist(bb, pos):
    return abs(bb[0] - pos[0]) + abs(bb[3] - pos[1])
