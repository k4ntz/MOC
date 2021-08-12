from utils import get_labels
import pandas as pd

all_train_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/train_labels.csv")
all_validation_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/validation_labels.csv")

label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
              "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]

def get_labels_validation(idx, boxes_batch):
    labels = []
    for i, boxes in zip(idx, boxes_batch):
        labels.append(get_labels(all_validation_labels.iloc[[i]], boxes))
    return labels
