from utils import get_labels

def get_labels_validation(idx, boxes_batch):
    labels = []
    for i, boxes in zip(idx, boxes_batch):
        labels.append(get_labels(all_validation_labels.iloc[[i]], boxes))
    return labels
