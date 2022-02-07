import numpy as np
import os
import torch
import pandas as pd

def compute_counts(boxes_pred, boxes_gt):
    """
    Compute error rates, perfect number, overcount number, undercount number

    :param boxes_pred: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param boxes_gt: [[y_min, y_max, x_min, x_max] * N] * B
    :return:
        error_rate: mean of ratio of differences
        perfect: integer
        overcount: integer
        undercount: integer
    """

    error_rates = []
    perfect = 0
    overcount = 0
    undercount = 0
    for pred, gt in zip(boxes_pred, boxes_gt):
        M, N = len(pred), len(gt)
        error_rates.append(abs(M - N) / N)
        perfect += (M == N)
        overcount += (M > N)
        undercount += (M < N)

    return np.mean(error_rates), perfect, overcount, undercount


def convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=False):
    """

    All inputs should be tensors

    :param z_where: (B, N, 4). [sx, sy, tx, ty]. N is arch.G ** 2
    :param z_pres: (B, N) Must be binary and byte tensor
    :param z_pres_prob: (B, N). In range (0, 1)
    :return: [[y_min, y_max, x_min, x_max, conf] * N] * B
    """
    B, N, _ = z_where.size()
    z_pres = z_pres.bool()
    # import ipdb; ipdb.set_trace()

    # each (B, N, 1)
    width, height, center_x, center_y = torch.split(z_where, 1, dim=-1)

    center_x = (center_x + 1.0) / 2.0
    center_y = (center_y + 1.0) / 2.0
    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2
    # (B, N, 4)
    pos = torch.cat([y_min, y_max, x_min, x_max], dim=-1)
    boxes = []
    for b in range(B):
        # (N, 4), (N,) -> (M, 4), where M is the number of z_pres == 1
        box = pos[b][z_pres[b]]
        # (N,) -> (M, 1)
        if with_conf:
            conf = z_pres_prob[b][z_pres[b]][:, None]
            # (M, 5)
            box = torch.cat([box, conf], dim=1)
        box = box.detach().cpu().numpy()
        boxes.append(box)

    return boxes

def read_boxes(path, size=None, indices=None):
    """
    Read bounding boxes and normalize to (0, 1)
    Note BB files structure was changed to left, top coordinates + width and height
    :param path: checkpointdir to bounding box root
    :param size: how many indices
    :param indices: relevant indices of the dataset
    :return: A list of list [[y_min, y_max, x_min, x_max] * N] * B
    """
    boxes_all = []
    boxes_moving_all = []

    for stack_idx in indices if (indices is not None) else range(size):
        for img_idx in range(4):
            boxes = []
            boxes_moving = []
            bbs = pd.read_csv(os.path.join(path, f"{stack_idx:05}_{img_idx}.csv"), header=None, usecols=[0, 1, 2, 3, 4])
            for y_min, y_max, x_min, x_max, moving in bbs.itertuples(index=False, name=None):
                boxes.append([y_min, y_max, x_min, x_max])
                if "M" in moving:
                    boxes_moving.append([y_min, y_max, x_min, x_max])
            boxes = np.array(boxes)
            boxes_moving = np.array(boxes_moving)
            boxes_all.append(boxes)
            boxes_moving_all.append(boxes_moving)
    return boxes_all, boxes_moving_all, boxes_moving_all


def compute_ap(pred_boxes, gt_boxes, iou_thresholds=None, recall_values=None):
    """
    Compute average precision over different iou thresholds.

    :param pred_boxes: [[y_min, y_max, x_min, x_max, conf] * N] * B
    :param gt_boxes: [[y_min, y_max, x_min, x_max] * N] * B
    :param iou_thresholds: a list of iou thresholds
    :param recall_values: a list of recall values to compute AP
    :return: AP at each iou threshold
    """
    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.1, 0.9, 9)

    AP = []
    precision_low_iou = -1
    recall_low_iou = -1
    for threshold in iou_thresholds:
        # Compute AP for this threshold
        # Number of total ground truths
        count_gt = 0
        hit = []
        # For each image, determine each prediction is a hit of not
        for pred, gt in zip(pred_boxes, gt_boxes):
            count_gt += len(gt)
            # Sort predictions within an image by decreasing confidence
            pred = sorted(pred, key=lambda x: -x[-1])

            if len(gt) == 0:
                hit.extend((False, conf) for *_, conf in pred)
                continue
            if len(pred) == 0:
                continue

            M, N = len(pred), len(gt)

            # (M, 4), (M) (N, 4)
            pred = np.array(pred)
            pred, conf = pred[:, :4], pred[:, -1]
            gt = np.array(gt)

            # (M, N)
            iou = compute_iou(pred, gt)
            # (M,)
            best_indices = np.argmax(iou, axis=1)
            # (M,)
            best_iou = iou[np.arange(M), best_indices]
            # (N,), thresholding results
            valid = best_iou > threshold
            used = [False] * N

            for i in range(M):
                # Only count first hit
                if valid[i] and not used[best_indices[i]]:
                    hit.append((True, conf[i]))
                    used[best_indices[i]] = True
                else:
                    hit.append((False, conf[i]))

        if len(hit) == 0:
            AP.append(0.0)
            continue

        # Sort
        # hit.sort(key=lambda x: -x[-1])
        hit = sorted(hit, key=lambda x: -x[-1])

        # print('yes')
        # print(hit)
        hit = [x[0] for x in hit]
        # print(hit)
        # Compute average precision
        # hit_cum = np.cumsum(np.array(hit, dtype=np.float32))
        hit_cum = np.cumsum(hit)
        num_cum = np.arange(len(hit)) + 1.0
        precision = hit_cum / num_cum
        recall = hit_cum / count_gt
        if threshold == 0.2:
            precision_low_iou = precision[-1]
            recall_low_iou = recall[-1]
        # Compute AP at selected recall values
        # print(precision)
        precs = []
        for val in recall_values:
            prec = precision[recall >= val]
            precs.append(0.0 if prec.size == 0 else prec.max())

        # Mean over recall values
        AP.append(np.mean(precs))
        # AP.extend(precs)

        # print(AP)
    print(AP)
    # mean over all thresholds
    return AP, precision_low_iou, recall_low_iou


def compute_iou(pred, gt):
    """

    :param pred: (M, 4), (y_min, y_max, x_min, x_max)
    :param gt: (N, 4)
    :return: (M, N)
    """
    compute_area = lambda b: (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2])

    area_pred = compute_area(pred)[:, None]
    area_gt = compute_area(gt)[None, :]
    # (M, 1, 4), (1, N, 4)
    pred = pred[:, None]
    gt = gt[None, :]
    # (M, 1) (1, N)

    # (M, N)
    top = np.maximum(pred[:, :, 0], gt[:, :, 0])
    bottom = np.minimum(pred[:, :, 1], gt[:, :, 1])
    left = np.maximum(pred[:, :, 2], gt[:, :, 2])
    right = np.minimum(pred[:, :, 3], gt[:, :, 3])

    h_inter = np.maximum(0.0, bottom - top)
    w_inter = np.maximum(0.0, right - left)
    # (M, N)
    area_inter = h_inter * w_inter
    iou = area_inter / (area_pred + area_gt - area_inter)

    return iou
