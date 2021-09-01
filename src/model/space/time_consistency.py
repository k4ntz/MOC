import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from .arch import arch
from .fg import SpaceFg
from .bg import SpaceBg
from .space import Space
from rtpt import RTPT


class TcSpace(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.space = Space()
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x, global_step):
        """
        Inference.
        With time-dimension for consistency
        :param x: (B, T, 3/4, H, W)
        :param global_step: global training step
        :return:
            loss: a scalar. Note it will be better to return (B,)
            log: a dictionary for visualization
        """
        y = [x[:, i] for i in range(4)]
        zero = torch.tensor(0.0).to(x.device)
        responses = [self.space(y1, global_step) for y1 in y]
        z_what_loss = z_what_loss_grid_cell(responses) if arch.adjacent_consistency_weight > 1e-3 else zero
        z_pres_loss = z_pres_loss_grid_cell(responses) if arch.pres_inconsistency_weight > 1e-3 else zero
        z_what_loss_pool = z_what_consistency_pool(responses) if arch.area_pool_weight > 1e-3 else zero
        z_what_loss_objects, objects_detected = z_what_consistency_objects(
            responses) if arch.area_object_weight > 1e-3 else (zero, zero)
        area_object_scaling = min(1, global_step / arch.full_object_weight)
        log = {
            'space_log': [get_log(r) for r in responses],
            'z_what_loss': z_what_loss,
            'z_what_loss_pool': z_what_loss_pool,
            'z_what_loss_objects': z_what_loss_objects,
            'z_pres_loss': z_pres_loss,
            'objects_detected': objects_detected
        }
        loss = sum([get_loss(r) for r in responses]) \
               + z_what_loss * arch.adjacent_consistency_weight \
               + z_pres_loss * arch.pres_inconsistency_weight \
               + z_what_loss_pool * arch.area_pool_weight \
               + z_what_loss_objects * area_object_scaling * arch.area_object_weight
        return loss, log


def sq_delta(t1, t2):
    return (t1 - t2).square()


def get_log(res):
    return res[1]


def get_loss(res):
    loss = res[0]
    return loss


def z_what_consistency_objects(responses):
    # (T, B, G*G, 1)
    cos = nn.CosineSimilarity(dim=1)
    z_pres = torch.stack([get_log(r)['z_pres_prob'] for r in responses])
    # (T, B, G*G, D)
    z_whats = torch.stack([get_log(r)['z_what'] for r in responses])
    B = z_whats.shape[1]
    T = z_whats.shape[0]
    D = z_whats.shape[3]
    z_pres = z_pres.reshape(T, B, arch.G, arch.G)
    z_pres_idx = (z_pres[:-1] > arch.object_threshold).nonzero(as_tuple=False)
    z_whats = z_whats.reshape(T, B, arch.G, arch.G, D)
    # (T, B, G+2, G+2)
    z_pres_same_padding = torch.nn.functional.pad(z_pres, (1, 1, 1, 1), mode='circular')
    # (T, B, G+2, G+2, D)
    z_what_same_padding = torch.nn.functional.pad(z_whats, (0, 0, 1, 1, 1, 1), mode='circular')
    # idx: (4,)
    object_consistency_loss = torch.tensor(0.0).to(z_whats.device)
    for idx in z_pres_idx:
        # (3, 3)
        z_pres_area = z_pres_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
        # (3, 3, D)
        z_what_area = z_what_same_padding[idx[0] + 1, idx[1], idx[2]:idx[2] + 3, idx[3]:idx[3] + 3]
        # Tuple((#hits,) (#hits,))
        z_what_idx = (z_pres_area > arch.object_threshold).nonzero(as_tuple=True)
        # (1, D)
        z_what_prior = z_whats[idx.tensor_split(4)]
        # (#hits, D)
        z_whats_now = z_what_area[z_what_idx]
        if z_whats_now.nelement() == 0:
            continue
        # (#hits,)
        if arch.cosine_sim:
            z_sim = cos(z_what_prior, z_whats_now)
        else:
            z_means = nn.functional.mse_loss(z_what_prior, z_whats_now, reduction='none')
            # (#hits,) in (0,1]
            z_sim = 1 / (torch.mean(z_means, dim=1) + 1)
        object_consistency_loss += -arch.z_cos_match_weight * torch.max(z_sim) + torch.sum(z_sim)
    return object_consistency_loss, torch.tensor(len(z_pres_idx)).to(z_whats.device)


def z_what_consistency_pool(responses):
    # O(n^2) matching of only high z_pres
    # (T, B, G*G, D)
    z_whats = torch.stack([get_log(r)['z_what'] for r in responses])
    z_whats = torch.abs(z_whats)
    B = z_whats.shape[1]
    T = z_whats.shape[0]
    D = z_whats.shape[3]
    # (T, B, G*G, 1)
    z_pres = torch.stack([get_log(r)['z_pres_prob'] for r in responses])

    z_what_weighted = z_whats * z_pres
    pool = nn.MaxPool2d(3, stride=1, padding=1)
    # (T * B, D, G, G)
    z_what_reshaped = z_what_weighted.transpose(2, 3).reshape(T, B, D, arch.G, arch.G).reshape(T * B, D, arch.G, arch.G)
    z_what_smoothed = pool(z_what_reshaped)
    # (T, B, G * G, D)
    z_what_smoothed = z_what_smoothed.reshape(T, B, D, arch.G, arch.G).reshape(T, B, D, arch.G * arch.G).transpose(2, 3)
    if arch.cosine_sim:
        # (T-1, B, G * G)
        cos = nn.CosineSimilarity(dim=3, eps=1e-6)(z_what_smoothed[:-1], z_what_weighted[1:])
        z_pres_weight = z_pres.squeeze()
        z_pres_weight = (z_pres_weight[:-1] + z_pres_weight[1:]) * 0.5
        return -torch.sum(cos * z_pres_weight)
    else:
        return torch.sum(nn.functional.mse_loss(z_what_smoothed[:-1], z_what_weighted[1:], reduction='none'))


def z_what_loss_grid_cell(responses):
    # (T, B, G*G, D)
    z_whats = torch.stack([get_log(r)['z_what'] for r in responses])
    # (T-1, B, G*G, D)
    z_what_deltas = sq_delta(z_whats[1:], z_whats[:-1])
    return torch.sum(z_what_deltas)


def z_pres_loss_grid_cell(responses):
    z_press = torch.stack([get_log(r)['z_pres'] for r in responses])
    # (T-2, B, G*G, 1)
    z_pres_similarity = (1 - sq_delta(z_press[2:], z_press[:-2]))
    z_pres_deltas = (sq_delta(z_press[2:], z_press[1:-1]) + sq_delta(z_press[:-2], z_press[1:-1]))
    z_pres_inconsistencies = z_pres_similarity * z_pres_deltas
    return torch.sum(z_pres_inconsistencies)
