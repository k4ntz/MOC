import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from .arch import arch
from .fg import SpaceFg
from .bg import SpaceBg
from .space import Space
from rtpt import RTPT
from .arch import arch


class TcSpace(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.space = Space()
        self.adjacency_weight = arch.adjacent_consistency_weight
        self.pres_inconsistency_weight = arch.pres_inconsistency_weight
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x, global_step):
        """
        Inference.
        With time-dimension for consistency
        :param x: (B, T, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalar. Note it will be better to return (B,)
            log: a dictionary for visualization
        """

        # over_time = []
        # for i in range(x.shape[1]):
        #     over_time.append(self.space(x[:, i], global_step))


        # # (T, B, G*G, 1)
        # z_press = torch.stack([get_log(res)['z_pres'] for res in over_time])
        # # (T, B, G*G, 4)
        # z_wheres = torch.stack([get_log(res)['z_where'] for res in over_time])
        # # (T-2, B, G*G, 1)
        # z_pres_similarity = (1 - sqDelta(z_press[2:], z_press[:-2]))
        # z_pres_deltas = (sqDelta(z_press[2:], z_press[1:-1]) + sqDelta(z_press[:-2], z_press[1:-1]))
        # z_pres_inconsistencies = z_pres_similarity * z_pres_deltas

        # losses = torch.tensor([get_loss(res) for res in over_time])
        # loss = losses.mean()
        # log = {
        #     # 'z_what_deltas': z_what_deltas,
        #     # 'z_pres_inconsistencies': z_pres_inconsistencies,
        #     # 'z_pres_similarity': z_pres_similarity,
        #     # 'z_pres_deltas': z_pres_deltas,
        #     'space_log': [get_log(res) for res in over_time]
        # }
        y = [x[:, i] for i in range(4)]
        responses = [self.space(y1, global_step) for y1 in y]
        # (T, B, G*G, D)
        z_whats = torch.stack([get_log(r)['z_what'] for r in responses])
        # (T-1, B, G*G, D)
        z_what_deltas = sq_delta(z_whats[1:], z_whats[:-1])
        z_what_loss = torch.sum(z_what_deltas)

        log = {
            'space_log': [get_log(r) for r in responses],
            'z_what_loss': z_what_loss
        }
        loss = sum([get_loss(r) for r in responses])
        # print(f'z_what_magnitude: {z_what_loss}')
        return loss, log


def sq_delta(t1, t2):
    return (t1 - t2).square()


def get_log(res):
    return res[1]


def get_loss(res):
    loss = res[0]
    return loss


def z_what_consistency(responses):
    # (T, B, G*G, D)
    z_whats = torch.stack([get_log(r)['z_what'] for r in responses])
    B = z_whats.shape[1]
    T = z_whats.shape[0]
    # (T, B, G*G, 1) -> (T * B, 1, G, G)
    z_pres = torch.stack([get_log(r)['z_pres_prob'] for r in responses]).reshape((T * B, 1, arch.G, arch.G))
    conv = nn.Conv2d(1, 1, 3, stride=1)
    z_pres_smoothed = conv(z_pres)
    z_what_weighted = z_whats * z_pres.reshape(T, B, G*G, 1)
    z_what_smoothed = z_whats * z_pres_smoothed.reshape(T, B, G*G, 1)
    # (T-1, B, G, G)
    cos = nn.CosineSimilarity(dim=4, eps=1e-6)(z_what_smoothed[:-1], z_what_weighted[1:])
    return -torch.sum(cos)
