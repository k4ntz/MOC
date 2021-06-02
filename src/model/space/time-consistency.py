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
        over_time = []
        for i in x.shape[1]:
            over_time.append(self.space(x[:, i], global_step))
        # (T, B, G*G, D)
        z_whats = torch.tensor([get_log(res)['z_what'] for res in over_time])
        z_what_deltas = z_whats[1:] - z_whats[:-1]

        losses = torch.tensor([get_loss(res) for res in over_time])
        loss = losses.mean() + self.adjacency_weight * z_what_deltas.mean()
        log = {
            'z_what_deltas': z_what_deltas,
            'space_log': [get_log(res) for res in over_time]
        }
        return loss, log


def get_log(res):
    return res[1]


def get_loss(res):
    return res[0]
