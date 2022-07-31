import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from .arch import arch
from .fg import SpaceFg
from .bg import SpaceBg
import time
from collections import defaultdict


class Space(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.fg_module = SpaceFg()
        self.bg_module = SpaceBg()

    # @profile
    def forward(self, x, motion=None, motion_z_pres=None, motion_z_where=None, global_step=100000000):
        """
        Inference.
        With time-dimension for consistency
        :param x: (B, 3, H, W)
        :param motion: (B, 1, H, W)
        :param motion_z_pres: z_pres hint from motion (B, G * G, 1)
        :param motion_z_where: z_where hint from motion (B, G * G, 4)
        :param global_step: global training step
        :return:
            loss: a scalar. Note it will be better to return (B,)
            log: a dictionary for visualization
        """

        # Background extraction
        # (B, 3, H, W), (B, 3, H, W), (B, T)
        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x, global_step)

        # Foreground extraction
        fg_likelihood, fg, alpha_map, kl_fg, log_fg = self.fg_module(x, motion, motion_z_pres,
                                                                     motion_z_where, global_step)

        # Fix alpha trick
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)

        # Compute final mixture likelihood
        # (B, 3, H, W)
        fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
        bg_likelihood = (bg_likelihood + (1 - alpha_map + 1e-5).log())
        # (B, 2, 3, H, W)
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
        # (B, 3, H, W)
        log_like = torch.logsumexp(log_like, dim=1)
        # (B,)
        log_like = log_like.flatten(start_dim=1).sum(1)

        # Take mean as reconstruction
        y = alpha_map * fg + (1.0 - alpha_map) * bg

        # Elbo
        elbo = log_like - kl_bg - kl_fg

        # Mean over batch
        loss = -elbo.mean()
        x = x[:, :3]
        log = {
            'imgs': x,
            'y': y,
            # (B,)
            'mse': ((y - x) ** 2).flatten(start_dim=1).sum(dim=1),
            'log_like': log_like,
            'loss': loss
        }
        log.update(log_fg)
        log.update(log_bg)

        return loss, log

    def scene_description(self, x, z_classifier, only_z_what=True, **kwargs):
        """
        Computes a dictionary where each entity is located
        :param x: (3, H, W)
        :param z_classifier: a classifier mapping encodings (z_where, z_depth and z_what concatenated)
            to labels. Has to be able to do z_classifier.predict(list of z_encodings) to return a list of labels
        :param only_z_what: predicts labels only on active (z_pres) z_what encodings, but disregards z_where / z_depth
        :return: dict[label, list of (int, int)], all positions of the object in question
        """
        _, log = self.forward(x.unsqueeze(dim=0))
        z_where, z_pres_prob, z_what, z_depth = log['z_where'], log['z_pres_prob'], log['z_what'], log['z_depth']
        z_where, z_what, z_depth = z_where.detach().cpu(), z_what.detach().cpu(), z_depth.detach().cpu()
        z_pres_prob = z_pres_prob.squeeze().detach().cpu()
        z_pres = z_pres_prob > 0.5
        if only_z_what:
            z_encs = z_what[z_pres]
        else:
            z_encs = torch.concat((z_where[z_pres], z_depth[z_pres], z_what[z_pres]), dim=1)
        labels = z_classifier.predict(z_encs, **kwargs)
        pos = z_where[2:]
        result = defaultdict(list)
        for label, p in zip(labels, pos):
            result[label].append(tuple(coordinate.item() for coordinate in p))
        return result
