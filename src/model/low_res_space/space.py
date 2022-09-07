import torch
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDict
from .lr_arch import lr_arch as arch
from .fg import LrSpaceFg
from .bg import LrSpaceBg
import time


class LrSpace(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.fg_module = LrSpaceFg()
        self.bg_module = LrSpaceBg()

    # @profile
    def forward(self, x, motion, motion_z_pres, motion_z_where, global_step):
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
