import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from .utils import NumericalRelaxedBernoulli, kl_divergence_bern_bern, get_boundary_kernel_new, get_boundary_kernel
from .utils import spatial_transform, linear_annealing, inverse_spatial_transform
from .lr_arch import lr_arch as arch
import time


class LrSpaceFg(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.img_encoder = ImgEncoderFg()
        self.z_what_net = ZWhatEnc()
        self.glimpse_dec = GlimpseDec()
        self.boundary_kernel = get_boundary_kernel_new(kernel_size=32, boundary_width=6)

        self.fg_sigma = arch.fg_sigma
        # Temperature for gumbel-softmax
        self.register_buffer('tau', torch.tensor(arch.tau_start_value))

        # Priors
        self.register_buffer('prior_z_pres_prob', torch.tensor(arch.z_pres_start_value))
        self.register_buffer('prior_what_mean', torch.zeros(1))
        self.register_buffer('prior_what_std', torch.ones(1))
        self.register_buffer('prior_depth_mean', torch.zeros(1))
        self.register_buffer('prior_depth_std', torch.ones(1))
        self.prior_scale_mean_new = torch.tensor(arch.z_scale_mean_start_value)
        self.prior_scale_std_new = torch.tensor(arch.z_scale_std_value)
        self.prior_shift_mean_new = torch.tensor(0.)
        self.prior_shift_std_new = torch.tensor(1.)
        self.boundary_filter = get_boundary_kernel(sigma=20)
        self.register_buffer('prior_scale_mean',
                             torch.tensor([arch.z_scale_mean_start_value] * 2).view(arch.z_where_scale_dim, 1, 1))
        self.register_buffer('prior_scale_std',
                             torch.tensor([arch.z_scale_std_value] * 2).view(arch.z_where_scale_dim, 1, 1))
        self.register_buffer('prior_shift_mean',
                             torch.tensor([0., 0.]).view(arch.z_where_shift_dim, 1, 1))
        self.register_buffer('prior_shift_std',
                             torch.tensor([1., 1.]).view(arch.z_where_shift_dim, 1, 1))

    @property
    def z_what_prior(self):
        return Normal(self.prior_what_mean, self.prior_what_std)

    @property
    def z_depth_prior(self):
        return Normal(self.prior_depth_mean, self.prior_depth_std)

    @property
    def z_scale_prior(self):
        return Normal(self.prior_scale_mean_new, self.prior_scale_std_new)

    @property
    def z_shift_prior(self):
        return Normal(self.prior_shift_mean_new, self.prior_shift_std_new)

    def anneal(self, global_step):
        """
        Update everything

        :param global_step: global step (training)
        :return:
        """
        self.prior_z_pres_prob = linear_annealing(self.prior_z_pres_prob.device, global_step,
                                                  arch.z_pres_start_step, arch.z_pres_end_step,
                                                  arch.z_pres_start_value, arch.z_pres_end_value)
        self.prior_scale_mean_new = linear_annealing(self.prior_z_pres_prob.device, global_step,
                                                     arch.z_scale_mean_start_step, arch.z_scale_mean_end_step,
                                                     arch.z_scale_mean_start_value, arch.z_scale_mean_end_value)
        self.tau = linear_annealing(self.tau.device, global_step,
                                    arch.tau_start_step, arch.tau_end_step,
                                    arch.tau_start_value, arch.tau_end_value)

    # @profile
    def forward(self, x, motion, motion_z_pres, motion_z_where, global_step):
        """
        Forward pass

        :param x: (B, 3, H, W)
        :param motion: (B, 1, H, W)
        :param motion_z_pres: z_pres hint from motion
        :param motion_z_where: z_where hint from motion
        :param global_step: global step (training)
        :return:
            fg_likelihood: (B, 3, H, W)
            y_nobg: (B, 3, H, W), foreground reconstruction
            alpha_map: (B, 1, H, W)
            kl: (B,) total foreground kl
            boundary_loss: (B,)
            log: a dictionary containing anything we need for visualization
        """
        B = x.size(0)
        # if global_step:
        self.anneal(global_step)

        # Everything is (B, G*G, D), where D varies
        z_pres, z_depth, z_scale, z_shift, z_where, z_pres_logits, z_depth_post, \
        z_scale_post, z_shift_post, z_pres_prob_pure, \
        z_where_pure = self.img_encoder(x, motion_z_pres, motion_z_where, self.tau, global_step)

        # (B, 3, H, W) -> (B*G*G, 3, H, W). Note we must use repeat_interleave instead of repeat
        x_repeat = torch.repeat_interleave(x, arch.G ** 2, dim=0)
        # (B*G*G, 3, H, W), where G is the grid size
        # Extract glimpse
        x_att = spatial_transform(x_repeat, z_where.view(B * arch.G ** 2, 4),
                                  (B * arch.G ** 2, 3, arch.glimpse_size, arch.glimpse_size))

        # (B*G*G, D)
        z_what, z_what_post = self.z_what_net(x_att)

        # Decode z_what into small reconstructed glimpses
        # All (B*G*G, 3, H, W)
        o_att, alpha_att = self.glimpse_dec(z_what)
        # z_pres: (B, G*G, 1) -> (B*G*G, 1, 1, 1)
        alpha_att_hat = alpha_att * z_pres.view(-1, 1, 1, 1)
        # (B*G*G, 3, H, W)
        y_att = alpha_att_hat * o_att

        # Compute pixel-wise object weights
        # (B*G*G, 1, H, W). These are glimpse size
        importance_map = alpha_att_hat * 100.0 * torch.sigmoid(-z_depth.view(B * arch.G ** 2, 1, 1, 1))
        # (B*G*G, 1, H, W). These are of full resolution
        importance_map_full_res = inverse_spatial_transform(importance_map, z_where.view(B * arch.G ** 2, 4),
                                                            (B * arch.G ** 2, 1, *arch.img_shape))

        # (B*G*G, 1, H, W) -> (B, G*G, 1, H, W)
        importance_map_full_res = importance_map_full_res.view(B, arch.G ** 2, 1, *arch.img_shape)

        # flow_att = torch.ones_like(flow_att) * (1 - flow_cooling_scaling) + flow_cooling_scaling * flow_att
        # importance_map_full_res = importance_map_full_res * flow_att
        # Normalize (B, >G*G<, 1, H, W)
        importance_map_full_res_norm = torch.softmax(importance_map_full_res, dim=1)
        # To full resolution
        # (B*G*G, 3, H, W) -> (B, G*G, 3, H, W)
        y_each_cell = inverse_spatial_transform(y_att, z_where.view(B * arch.G ** 2, 4),
                                                (B * arch.G ** 2, 3, *arch.img_shape)).view(B, arch.G ** 2, 3,
                                                                                            *arch.img_shape)
        # Weighted sum, (B, 3, H, W)
        y_nobg = (y_each_cell * importance_map_full_res_norm).sum(dim=1)

        # To full resolution
        # (B*G*G, 1, H, W) -> (B, G*G, 1, H, W)
        alpha_map_pure = inverse_spatial_transform(alpha_att_hat, z_where.view(B * arch.G ** 2, 4),
                                                   (B * arch.G ** 2, 1, *arch.img_shape)).view(B, arch.G ** 2, 1,
                                                                                               *arch.img_shape)
        # Weighted sum, (B, 1, H, W)
        alpha_map_pure = (alpha_map_pure * importance_map_full_res_norm).sum(dim=1)
        motion_cooling = max(0, 1 - (global_step - arch.motion_cooling_start_step) / arch.motion_cooling_end_step)
        # (B, 1, H, W)
        alpha_map = (1 - motion_cooling) * alpha_map_pure + motion_cooling * motion

        # Everything is computed. Now let's compute loss
        # Compute KL divergences
        # (B, G*G, 1)
        kl_z_pres = kl_divergence_bern_bern(z_pres_logits, self.prior_z_pres_prob)

        # (B, G*G, 1)
        kl_z_depth = kl_divergence(z_depth_post, self.z_depth_prior)

        # (B, G*G, 2)
        kl_z_scale = kl_divergence(z_scale_post, self.z_scale_prior)
        kl_z_shift = kl_divergence(z_shift_post, self.z_shift_prior)

        # Reshape z_what and z_what_post
        # (B*G*G, D) -> (B, G*G, D)
        z_what = z_what.view(B, arch.G ** 2, arch.z_what_dim)
        z_what_post = Normal(*[x.view(B, arch.G ** 2, arch.z_what_dim)
                               for x in [z_what_post.mean, z_what_post.stddev]])
        # (B, G*G, D)
        kl_z_what = kl_divergence(z_what_post, self.z_what_prior)

        # dimensionality check
        assert ((kl_z_pres.size() == (B, arch.G ** 2, 1)) and
                (kl_z_depth.size() == (B, arch.G ** 2, 1)) and
                (kl_z_scale.size() == (B, arch.G ** 2, 2)) and
                (kl_z_shift.size() == (B, arch.G ** 2, 2)) and
                (kl_z_what.size() == (B, arch.G ** 2, arch.z_what_dim))
                )

        # Reduce (B, G*G, D) -> (B,)
        kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what = [
            x.flatten(start_dim=1).sum(1) for x in [kl_z_pres, kl_z_depth, kl_z_scale, kl_z_shift, kl_z_what]
        ]
        # (B,)
        kl_z_where = kl_z_scale + kl_z_shift

        # Compute foreground likelhood
        fg_dist = Normal(y_nobg, self.fg_sigma)

        fg_likelihood = fg_dist.log_prob(x)

        kl = kl_z_what + kl_z_where + kl_z_pres + kl_z_depth

        # Dimensionality check
        assert (
                (z_pres.size() == (B, arch.G ** 2, 1)) and
                (z_depth.size() == (B, arch.G ** 2, 1)) and
                (z_scale.size() == (B, arch.G ** 2, 2)) and
                (z_shift.size() == (B, arch.G ** 2, 2)) and
                (z_where.size() == (B, arch.G ** 2, 4)) and
                (z_what.size() == (B, arch.G ** 2, arch.z_what_dim))
        )
        log = {
            'fg': y_nobg,
            'z_what': z_what,
            'z_where': z_where,
            'z_where_pure': z_where_pure,
            'z_pres': z_pres,
            'z_depth': z_depth,
            'motion_z_pres': motion_z_pres,
            'motion_z_where': motion_z_where,
            'z_pres_prob': torch.sigmoid(z_pres_logits),
            'z_pres_prob_pure': torch.sigmoid(8.8 * z_pres_prob_pure),
            'prior_z_pres_prob': self.prior_z_pres_prob.unsqueeze(0),
            'o_att': o_att,
            'alpha_att_hat': alpha_att_hat,
            'alpha_att': alpha_att,
            'alpha_map': alpha_map,
            'alpha_map_pure': alpha_map_pure,
            'importance_map_full_res_norm': importance_map_full_res_norm,

            'kl_z_what': kl_z_what,
            'kl_z_pres': kl_z_pres,
            'kl_z_scale': kl_z_scale,
            'kl_z_shift': kl_z_shift,
            'kl_z_depth': kl_z_depth,
            'kl_z_where': kl_z_where,
        }
        return fg_likelihood, y_nobg, alpha_map, kl, log


class ImgEncoderFg(nn.Module):
    """
    Foreground image encoder.
    """

    def __init__(self):
        super(ImgEncoderFg, self).__init__()

        assert arch.G in [4, 8, 16]
        # Adjust stride such that the output dimension of the volume matches (G, G, ...)
        last_stride = 2 if arch.G in [8, 4] else 1
        second_to_last_stride = 2 if arch.G in [4] else 1

        # Foreground Image Encoder in the paper
        # Encoder: (B, C, Himg, Wimg) -> (B, E, G, G)
        # G is H=W in the paper
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            # 64, 64
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            # 32, 32
            # 16, 16
            nn.Conv2d(32, 64, 3, second_to_last_stride, 1),
            nn.CELU(),
            nn.GroupNorm(16, 64),
            nn.Conv2d(64, arch.img_enc_dim_fg, 3, last_stride, 1),
            nn.CELU(),
            nn.GroupNorm(8, arch.img_enc_dim_fg)
        )

        # Residual Connection in the paper
        # Remark: this residual connection is not important
        # Lateral connection (B, E, G, G) -> (B, E, G, G)
        self.enc_lat = nn.Sequential(
            nn.Conv2d(arch.img_enc_dim_fg, arch.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, arch.img_enc_dim_fg),
            nn.Conv2d(arch.img_enc_dim_fg, arch.img_enc_dim_fg, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, arch.img_enc_dim_fg)
        )

        # Residual Encoder in the paper
        # Remark: also not important
        # enc + lateral -> enc (B, 2*E, G, G) -> (B, 128, G, G)
        self.enc_cat = nn.Sequential(
            nn.Conv2d(arch.img_enc_dim_fg * 2, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128)
        )

        # Image encoding -> latent distribution parameters (B, 128, G, G) -> (B, D, G, G)
        self.z_scale_net = nn.Conv2d(128, arch.z_where_scale_dim * 2, 1)
        self.z_shift_net = nn.Conv2d(128, arch.z_where_shift_dim * 2, 1)
        self.z_pres_net = nn.Conv2d(128, arch.z_pres_dim, 1)
        self.z_depth_net = nn.Conv2d(128, arch.z_depth_dim * 2, 1)

        # (G, G). Grid center offset. (offset_x[i, j], offset_y[i, j]) is the center for cell (i, j)
        offset_y, offset_x = torch.meshgrid([torch.arange(arch.G), torch.arange(arch.G)])

        # (2, G, G). I do this just to ensure that device is correct.
        self.register_buffer('offset', torch.stack((offset_x, offset_y), dim=0).float())

    def forward(self, x, motion_z_pres, motion_z_where, tau, global_step):
        """
        Given image, infer z_pres, z_depth, z_where

        :param x: (B, 3, H, W)
        :param motion_z_pres: z_pres hint from motion
        :param motion_z_where: z_where hint from motion
        :param tau: temperature for the relaxed bernoulli
        :param global_step: step for cooling
        :return
            z_pres: (B, G*G, 1)
            z_depth: (B, G*G, 1)
            z_scale: (B, G*G, 2)
            z_shift: (B, G*G, 2)
            z_where: (B, G*G, 4)
            z_pres_logits: (B, G*G, 1)
            z_depth_post: Normal, (B, G*G, 1)
            z_scale_post: Normal, (B, G*G, 2)
            z_shift_post: Normal, (B, G*G, 2)
        """
        B = x.size(0)
        # (B, C, H, W)
        img_enc = self.enc(x)
        # (B, E, G, G)
        lateral_enc = self.enc_lat(img_enc)
        # (B, 2E, G, G) -> (B, 128, H, W) / G, G
        cat_enc = self.enc_cat(torch.cat((img_enc, lateral_enc), dim=1))
        cat_enc_z_pres = cat_enc

        def reshape(*args):
            """(B, D, G, G) -> (B, G*G, D)"""
            out = []
            for x in args:
                B, D, G, G = x.size()
                y = x.permute(0, 2, 3, 1).view(B, G * G, D)
                out.append(y)
            return out[0] if len(args) == 1 else out

        # Compute posteriors
        # (B, 1, G, G)
        z_pres_original = self.z_pres_net(cat_enc_z_pres)
        z_pres_logits_pure = torch.tanh(z_pres_original)
        motion_cooling_scaling = max(0, 1 - (global_step - arch.motion_cooling_start_step) / arch.motion_cooling_end_step)
        transformed_motion_z_pres = motion_z_pres * 2 - 1
        # (B, 1, G, G) - > (B, G*G, 1)
        z_pres_logits = (1 - motion_cooling_scaling) * reshape(
            z_pres_logits_pure) + motion_cooling_scaling * transformed_motion_z_pres
        # 8.8 is only here to allow sigmoid(tanh(x)) to be around zero and one
        z_pres_logits = 8.8 * z_pres_logits
        z_pres_post = NumericalRelaxedBernoulli(logits=z_pres_logits, temperature=tau)
        # Unbounded
        z_pres_y = z_pres_post.rsample()
        # in (0, 1)
        z_pres = torch.sigmoid(z_pres_y)

        # (B, 1, G, G)
        z_depth_mean, z_depth_std = self.z_depth_net(cat_enc).chunk(2, 1)
        # (B, 1, G, G) -> (B, G*G, 1)
        z_depth_mean, z_depth_std = reshape(z_depth_mean, z_depth_std)
        z_depth_std = F.softplus(z_depth_std)
        z_depth_post = Normal(z_depth_mean, z_depth_std)
        # (B, G*G, 1)
        z_depth = z_depth_post.rsample()

        # (B, 2, G, G)
        scale_std_bias = 1e-15
        z_scale_mean, _z_scale_std = self.z_scale_net(cat_enc).chunk(2, 1)
        z_scale_std = F.softplus(_z_scale_std) + scale_std_bias
        # (B, 2, G, G) -> (B, G*G, 2)
        z_scale_mean, z_scale_std = reshape(z_scale_mean, z_scale_std)
        z_scale_post = Normal(z_scale_mean, z_scale_std)
        z_scale = z_scale_post.rsample()

        # (B, 2, G, G)
        z_shift_mean, z_shift_std = self.z_shift_net(cat_enc).chunk(2, 1)
        z_shift_std = F.softplus(z_shift_std)
        # (B, 2, G, G) -> (B, G*G, 2)
        z_shift_mean, z_shift_std = reshape(z_shift_mean, z_shift_std)
        z_shift_post = Normal(z_shift_mean, z_shift_std)
        z_shift = z_shift_post.rsample()

        # scale: unbounded to (0, 1), (B, G*G, 2)
        z_scale = z_scale.sigmoid()
        # offset: (2, G, G) -> (G*G, 2)
        offset = self.offset.permute(1, 2, 0).view(arch.G ** 2, 2)
        # (B, G*G, 2) and (G*G, 2)
        # where: (-1, 1)(local) -> add center points -> (0, 2) -> (-1, 1)
        z_shift = (2.0 / arch.G) * (offset + 0.5 + z_shift.tanh()) - 1

        # (B, G*G, 4)
        z_where_pure = torch.cat((z_scale, z_shift), dim=-1)
        z_where = (1 - motion_cooling_scaling) * z_where_pure + motion_cooling_scaling * motion_z_where
        # Check dimensions
        assert (
                (z_pres.size() == (B, arch.G ** 2, 1)) and
                (z_depth.size() == (B, arch.G ** 2, 1)) and
                (z_shift.size() == (B, arch.G ** 2, 2)) and
                (z_scale.size() == (B, arch.G ** 2, 2)) and
                (z_where.size() == (B, arch.G ** 2, 4))
        )

        return z_pres, z_depth, z_scale, z_shift, z_where, \
               z_pres_logits, z_depth_post, z_scale_post, z_shift_post, z_pres_logits_pure, \
               z_where_pure


class ZWhatEnc(nn.Module):

    def __init__(self):
        super(ZWhatEnc, self).__init__()

        self.enc_cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.CELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
        )

        self.enc_what = nn.Linear(128, arch.z_what_dim * 2)

    def forward(self, x):
        """
        Encode a (32, 32) glimpse into z_what

        :param x: (B, C, H, W)
        :return:
            z_what: (B, D)
            z_what_post: (B, D)
        """
        x = self.enc_cnn(x)
        # (B, D), (B, D)
        z_what_mean, z_what_std = self.enc_what(x.flatten(start_dim=1)).chunk(2, -1)
        z_what_std = F.softplus(z_what_std)
        z_what_post = Normal(z_what_mean, z_what_std)
        z_what = z_what_post.rsample()

        return z_what, z_what_post


class GlimpseDec(nn.Module):
    """Decoder z_what into reconstructed objects"""

    def __init__(self):
        super(GlimpseDec, self).__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(arch.z_what_dim, 128, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 32 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 32),

            nn.Conv2d(32, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
        )

        self.dec_o = nn.Conv2d(16, 3, 3, 1, 1)

        self.dec_alpha = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, x):
        """
        Decoder z_what into glimpse

        :param x: (B, D)
        :return:
            o_att: (B, 3, H, W)
            alpha_att: (B, 1, H, W)
        """
        x = self.dec(x.view(x.size(0), -1, 1, 1))

        o = torch.sigmoid(self.dec_o(x))
        alpha = torch.sigmoid(self.dec_alpha(x))

        return o, alpha
