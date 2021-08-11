import torch
from torch import nn
from attrdict import AttrDict
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from .arch import arch


class SpaceBg(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        self.image_enc = ImageEncoderBg()

        # Compute mask hidden states given image features
        self.rnn_mask = nn.LSTMCell(arch.z_mask_dim + arch.img_enc_dim_bg, arch.rnn_mask_hidden_dim)
        self.rnn_mask_h = nn.Parameter(torch.zeros(arch.rnn_mask_hidden_dim))
        self.rnn_mask_c = nn.Parameter(torch.zeros(arch.rnn_mask_hidden_dim))

        # Dummy z_mask for first step of rnn_mask
        self.z_mask_0 = nn.Parameter(torch.zeros(arch.z_mask_dim))
        # Predict mask latent given h
        self.predict_mask = PredictMask()
        # Compute masks given mask latents
        self.mask_decoder = MaskDecoder()
        # Encode mask and image into component latents
        self.comp_encoder = CompEncoder()
        # Component decoder
        if arch.K > 1:
            self.comp_decoder = CompDecoder()
        else:
            self.comp_decoder = CompDecoderStrong()

        # ==== Prior related ====
        self.rnn_mask_prior = nn.LSTMCell(arch.z_mask_dim, arch.rnn_mask_prior_hidden_dim)
        # Initial h and c
        self.rnn_mask_h_prior = nn.Parameter(torch.zeros(arch.rnn_mask_prior_hidden_dim))
        self.rnn_mask_c_prior = nn.Parameter(torch.zeros(arch.rnn_mask_prior_hidden_dim))
        # Compute mask latents
        self.predict_mask_prior = PredictMask()
        # Compute component latents
        self.predict_comp_prior = PredictComp()
        # ==== Prior related ====

        self.bg_sigma = arch.bg_sigma

    def anneal(self, global_step):
        pass

    def forward(self, x, global_step):
        """
        Background inference backward pass

        :param x: shape (B, C, H, W)
        :param global_step: global training step
        :return:
            bg_likelihood: (B, 3, H, W)
            bg: (B, 3, H, W)
            kl_bg: (B,)
            log: a dictionary containing things for visualization
        """
        B, T, _, H, W = x.size()

        # (B, T, D)
        x_enc = self.image_enc(x)
        # (B * T, D)
        x_enc = x_enc.view(B * T, -1)

        # Mask and component latents over the K slots
        masks = []
        z_masks = []
        # These two are Normal instances
        z_mask_posteriors = []
        z_comp_posteriors = []

        # Initialization: encode x and dummy z_mask_0
        z_mask = self.z_mask_0.expand(B * arch.n_img, arch.z_mask_dim)
        h = self.rnn_mask_h.expand(B * arch.n_img, arch.rnn_mask_hidden_dim)
        c = self.rnn_mask_c.expand(B * arch.n_img, arch.rnn_mask_hidden_dim)

        K = arch.K
        for i in range(K):
            # Encode x and z_{mask, 1:k}, (B * T, D)
            rnn_input = torch.cat((z_mask, x_enc), dim=1)
            (h, c) = self.rnn_mask(rnn_input, (h, c))

            # Predict next mask from x and z_{mask, 1:k-1}
            # Reshape to re-add T dimension
            z_mask_loc, z_mask_scale = self.predict_mask(h)
            z_mask_post = Normal(z_mask_loc, z_mask_scale)
            z_mask = z_mask_post.rsample()
            z_masks.append(z_mask)
            z_mask_posteriors.append(z_mask_post)
            # Decode masks (B, T, H, W)
            mask = self.mask_decoder(z_mask).reshape(B, T, H, W)
            masks.append(mask)

        # (B, T, K, H, W), in range (0, 1)
        masks = torch.stack(masks, dim=2)

        # SBP to ensure they sum to 1
        masks = self.SBP(masks)
        # An alternative is to use softmax
        # masks = F.softmax(masks, dim=1)

        # Reshape (B, T, K, H, W) -> (B*T*K, 1, H, W)
        masks = masks.view(B * K * T, 1, H, W)

        # Concatenate images (B*K*T, 4, H, W)
        comp_vae_input = torch.cat(((masks + 1e-5).log(), x[:, :, None].repeat(1, 1, K, 1, 1, 1).view(B * K * T, 3, H, W)), dim=1)

        # Component latents, each (B*K*T, L)
        z_comp_loc, z_comp_scale = self.comp_encoder(comp_vae_input)
        z_comp_post = Normal(z_comp_loc, z_comp_scale)
        z_comp = z_comp_post.rsample()

        # Record component posteriors here. We will use this for computing KL
        z_comp_loc_reshape = z_comp_loc.view(B, T, K, -1)
        z_comp_scale_reshape = z_comp_scale.view(B, T, K, -1)
        for i in range(arch.K):
            z_comp_post_this = Normal(z_comp_loc_reshape[:, :, i], z_comp_scale_reshape[:, :, i])
            z_comp_posteriors.append(z_comp_post_this)

        # Decode into component images, (B*T*K, 3, H, W)
        comps = self.comp_decoder(z_comp)

        # Reshape (B*T*K, ...) -> (B, T, K, 3, H, W)
        comps = comps.view(B, T, K, 3, H, W)
        masks = masks.view(B, T, K, 1, H, W)

        # Now we are ready to compute the background likelihoods
        # (B, T, K, 3, H, W)
        comp_dist = Normal(comps, torch.full_like(comps, self.bg_sigma))
        log_likelihoods = comp_dist.log_prob(x[:, :, None].expand_as(comps))

        # (B, T, K, 3, H, W) -> (B, T, 3, H, W), mixture likelihood
        log_sum = log_likelihoods + (masks + 1e-5).log()
        bg_likelihood = torch.logsumexp(log_sum, dim=2)

        # Background reconstruction
        bg = (comps * masks).sum(dim=2)

        # Below we compute priors and kls

        # Conditional KLs
        z_mask_total_kl = 0.0
        z_comp_total_kl = 0.0

        # Initial h and c. This is h_1 and c_1 in the paper
        h = self.rnn_mask_h_prior.expand(B * arch.n_img, arch.rnn_mask_prior_hidden_dim)
        c = self.rnn_mask_c_prior.expand(B * arch.n_img, arch.rnn_mask_prior_hidden_dim)

        for i in range(arch.K):
            # Compute prior distribution over z_masks
            z_mask_loc_prior, z_mask_scale_prior = self.predict_mask_prior(h)
            z_mask_prior = Normal(z_mask_loc_prior, z_mask_scale_prior)
            # Compute component prior, using posterior samples
            z_comp_loc_prior, z_comp_scale_prior = self.predict_comp_prior(z_masks[i])
            z_comp_prior = Normal(z_comp_loc_prior, z_comp_scale_prior)
            # Compute KLs as we go.
            z_mask_kl = kl_divergence(z_mask_posteriors[i], z_mask_prior).sum(dim=2)
            z_comp_kl = kl_divergence(z_comp_posteriors[i], z_comp_prior).sum(dim=2)
            # (B*T)
            z_mask_total_kl += z_mask_kl
            z_comp_total_kl += z_comp_kl

            # Compute next state. Note we condition on posterior samples.
            # Again, this is conditional prior.
            (h, c) = self.rnn_mask_prior(z_masks[i], (h, c))

        # For visualization
        kl_bg = z_mask_total_kl + z_comp_total_kl
        log = {
            # (B, K, 3, H, W)
            'comps': comps,
            # (B, 1, 3, H, W)
            'masks': masks,
            # (B, 3, H, W)
            'bg': bg,
            'kl_bg': kl_bg
        }

        return bg_likelihood, bg, kl_bg, log

    @staticmethod
    def SBP(masks):
        """
        Stick breaking process to produce masks
        :param: masks (B, T, K, H, W). In range (0, 1)
        :return: (B, K, 1, H, W)
        """
        B, T, K, H, W = masks.size()

        # (B, T, 1, H, W)
        remained = torch.ones_like(masks[:, :, 0])
        new_masks = []
        for k in range(K):
            if k < K - 1:
                mask = masks[:, :, k] * remained
            else:
                mask = remained
            remained = remained - mask
            new_masks.append(mask)

        new_masks = torch.stack(new_masks, dim=1)

        return new_masks


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ImageEncoderBg(nn.Module):
    """Background image encoder"""

    def __init__(self):
        embed_size = arch.img_shape[0] // 16
        nn.Module.__init__(self)
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # 16x downsampled: (64, H/16, W/16)
            Flatten(),
            nn.Linear(64 * embed_size ** 2, arch.img_enc_dim_bg),
            nn.ELU(),
        )

    def forward(self, x):
        """
        Encoder image into a feature vector
        Args:
            x: (B, T, 3, H, W)
        Returns:
            enc: (B, T, D)
        """
        B, T, _, H, W = x.size()
        return self.enc(x.view(B*T, -1, H, W)).view(B, T, -1)


class PredictMask(nn.Module):
    """
    Predict z_mask given states from rnn. Used in inference
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.fc = nn.Linear(arch.rnn_mask_hidden_dim, arch.z_mask_dim * 2)

    def forward(self, h):
        """
        Predict z_mask given states from rnn. Used in inference

        :param h: hidden state from rnn_mask (B*T, L)
        :return:
            z_mask_loc: (B*T, D)
            z_mask_scale: (B*T, D)

        """
        x = self.fc(h)
        z_mask_loc = x[:, :arch.z_mask_dim]
        z_mask_scale = F.softplus(x[:, arch.z_mask_dim:]) + 1e-4

        return z_mask_loc, z_mask_scale


class MaskDecoder(nn.Module):
    """Decode z_mask into mask"""

    def __init__(self):
        super(MaskDecoder, self).__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(arch.z_mask_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 1, 3, 1, 1)
        )

    def forward(self, z_mask):
        """
        Decode z_mask into mask

        :param z_mask: (B, D)
        :return: mask: (B, 1, H, W)
        """
        BT = z_mask.size(0)
        # 1d -> 3d, (BT, D, 1, 1)
        z_mask = z_mask.view(BT, -1, 1, 1)
        return torch.sigmoid(self.dec(z_mask))


class CompEncoder(nn.Module):
    """
    Predict component latent parameters given image and predicted mask concatenated
    """

    def __init__(self):
        nn.Module.__init__(self)

        embed_size = arch.img_shape[0] // 16
        self.enc = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            Flatten(),
            # 16x downsampled: (64, 4, 4)
            nn.Linear(64 * embed_size ** 2, arch.z_comp_dim * 2),
        )

    def forward(self, x):
        """
        Predict component latent parameters given image and predicted mask concatenated

        :param x: (B*T*K, 3+1, H, W). Image and mask concatenated
        :return:
            z_comp_loc: (B*T*K, L)
            z_comp_scale: (B*T*K, L)
        """
        x = self.enc(x)
        z_comp_loc = x[:, :arch.z_comp_dim]
        z_comp_scale = F.softplus(x[:, arch.z_comp_dim:]) + 1e-4

        return z_comp_loc, z_comp_scale


class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        Broadcast a 1-D variable to 3-D, plus two coordinate dimensions

        :param x: (B, L)
        :param width: W
        :param height: H
        :return: (B, L + 2, W, H)
        """
        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.expand(B, L, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, 2, H, W)
        coords = coords[None].expand(B, 2, height, width)

        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)

        return x


class CompDecoder(nn.Module):
    """
    Decoder z_comp into component image
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.spatial_broadcast = SpatialBroadcast()
        # Input will be (B, L+2, H, W)
        self.decoder = nn.Sequential(
            nn.Conv2d(arch.z_comp_dim + 2, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # 16x downsampled: (32, 4, 4)
            nn.Conv2d(32, 3, 1, 1),
        )

    def forward(self, z_comp):
        """
        :param z_comp: (B, L)
        :return: component image (B, 3, H, W)
        """
        h, w = arch.img_shape
        # (B, L) -> (B, L+2, H, W)
        z_comp = self.spatial_broadcast(z_comp, h + 8, w + 8)
        # -> (B, 3, H, W)
        comp = self.decoder(z_comp)
        comp = torch.sigmoid(comp)
        return comp


class CompDecoderStrong(nn.Module):

    def __init__(self):
        super(CompDecoderStrong, self).__init__()

        self.dec = nn.Sequential(
            nn.Conv2d(arch.z_comp_dim, 256, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 256 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 256),

            nn.Conv2d(256, 128 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(16, 128),

            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.CELU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(8, 64),

            nn.Conv2d(64, 16 * 4 * 4, 1),
            nn.PixelShuffle(4),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),

            nn.Conv2d(16, 16, 3, 1, 1),
            nn.CELU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 3, 3, 1, 1)

        )

    def forward(self, x):
        """

        :param x: (B, L)
        :return:
        """
        x = x.view(*x.size(), 1, 1)
        comp = torch.sigmoid(self.dec(x))
        return comp


class PredictComp(nn.Module):
    """
    Predict component latents given mask latent
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(
            nn.Linear(arch.z_mask_dim, arch.predict_comp_hidden_dim),
            nn.ELU(),
            nn.Linear(arch.predict_comp_hidden_dim, arch.predict_comp_hidden_dim),
            nn.ELU(),
            nn.Linear(arch.predict_comp_hidden_dim, arch.z_comp_dim * 2),
        )

    def forward(self, h):
        """
        :param h: (B, L) hidden state from rnn_mask
        :return:
            z_comp_loc: (B, D)
            z_comp_scale: (B, D)
        """
        x = self.mlp(h)
        z_comp_loc = x[:, :arch.z_comp_dim]
        z_comp_scale = F.softplus(x[:, arch.z_comp_dim:]) + 1e-4

        return z_comp_loc, z_comp_scale
