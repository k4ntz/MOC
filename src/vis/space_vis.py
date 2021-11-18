from torch.utils.tensorboard import SummaryWriter
import imageio
import numpy as np
import torch

import matplotlib
from utils import spatial_transform
from .utils import bbox_in_one, colored_bbox_in_one_image
from attrdict import AttrDict
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader
from collections import Counter
from torchvision.utils import draw_bounding_boxes as draw_bb
from PIL import Image
import PIL
from eval import convert_to_boxes, read_boxes
import math

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def grid_mult_img(grid, imgs, target_shape, scaling=4):
    grid = grid.reshape(target_shape)
    to_G = imgs.shape[2] // grid.shape[2]
    grid = grid.repeat_interleave(to_G, dim=2).repeat_interleave(to_G, dim=3)
    vis_imgs = (imgs + 0.3) / 1.3
    grid = vis_imgs * (scaling * grid + 1) / 4  # Intended oversaturation for visualization
    return make_grid(grid, 4, normalize=False, pad_value=1)


def grid_z_where_vis(z_where, imgs, z_pres, scaling=4):
    vis_imgs = (imgs + 0.3) / 1.3
    boxes_batch = convert_to_boxes(z_where, z_pres.squeeze(), z_pres.squeeze(), with_conf=False)
    grid = torch.zeros_like(vis_imgs)[:, 0:1]
    for i, boxes in enumerate(boxes_batch):
        for box in boxes:
            y_min, y_max, x_min, x_max = [int(e * 128) for e in box]
            grid[i][0][y_min:y_max, x_min:x_max] = 1
    grid = vis_imgs * (scaling * grid + 1) / 4  # Indented oversaturation for visualization
    return make_grid(grid, 4, normalize=False, pad_value=1)

class SpaceVis:
    @torch.no_grad()
    def train_vis(self, model, writer: SummaryWriter, log, global_step, mode, cfg, dataset, num_batch=8):
        """
        """
        print("Visualizing...")
        writer.add_scalar(f'{mode}/z_what_delta', torch.sum(log['z_what_loss']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/z_what_loss_pool', torch.sum(log['z_what_loss_pool']).item(),
                          global_step=global_step)
        writer.add_scalar(f'{mode}/z_what_loss_objects', torch.sum(log['z_what_loss_objects']).item(),
                          global_step=global_step)
        writer.add_scalar(f'{mode}/z_pres_loss', torch.sum(log['z_pres_loss']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss', torch.sum(log['flow_loss']).item(), global_step=global_step)
        writer.add_scalar(f'{mode}/flow_loss_z_where', torch.sum(log['flow_loss_z_where']).item(),
                          global_step=global_step)
        writer.add_scalar(f'{mode}/objects_detected', torch.sum(log['objects_detected']).item(),
                          global_step=global_step)
        writer.add_scalar(f'{mode}/total_loss', log['loss'], global_step=global_step)
        # FYI: For visualization only use some images of each stack in the batch
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
                if isinstance(log[key], torch.Tensor) and log[key].ndim > 0:
                    log[key] = log[key][2:num_batch * 4:4]
        log_img = AttrDict(log)

        # (B, 3, H, W) TR: Changed to z_pres_prob, why sample?
        fg_box = bbox_in_one(
            log_img.fg, log_img.z_pres_prob, log_img.z_where
        )
        # (B, 1, 3, H, W)
        imgs = log_img.imgs[:, None]
        fg = log_img.fg[:, None]
        recon = log_img.y[:, None]
        fg_box = fg_box[:, None]
        bg = log_img.bg[:, None]
        # (B, K, 3, H, W)
        comps = log_img.comps
        # (B, K, 3, H, W)
        masks = log_img.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log_img.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.reshape(B * N, 3, H, W)

        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/0-separations', grid_image, global_step)

        grid_image = make_grid(log_img.imgs, 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/1-image', grid_image, global_step)

        grid_image = make_grid(log_img.y, 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/2-reconstruction_overall', grid_image, global_step)

        grid_image = make_grid(log_img.bg, 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/3-background', grid_image, global_step)

        B = log_img.motion_z_pres.shape[0]
        G = int(math.sqrt(log_img.motion_z_pres.shape[1]))
        motion_z_pres_shape = (B, 1, G, G)
        writer.add_image(f'{mode}/4-motion', grid_mult_img(log_img.motion_z_pres, log_img.imgs, motion_z_pres_shape),
                         global_step)

        count = log_img.z_pres.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        loss = log['loss'].mean()
        writer.add_scalar(f'{mode}/loss', loss, global_step=global_step)
        writer.add_scalar(f'{mode}/count', count, global_step=global_step)

        mse = (log_img.y - log_img.imgs) ** 2
        mse = mse.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        writer.add_scalar(f'{mode}/mse', mse, global_step=global_step)
        writer.add_scalar(f'{mode}/log_like', log_img['log_like'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/What_KL', log_img['kl_z_what'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Where_KL', log_img['kl_z_where'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Pres_KL', log_img['kl_z_pres'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Depth_KL', log_img['kl_z_depth'].mean(), global_step=global_step)
        writer.add_scalar(f'{mode}/Bg_KL', log_img['kl_bg'].mean(), global_step=global_step)

        z_pres_grid = grid_mult_img(log_img.z_pres_prob, log_img.imgs, motion_z_pres_shape)
        writer.add_image(f'{mode}/5-z_pres', z_pres_grid, global_step)

        z_pres_pure_grid = grid_mult_img(log_img.z_pres_prob_pure, log_img.imgs, motion_z_pres_shape, scaling=6)
        writer.add_image(f'{mode}/6-z_pres_pure', z_pres_pure_grid, global_step)

        writer.add_image(f'{mode}/7-z_where', grid_z_where_vis(log_img.z_where, log_img.imgs, log_img.motion_z_pres),
                         global_step)

        gg_z_pres = log_img.z_pres_prob_pure.reshape(log_img.motion_z_pres.shape) > 0.5
        writer.add_image(f'{mode}/8-z_where_pure', grid_z_where_vis(log_img.z_where_pure, log_img.imgs, gg_z_pres),
                                                                    global_step)

        alpha_map = make_grid(log_img.alpha_map, 4, normalize=False, pad_value=1)
        writer.add_image(f'{mode}/9-alpha_map', alpha_map, global_step)
        # bb_image = draw_image_bb(model, cfg, dataset, global_step, num_batch)
        # grid_image = make_grid(bb_image, 4, normalize=False, pad_value=1)


    @torch.no_grad()
    def show_vis(self, model, dataset, indices, path, device):
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
        data = next(iter(dataloader))
        data = data.to(device)
        loss, log = model(data, 100000000)
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
        log = AttrDict(log)
        # (B, 3, H, W)
        fg_box = bbox_in_one(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.view(B * N, 3, H, W)

        # (3, H, W)
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)

        # (H, W, 3)
        image = torch.clamp(grid_image, 0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        imageio.imwrite(path, image)

    @torch.no_grad()
    def show_bb(self, model, image, path, device):
        image = image.to(device)
        loss, log = model(image, 100000000)
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
        log = AttrDict(log)
        # (B, 3, H, W)
        fg_box = colored_bbox_in_one_image(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        plt.imshow(fg_box[0][0].permute(1, 2, 0))
        plt.show()


def draw_image_bb(model, cfg, dataset, global_step, num_batch):
    indices = np.random.choice(len(dataset), size=num_batch, replace=False)
    dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
    data, motion_z_pres, motion_z_where = next(iter(dataloader))
    data = data.to(cfg.device)
    motion_z_pres = motion_z_pres.to(cfg.device)
    motion_z_where = motion_z_where.to(cfg.device)
    loss, log = model(data, motion_z_pres, motion_z_where, global_step)
    bb_path = f"{cfg.dataset_roots.ATARI}/{cfg.gamelist[0]}/train/bb"
    rgb_folder_src = f"{cfg.dataset_roots.ATARI.replace('space_like', 'rgb')}/{cfg.gamelist[0]}/train"
    boxes_gt, boxes_gt_moving, _ = read_boxes(bb_path, indices=indices)
    boxes_pred = []
    z_where, z_pres_prob = log['z_where'][2:num_batch * 4:4], log['z_pres_prob'][2:num_batch * 4:4]
    z_where = z_where.detach().cpu()
    z_pres_prob = z_pres_prob.detach().cpu().squeeze()
    z_pres = z_pres_prob > 0.5
    boxes_batch = convert_to_boxes(z_where, z_pres, z_pres_prob, with_conf=True)
    boxes_pred.extend(boxes_batch)
    result = []
    for idx, pred, gt, gt_m in zip(indices, boxes_pred[2::4], boxes_gt[2::4], boxes_gt_moving[2::4]):
        pil_img = Image.open(f'{rgb_folder_src}/{idx:05}_2.png', ).convert('RGB')
        pil_img = pil_img.resize((128, 128), PIL.Image.BILINEAR)
        image = np.array(pil_img)
        torch_img = torch.from_numpy(image).permute(2, 1, 0)
        pred_tensor = torch.FloatTensor(pred) * 128
        pred_tensor = torch.index_select(pred_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        gt_tensor = torch.FloatTensor(gt) * 128
        gt_tensor = torch.index_select(gt_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        gt_m_tensor = torch.FloatTensor(gt_m) * 128
        gt_m_tensor = torch.index_select(gt_m_tensor, 1, torch.LongTensor([0, 2, 1, 3]))
        bb_img = draw_bb(torch_img, gt_tensor, colors=["red"] * len(gt_tensor))
        bb_img = draw_bb(bb_img, gt_m_tensor, colors=["blue"] * len(gt_m_tensor))
        bb_img = draw_bb(bb_img, pred_tensor, colors=["green"] * len(pred_tensor))
        bb_img = bb_img.permute(0, 2, 1)
        result.append(bb_img)
    return torch.stack(result)
