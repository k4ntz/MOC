import torch
import matplotlib

# matplotlib.use('agg')
from utils import spatial_transform, inverse_spatial_transform
from .color_box import boxes, gbox, rbox


def visualize(x, z_pres, z_where_scale, z_where_shift, rbox=rbox, gbox=gbox, num_obj=8 * 8):
    """
        x: (bs, 3, *img_shape)
        z_pres: (bs, 4, 4, 1)
        z_where_scale: (bs, 4, 4, 2)
        z_where_shift: (bs, 4, 4, 2)
    """
    B, _, *img_shape = x.size()
    bs = z_pres.size(0)
    # num_obj = 8 * 8
    z_pres = z_pres.view(-1, 1, 1, 1)
    # z_scale = z_where[:, :, :2].view(-1, 2)
    # z_shift = z_where[:, :, 2:].view(-1, 2)
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    bbox = inverse_spatial_transform(z_pres * gbox + (1 - z_pres) * rbox,
                                     torch.cat((z_scale, z_shift), dim=1),
                                     torch.Size([bs * num_obj, 3, *img_shape]))
    bbox = (bbox + torch.stack(num_obj * (x,), dim=1).view(-1, 3, *img_shape)).clamp(0.0, 1.0)
    return bbox


def add_bbox(x, score, z_where_scale, z_where_shift, rbox=rbox, gbox=gbox, num_obj=8 * 8):
    B, _, *img_shape = x.size()
    bs = score.size(0)
    score = score.view(-1, 1, 1, 1)
    z_scale = z_where_scale.view(-1, 2)
    z_shift = z_where_shift.view(-1, 2)
    bbox = inverse_spatial_transform(score * gbox + (1 - score) * rbox,
                                     torch.cat((z_scale, z_shift), dim=1),
                                     torch.Size([bs * num_obj, 3, *img_shape]))
    bbox = (bbox + x.repeat(1, 3, 1, 1).view(-1, 3, *img_shape)).clamp(0.0, 1.0)
    return bbox


def bbox_in_one(x, z_pres, z_where, gbox=gbox):
    z_where_scale = z_where[..., :2]
    z_where_shift = z_where[..., 2:]
    B, _, *img_shape = x.size()
    B, N, _ = z_pres.size()
    z_pres = z_pres.reshape(-1, 1, 1, 1)
    z_scale = z_where_scale.reshape(-1, 2)
    z_shift = z_where_shift.reshape(-1, 2)
    # argmax_cluster = argmax_cluster.view(-1, 1, 1, 1)
    # kbox = boxes[argmax_cluster.view(-1)]
    bbox = inverse_spatial_transform(z_pres * gbox,  # + (1 - z_pres) * rbox,
                                     torch.cat((z_scale, z_shift), dim=1),
                                     torch.Size([B * N, 3, *img_shape]))
    bbox = (bbox.reshape(B, N, 3, *img_shape).sum(dim=1).clamp(0.0, 1.0) + x).clamp(0.0, 1.0)
    return bbox


def colored_bbox_in_one_image(x, z_pres, z_where_scale, z_where_shift, gbox=gbox):
    B, _, *img_shape = x.size()
    B, N, _ = z_pres.size()
    z_pres = z_pres.view(-1, 1, 1, 1)
    z_scale = z_where_scale.reshape(-1, 2)
    z_shift = z_where_shift.reshape(-1, 2)
    # argmax_cluster = argmax_cluster.view(-1, 1, 1, 1)
    # kbox = boxes[argmax_cluster.view(-1)]
    import ipdb;
    ipdb.set_trace()
    bbox = inverse_spatial_transform(z_pres * gbox,  # + (1 - z_pres) * rbox,
                                     torch.cat((z_scale, z_shift), dim=1),
                                     torch.Size([B * N, 3, *img_shape]))

    bbox = bbox.view(B, N, 3, *img_shape).sum(dim=1).clamp(0.0, 1.0)
    bbox = (bbox + x).clamp(0.0, 1.0)
    return bbox


# Times 10 to prevent index out of bound.
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 10


def fill_image_with_scene(image, scene):
    """

    """
    image = image.to("cpu")
    B, _, *img_shape = 1, image.size()
    img_shape = 128, 128
    for label_n, z_where in scene.items():
        cbox = boxes[label_n]
        N = len(z_where)
        z_pres = torch.ones(N)
        z_where_scale = torch.tensor(z_where)[..., :2]
        z_where_shift = torch.tensor(z_where)[..., 2:]
        z_scale = z_where_scale.reshape(-1, 2)
        z_shift = z_where_shift.reshape(-1, 2)
        # argmax_cluster = argmax_cluster.view(-1, 1, 1, 1)
        # kbox = boxes[argmax_cluster.view(-1)]
        bbox = inverse_spatial_transform(z_pres.reshape(((len(z_pres), 1, 1, 1))) * cbox,  # + (1 - z_pres) * rbox,
                                         torch.cat((z_scale, z_shift), dim=1),
                                         torch.Size([B * N, 3, *img_shape]))
        image = (bbox.reshape(N, 3, *img_shape)
                 .sum(dim=0).clamp(0.0, 1.0) + image).clamp(0.0, 1.0)
    return image
