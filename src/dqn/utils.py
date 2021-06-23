import os
import argparse
import torch
import math

from argparse import ArgumentParser
from dqn.dqn_config import cfg
from config import cfg as space_cfg
from eval.ap import convert_to_boxes


def get_config():
    parser = ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )

    parser.add_argument(
        '--space-config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to SPACE config file'
    )

    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    if args.space_config_file:
        space_cfg.merge_from_file(args.space_config_file)

    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')

    # Use config file name as the default experiment name
    if space_cfg.exp_name == '':
        if args.space_config_file:
            space_cfg.exp_name = os.path.splitext(os.path.basename(args.space_config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')

    # Seed
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return cfg, space_cfg


# helper function to process a single frame of z stuff
# it depends whether tp use nn with just fcl or cnn in nn 
def process_z_stuff(z_where, z_pres_prob, z_what, cfg, i_episode, logger=None):
    device = cfg.device
    boxes_len = int(math.sqrt(z_pres_prob.shape[0]))
    z_stuff = None
    indices = None
    # move z stuff to devices
    z_where = z_where.to(device)
    z_pres_prob = z_pres_prob.to(device)
    z_what = z_what.to(device)
    # clean up
    torch.cuda.empty_cache()
    # create z pres < 0.5
    z_pres = (z_pres_prob.detach().cpu().squeeze() > 0.5)
    # get coordinates
    coord_x = torch.FloatTensor([i % boxes_len for i, x in enumerate(z_pres) if x]).to(device)
    coord_y = torch.FloatTensor([math.floor(i / boxes_len) for i, x in enumerate(z_pres) if x]).to(device)
    # calc dimensions
    n_features = 4
    if cfg.train.use_zwhat:
        n_features = 36
    # for video stuff
    z_temp = z_where[z_pres]
    if cfg.train.cnn_features:
        # use cnn for feature extraction so prepare as image of z stuff
        z_pres_u = z_pres.unsqueeze(0)
        z_stuff = torch.zeros_like(torch.rand((1, boxes_len * boxes_len, n_features)), device=device)
        # cat z where and z what
        z_where_what = torch.cat((z_where.unsqueeze(0), z_what.unsqueeze(0)), 2).to(device)
        # get z stuff with z pres
        z_stuff[z_pres_u] = z_where_what[z_pres_u]
        z_stuff = z_stuff.squeeze(0)
        if cfg.train.reshape_input: 
            # reshape in useful structure
            z_stuff = z_stuff.permute(1,0)
            z_stuff = torch.reshape(z_stuff, (z_stuff.shape[0], boxes_len, boxes_len)).unsqueeze(3)
        # fill video with boxes
        indices = []
        for i, z_obj in enumerate(z_temp):
            indices.append(3)
    else:
    ################################################################################################
        # use linear network and use spare representation with coordinates
        # normalize z where centers to [0:1], add coordinates to its center values and normalize again
        z_temp[:, 2] = (((z_temp[:, 2] + 1.0) / 2.0) + coord_x) / boxes_len
        z_temp[:, 3] = (((z_temp[:, 3] + 1.0) / 2.0) + coord_y) / boxes_len
        # define what is player, ball and enemy
        indices = []
        n_objects = 3
        if not cfg.train.use_enemy:
            n_objects = 2
        if cfg.train.use_zwhat:
            n_features = 36
            # get z whats with z pres
            z_what_pres = z_what[z_pres]
            z_temp = torch.cat((z_temp, z_what_pres), 1).to(device)
        z_stuff = torch.zeros_like(torch.rand((n_objects, n_features)), device=device)
        for i, z_obj in enumerate(z_temp):
            x_pos = z_obj[2]
            y_pos = z_obj[3]
            size_relation = z_obj[0]/z_obj[1]
            # if in slot of right paddle
            if x_pos < 0.9315 and x_pos > 0.9305 and (size_relation < 0.9 or (y_pos < 0.21 or y_pos > 0.86)):
                # put right paddle at first
                z_stuff[0] = z_obj
                indices.append(0)
            # if its in slot of left paddle
            elif x_pos < 0.0702 and x_pos > 0.0687 and (size_relation < 0.9 or (y_pos < 0.21 or y_pos > 0.86)):
                # put left paddle at last
                if cfg.train.use_enemy:
                    z_stuff[2] = z_obj
                    indices.append(2)
                else:
                    indices.append(3)
            # if it is no paddle and has roughly size relation of ball
            elif size_relation > 0.7:
                # put ball in the middle
                z_stuff[1] = z_obj
                indices.append(1)
            else:
                # append black cause 4th box or sth like that
                indices.append(3)
        z_stuff = z_stuff.unsqueeze(0)
    ################################################################################################
    # log video with given classes
    if i_episode % cfg.video_steps == 0 and logger is not None:
        boxes_batch = convert_to_boxes(z_where.unsqueeze(0), z_pres.unsqueeze(0), z_pres_prob)
        logger.draw_bounding_box(boxes_batch, indices)
    return z_stuff


# use SPACE model
def get_z_stuff(image, space_model, cfg, i_episode, logger=None):
    # TODO: treat global_step in a more elegant way
    with torch.no_grad():
        loss, log = space_model(image, global_step=100000000)
        # (B, N, 4), (B, N, 1), (B, N, D)
        z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
        return process_z_stuff(z_where[0], z_pres_prob[0], z_what[0], cfg, i_episode, logger)
    return None