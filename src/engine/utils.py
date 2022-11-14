import os
import argparse
from argparse import ArgumentParser
from config import cfg


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
        '--resume_ckpt',
        help='Provide a checkpoint to restart training from',
        default=''
    )

    parser.add_argument(
        '--arch-type',
        help='architecture type',
        choices=['baseline', '+m', '+moc'],
        required=True
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
    if cfg.model.lower() in ['lrspace', "lrtcspace", "tclrspace"]:
        cfg.arch = cfg.lr_arch
    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')
    if args.arch_type is None or args.arch_type == 'baseline':
        cfg.model = 'TcSpace'
        cfg.arch.area_object_weight = 0.0
    elif args.arch_type == "m": #
        cfg.model = 'TcSpace'
        cfg.arch.area_object_weight = 0.0
    elif args.arch_type == "moc": #
        cfg.model = 'TcSpace'
        cfg.arch.area_object_weight = 10.0
    cfg.resume_ckpt = args.resume_ckpt
    arch_type = '' if args.arch_type == "baseline" else args.arch_type
    cfg.arch_type = args.arch_type

    if args.resume_ckpt == '':
        cfg.resume_ckpt = f"../trained_models/{cfg.exp_name}_space{arch_type}_seed{cfg.seed}.pth"
        print(f"Using checkpoint from {cfg.resume_ckpt}")

    import torch
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(cfg.seed)

    return cfg, args.task
