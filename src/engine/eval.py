from model import get_model
from eval import get_evaluator
from dataset import get_dataset, get_dataloader
from utils import Checkpointer
import os
import os.path as osp
from torch import nn
from torch.utils.tensorboard import SummaryWriter


def eval(cfg):
    assert cfg.eval.checkpoint in ['best', 'last']
    assert cfg.eval.metric in ['ap_dot5', 'ap_avg']
    if True:
        cfg.device = 'cuda:0'
        cfg.device_ids = [0]

    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'see below')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)

    print('Loading data')
    testset = get_dataset(cfg, 'test')
    model = get_model(cfg)
    model = model.to(cfg.device)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
    evaluator = get_evaluator(cfg)
    model.eval()

    if cfg.resume_ckpt:
        checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, cfg.device)
    elif cfg.eval.checkpoint == 'last':
        checkpoint = checkpointer.load_last('', model, None, None, cfg.device)
    elif cfg.eval.checkpoint == 'best':
        checkpoint = checkpointer.load_best(cfg.eval.metric, model, None, None, cfg.device)
    if cfg.parallel:
        assert 'cpu' not in cfg.device
        model = nn.DataParallel(model, device_ids=cfg.device_ids)

    evaldir = osp.join(cfg.evaldir, cfg.exp_name)
    info = {
        'exp_name': cfg.exp_name + str(cfg.seed)
    }
    # evaluator.test_eval(model, testset, testset.bb_path, cfg.device, evaldir,
    #                     info, , cfg=cfg)
    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    global_step = 100000
    writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step)
    eval_checkpoint = [model, None, None, "last", global_step]
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, "eval", cfg.exp_name), max_num=cfg.train.max_ckpt,
                                load_time_consistency=cfg.load_time_consistency, add_flow=cfg.add_flow)
    evaluator.test_eval(model, testset, testset.bb_path, writer, global_step,
                        cfg.device, eval_checkpoint, checkpointer, cfg)
