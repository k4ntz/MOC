from model import get_model
from eval import get_evaluator
from dataset import get_dataset, get_dataloader
from solver import get_optimizers
from utils import Checkpointer, MetricLogger
import os
import os.path as osp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from vis import get_vislogger
import time
import torch
from torch.nn.utils import clip_grad_norm_
from rtpt import RTPT
from tqdm import tqdm
import shutil


def train(cfg):
    print('Experiment name:', cfg.exp_name)
    print('Dataset:', cfg.dataset)
    print('Model name:', cfg.model)
    print('Resume:', cfg.resume)
    if cfg.resume:
        print('Checkpoint:', cfg.resume_ckpt if cfg.resume_ckpt else 'last checkpoint')
    print('Using device:', cfg.device)
    if 'cuda' in cfg.device:
        print('Using parallel:', cfg.parallel)
    if cfg.parallel:
        print('Device ids:', cfg.device_ids)

    print('Loading data')
    rtpt = RTPT(name_initials='TRo', experiment_name='Train TcSP', max_iterations=cfg.train.max_epochs)
    rtpt.start()
    dataset = get_dataset(cfg, 'train')
    trainloader = get_dataloader(cfg, 'train')
    if cfg.train.eval_on:
        valset = get_dataset(cfg, 'val')
        # valloader = get_dataloader(cfg, 'val')
        evaluator = get_evaluator(cfg)
    model = get_model(cfg)
    model = model.to(cfg.device)

    if len(cfg.gamelist) >= 10:
        print("Training on every game")
        suffix = 'all'
    elif len(cfg.gamelist) == 1:
        suffix = cfg.gamelist[0]
        print(f"Training on {suffix}")
    elif len(cfg.gamelist) == 2:
        suffix = cfg.gamelist[0] + "_" + cfg.gamelist[1]
        print(f"Training on {suffix}")
    else:
        print("Can't train")
        exit(1)
    checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt,
                                load_time_consistency=cfg.load_time_consistency, add_flow=cfg.add_flow)
    model.train()

    optimizer_fg, optimizer_bg = get_optimizers(cfg, model)

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        checkpoint = checkpointer.load_last(cfg.resume_ckpt, model, optimizer_fg, optimizer_bg, cfg.device)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step'] + 1
    if cfg.parallel:
        model = nn.DataParallel(model, device_ids=cfg.device_ids)

    log_path = os.path.join(cfg.logdir, cfg.exp_name)
    if os.path.exists(log_path) and len(log_path) > 15 and cfg.logdir and cfg.exp_name and str(cfg.seed):
        shutil.rmtree(log_path)

    writer = SummaryWriter(log_dir=log_path, flush_secs=30,
                           purge_step=global_step)
    vis_logger = get_vislogger(cfg)
    metric_logger = MetricLogger()
    never_evaluated = True
    print(f'Start training, Global Step: {global_step}, Start Epoch: {start_epoch} Max: {cfg.train.max_steps}')
    end_flag = False
    start_log = 2
    for epoch in range(start_epoch, cfg.train.max_epochs):
        pbar = tqdm(total=len(trainloader))
        if end_flag:
            break
        start = time.perf_counter()
        for i, data in enumerate(trainloader):
            end = time.perf_counter()
            data_time = end - start
            start = end
            model.train()
            vids = data.to(cfg.device)
            loss, log = model(vids, global_step)
            # In case of using DataParallel
            loss = loss.mean()

            optimizer_fg.zero_grad()
            optimizer_bg.zero_grad()
            loss.backward()
            if cfg.train.clip_norm:
                clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
            # print("Before Step", torch.cuda.memory_summary(device=4, abbreviated=False))

            optimizer_fg.step()

            # if cfg.train.stop_bg == -1 or global_step < cfg.train.stop_bg:
            optimizer_bg.step()
            # print("After Step", torch.cuda.memory_summary(device=4, abbreviated=False))

            end = time.perf_counter()
            batch_time = end - start

            metric_logger.update(data_time=data_time)
            metric_logger.update(batch_time=batch_time)
            metric_logger.update(loss=loss.item())

            if global_step % cfg.train.print_every == 0:
                start = time.perf_counter()
                log.update({
                    'loss': metric_logger['loss'].median,
                })
                vis_logger.train_vis(model, writer, log, global_step, 'train', cfg, dataset)
                end = time.perf_counter()
                print(f'Log duration: {end - start}')
                log_state(cfg, epoch, global_step, i, log, metric_logger, trainloader)

            if cfg.resume and global_step - checkpoint['global_step'] < 100 and global_step % start_log == 0:
                start_log = int(start_log * 1.6)
                log_state(cfg, epoch, global_step, i, log, metric_logger, trainloader)

            if global_step % cfg.train.save_every == 0:
                start = time.perf_counter()
                checkpointer.save_last(model, optimizer_fg, optimizer_bg, epoch, global_step)
                print('Saving checkpoint takes {:.4f}s.'.format(time.perf_counter() - start))

            if (global_step % cfg.train.eval_every == 0 or never_evaluated) and cfg.train.eval_on:
                never_evaluated = False
                print('Validating...')
                start = time.perf_counter()
                eval_checkpoint = [model, optimizer_fg, optimizer_bg, epoch, global_step]
                evaluator.train_eval(model, valset, valset.bb_path, writer, global_step, cfg.device, eval_checkpoint,
                                     checkpointer, cfg)
                print('Validation takes {:.4f}s.'.format(time.perf_counter() - start))
            pbar.update(1)
            start = time.perf_counter()
            global_step += 1
            if global_step > cfg.train.max_steps:
                end_flag = True
                break
        rtpt.step()


def log_state(cfg, epoch, global_step, i, log, metric_logger, trainloader):
    print()
    print(
        'exp: {}, epoch: {}, iter: {}/{}, global_step: {}, loss: {:.2f}, z_what_con: {:.2f},'
        ' z_pres_con: {:.3f}, z_what_loss_pool: {:.3f}, z_what_loss_objects: {:.3f}, batch time: '
        '{:.4f}s, data time: {:.4f}s'.format(
            cfg.exp_name, epoch + 1, i + 1, len(trainloader), global_step, metric_logger['loss'].median,
            torch.sum(log['z_what_loss']).item(), torch.sum(log['z_pres_loss']).item(),
            torch.sum(log['z_what_loss_pool']).item(),
            torch.sum(log['z_what_loss_objects']).item(),
            metric_logger['batch_time'].avg, metric_logger['data_time'].avg))
    print()
