from engine.utils import get_config
from model import get_model
from utils import Checkpointer
import os.path as osp
import torch
from termcolor import colored
from torchviz import make_dot


GRAPH_FORM = False


cfg, task = get_config()

cfg.load_time_consistency = False

model = get_model(cfg)
model = model.to(cfg.device)
checkpointer = Checkpointer(osp.join(cfg.checkpointdir, cfg.exp_name), max_num=cfg.train.max_ckpt)
use_cpu = 'cpu' in cfg.device
if cfg.resume_ckpt:
    checkpoint = checkpointer.load(cfg.resume_ckpt, model, None, None, use_cpu=use_cpu)

if GRAPH_FORM:
    input = torch.randn([1, 5, 128, 128]).to(cfg.device)
    loss, log = model(input, global_step=100000000)
    z_where, z_pres_prob, z_what = log['z_where'], log['z_pres_prob'], log['z_what']
    dot = make_dot(z_what.mean(), params=dict(model.named_parameters()))
    dot.view() # will save a digraph.pdf
else:
    print(model)
# import ipdb; ipdb.set_trace()
