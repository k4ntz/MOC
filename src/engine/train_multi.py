import itertools
from .train import train
import numpy as np


def train_multi(cfg):
    experiment_sets = [
        {
            'aow': ('arch.area_object_weight', np.logspace(0, 2, 5), lambda v: f'{v:.1f}'),
            'cos': ('arch.cosine_sim', [True, False], lambda v: f'{v+0}'),
            'seed': ('seed', range(3), lambda v: f'{v}')
        },
    ]
    base_exp_name = cfg.exp_name
    for experiment_set in experiment_sets:
        configs = itertools.product(*[v[1] for v in experiment_set.values()])
        params = [v[0] for v in experiment_set.values()]
        params_short = [k for k in experiment_set.keys()]
        for conf in configs:
            conf = [c.item() if isinstance(c, np.float) else c for c in conf]
            cfg.merge_from_list([val for pair in zip(params, conf) for val in pair])
            conf_str = [config_line[2](c) for c, config_line in zip(conf, experiment_set.values())]
            cfg.merge_from_list(
                ['exp_name', base_exp_name + "_" + "_".join([val for pair in zip(params_short, conf_str) for val in pair])])
            train(cfg)
