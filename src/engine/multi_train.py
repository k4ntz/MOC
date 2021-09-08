import itertools
from .train import train
import numpy as np
from rtpt import RTPT

def boolean_print(v):
    return f'{v + 0}'


def identity_print(v):
    return f'{v}'


def multi_train(cfg):
    #  'aow': ('arch.area_object_weight', np.logspace(0, 1.5, 3), lambda v: f'{v:.1f}')
    #         {
    #             'fi': ('arch.flow_input', [False], boolean_print),
    #             'seed': ('seed', range(3), identity_print)
    #         },

    experiment_sets = [
        {
            'fce': ('arch.flow_cooling_end_step', [5000], lambda v: f'{v // 1000}k'),
            'flw': ('arch.flow_loss_weight', np.logspace(0, 1, 2), lambda v: f'{v:.1f}'),
        },
    ]
    base_exp_name = cfg.exp_name
    total_experiments = sum(len(list(itertools.product(*[v[1] for v in experiment_set.values()])))
                            for experiment_set in experiment_sets)
    rtpt = RTPT(name_initials='TRo', experiment_name='SPACE-Time', max_iterations=total_experiments)
    rtpt.start()
    for experiment_set in experiment_sets:
        configs = itertools.product(*[v[1] for v in experiment_set.values()])
        params = [v[0] for v in experiment_set.values()]
        params_short = [k for k in experiment_set.keys()]
        for conf in configs:
            conf = [c.item() if isinstance(c, np.float) else c for c in conf]
            cfg.merge_from_list([val for pair in zip(params, conf) for val in pair])
            conf_str = [config_line[2](c) for c, config_line in zip(conf, experiment_set.values())]
            cfg.merge_from_list(
                ['exp_name', base_exp_name + "_" + "_".join([f'{para}{v}' for para, v in zip(params_short, conf_str)])])
            train(cfg, rtpt_active=False)
            rtpt.step()
