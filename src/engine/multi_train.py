import itertools
from .train import train
import numpy as np
from rtpt import RTPT


def boolean_print(v):
    return f'{v + 0}'


def identity_print(v):
    return f'{v}'


def thousand_print(v):
    return f'{v // 1000}k'

def hundred_print(v):
    return f'{v / 1000:.1f}k'


def multi_train(cfg):
    #  'aow': ('arch.area_object_weight', np.logspace(0, 1.5, 3), lambda v: f'{v:.1f}')
    #         {
    #             'fi': ('arch.flow_input', [False], boolean_print),
    #             'seed': ('seed', range(3), identity_print)
    #         },

    # experiment_sets = [
    #     # Seed Experiment
    #     {
    #         'seed': ('seed', range(5), identity_print),
    #     },
    #     # Motion Kind Experiment
    #     {
    #         'mk': ('arch.motion_kind', ['mode', 'flow', 'median'], lambda v: v[:2]),
    #         'mlzp': ('arch.motion_loss_weight', np.logspace(0, 1, 2), lambda v: f'{v:.1f}'),
    #         'mlzw': ('arch.motion_loss_weight_z_where', np.logspace(2, 3, 2), lambda v: f'{v:.1f}'),
    #         'mlal': ('arch.motion_loss_weight_alpha', np.logspace(0, 1, 2), lambda v: f'{v:.1f}'),
    #     },
    #     # Motion Cooling End Step
    #     {
    #         'mcoo': ('arch.motion_cooling_end_step', [500, 1000, 2000, 4000, 8000], hundred_print),
    #     },
    #     # Long Run
    #     {
    #         'steps': ('train.max_steps', [50000], thousand_print),
    #         'aow': ('arch.area_object_weight', np.insert(np.logspace(1, 1, 1), 0, 0.0), lambda v: f'{v:.1f}'),
    #     },
    #     # Area Object Weight Experiment
    #     {
    #         'seed': ('seed', range(3), identity_print),
    #         'aow': ('arch.area_object_weight', np.logspace(1, 3, 3), lambda v: f'{v:.1f}'),
    #         'fow': ('arch.full_object_weight', [1000, 5000, 20000], thousand_print),
    #     }
    # ]

    experiment_sets = [
        # Seed Experiment
        {
            'seed': ('seed', range(0, 3), identity_print),
            'aow': ('arch.area_object_weight', np.insert(np.logspace(1, 1, 1), 0, 0.0), lambda v: f'{v:.1f}'),
        },
    ]
    base_exp_name = cfg.exp_name
    total_experiments = sum(len(list(itertools.product(*[v[1] for v in experiment_set.values()])))
                            for experiment_set in experiment_sets)
    i = 0
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
            print("Starting experiment with the following cfg:",  "==========",
                  cfg.exp_name, f"[{i}/{total_experiments}]", "==========")
            train(cfg, rtpt_active=False)
            i += 1
            rtpt.step()
