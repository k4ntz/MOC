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

def lr_print(v):
    s = f'{v}'
    split = s.split("e")
    try:
        return split[0][:3] + "e" + split[1]
    except:
        return s


def multi_train(base_cfg):
    #  'aow': ('arch.area_object_weight', np.logspace(0, 1.5, 3), lambda v: f'{v:.1f}')
    #         {
    #             'fi': ('arch.flow_input', [False], boolean_print),
    #             'seed': ('seed', range(3), identity_print)
    #         },

    # Kalman Filter is metric
    # As Input Metric -> SPACE-variant
    experiment_sets = [
        # Optimizer
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'opti': ('train.solver.fg.optim', ['RMSprop', 'Adam', 'AdamW'], identity_print),
        #     'fglr': ('train.solver.fg.lr', np.logspace(-5.5, -4.01, 4), lr_print),
        # },
        # # Dynamic
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'aow': ('arch.area_object_weight', np.insert(np.logspace(1, 3, 3), 1, 0.0), lambda v: f'{v:.1f}'),
        #     'mw': ('arch.motion_weight', np.insert(np.logspace(1, 5, 3), 1, 0.0), lambda v: f'{v:.1f}'),
        #     'mces': ('arch.motion_cooling_end_step', [1600, 2400, 800, 1], hundred_print)
        # },
        # # Cooling
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'aow': ('arch.area_object_weight', np.insert(np.logspace(1, 1, 1), 0, 0.0), lambda v: f'{v:.1f}'),
        #     'dyn': ('arch.dynamic_scheduling', [False], boolean_print),
        #     'mces': ('arch.motion_cooling_end_step', [800, 2400], hundred_print)
        # },
        # # Motion Kind
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'mk': ('arch.motion_kind', ['flow'], lambda v: v[:2]),
        # },
        # # No Variance
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'mvar': ('arch.use_variance', [False], boolean_print),
        # },
        # # Long Run
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'steps': ('train.max_steps', [20000], thousand_print),
        # },
        # # Agree Scheduling
        # {
        #     'seed': ('seed', range(3), identity_print),
        #     'agr': ('arch.agree_sim', [True, False], boolean_print),
        # },
        # # Steepness
        # {
        #     'seed': ('seed', range(3), identity_print),
        #     'ds': ('arch.dynamic_steepness', np.linspace(1.1, 3, 5), lambda v: f'{v:.1f}'),
        # },
        # # Direct vs Loss
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'mces': ('arch.motion_cooling_end_step', np.array([1, 1500]), identity_print),
        #     'mw': ('arch.motion_weight', np.array([0, 1]), identity_print),
        # },
        # # Cosine sim
        # {
        #     'seed': ('seed', range(1), identity_print),
        #     'cos': ('arch.cosine_sim', [False], boolean_print)

        # },
        # Some seeds
        # {
        #     'seed': ('seed', range(3), identity_print),
        #     'steps': ('train.max_steps', [3000], thousand_print),
        #     'mvar': ('arch.use_variance', [True, False], boolean_print),
        # },
        # {
        #     'seed': ('seed', range(3), identity_print),
        #     'steps': ('train.max_steps', [3000], thousand_print),
        #     'dyn': ('arch.dynamic_scheduling', [True, False], boolean_print),
        # },
        {
            'seed': ('seed', range(5), identity_print),
            'aow': ('arch.area_object_weight', np.insert(np.logspace(1, 1, 1), 0, 0.0), lambda v: f'{v:.1f}'),
        },
        # {
        #     'seed': ('seed', range(8, 9), identity_print),
        #     'aow': ('arch.area_object_weight', np.logspace(1, 1, 1), lambda v: f'{v:.1f}'),
        # },
    ]
    base_exp_name = base_cfg.exp_name
    total_experiments = sum(len(list(itertools.product(*[v[1] for v in experiment_set.values()])))
                            for experiment_set in experiment_sets)
    i = 0
    rtpt = RTPT(name_initials='TR', experiment_name='SPACE-Time', max_iterations=total_experiments)
    rtpt.start()
    for experiment_set in experiment_sets:
        configs = itertools.product(*[v[1] for v in experiment_set.values()])
        params = [v[0] for v in experiment_set.values()]
        params_short = [k for k in experiment_set.keys()]
        for conf in configs:
            cfg = base_cfg.clone()
            conf = [c.item() if isinstance(c, np.float) else c for c in conf]
            cfg.merge_from_list([val for pair in zip(params, conf) for val in pair])
            conf_str = [config_line[2](c) for c, config_line in zip(conf, experiment_set.values())]
            cfg.merge_from_list(
                ['exp_name', base_exp_name + "_" + "_".join([f'{para}{v}' for para, v in zip(params_short, conf_str)])])
            print("=========" * 10)
            print("==========", "Starting experiment with the following cfg:", "==========",
                  cfg.exp_name, f"[{i}/{total_experiments}]", "==========")
            print(cfg)
            print("=========" * 10)
            train(cfg, rtpt_active=False)
            i += 1
            rtpt.step()
