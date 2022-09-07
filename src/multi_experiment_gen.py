import itertools
experiment_sets = [
    {
        'aow': ('arch.area_object_weight', range(-1, 4)),
        'cos': ('arch.cosine_sim', [False])
    },
    # {
    #     'apw': ('arch.area_pool_weight', range(-4, 3)),
    #     'cos': ('arch.cosine_sim', [True, False])
    # },
]

for experiment_set in experiment_sets:
    configs = itertools.product(*[v[1] for v in experiment_set.values()])
    for conf in configs:
        exp_join = "_".join([f'{k}{cv}' for k, cv in zip(experiment_set, conf)])
        one_e = '1e'
        empty = ''
        para_join = " ".join([f'{v[0]} {empty if isinstance(cv, bool) else one_e}{cv}' for v, cv in zip(experiment_set.values(), conf)])
        print(f"cfg=\"exp_name mspacman_atari_{exp_join} {para_join}\"")
        print("exp_name=$(echo \"$cfg\" | cut -d \" \" -f 2)")
        print("../log_execution.sh \"Training TcSpace ${exp_name}\" python main.py --task train"
              " --config configs/atari_mspacman.yaml ${cfg}")
