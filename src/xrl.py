# main file for all rl algos

from xrl import reinforce
from xrl import genetic_rl as genetic
from xrl import dreamer_v2
from xrl import utils

# otherwise genetic loading model doesnt work, torch bug?
from xrl.genetic_rl import policy_net, CNNPolicy

# function to call reinforce algorithm
def use_reinforce(cfg):
    print("Selected algorithm: REINFORCE")
    if cfg.mode == "train":
        reinforce.train(cfg)
    elif cfg.mode == "eval":
        reinforce.eval(cfg)


# function to call deep neuroevolution algorithm
def use_genetic(cfg):
    print("Selected algorithm: Deep Neuroevolution")
    if cfg.mode == "train":
        genetic.train(cfg)
    elif cfg.mode == "eval":
        genetic.play_agent(cfg)


# function to call dreamerv2
def use_dreamerv2(cfg):
    print("Selected algorithm: DreamerV2")
    if cfg.mode == "train":
        dreamer_v2.train(cfg)
    elif cfg.mode == "eval":
        dreamer_v2.eval(cfg)


# main
if __name__ == '__main__':
    cfg = utils.get_config()
    # algo selection 
    # 1: REINFORCE
    # 2: Deep Neuroevolution
    # 3: DreamerV2
    if cfg.rl_algo == 1:
        use_reinforce(cfg)
    elif cfg.rl_algo == 2:
        use_genetic(cfg)
    elif cfg.rl_algo == 3:
        use_dreamerv2(cfg)
    else:
        print("Unknown algorithm selected")

    