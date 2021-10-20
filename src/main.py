from engine.utils import get_config
from engine.train import train
from engine.eval import eval
from engine.show import show
from engine.multi_train import multi_train


if __name__ == '__main__':

    task_dict = {
        'train': train,
        'multi_train': multi_train,
        'train_multi': multi_train,
        'eval': eval,
        'show': show,
    }
    cfg, task = get_config()
    assert task in task_dict
    task_dict[task](cfg)


