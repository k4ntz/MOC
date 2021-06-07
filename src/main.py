from engine.utils import get_config
from engine.train import train
from engine.eval import eval
from engine.show import show
from dataset.gather import gather


if __name__ == '__main__':

    task_dict = {
        'train': train,
        'eval': eval,
        'show': show,
        'gather': gather
    }
    cfg, task = get_config()
    assert task in task_dict
    task_dict[task](cfg)


