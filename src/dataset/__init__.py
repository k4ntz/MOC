from .atari import Atari
from .obj3d import Obj3D
from torch.utils.data import DataLoader
from .labels import label_list_pacman, label_list_carnival, label_list_pong, label_list_space_invaders,\
    label_list_tennis, label_list_boxing

__all__ = ['get_dataset', 'get_dataloader', 'get_label_list']


def get_dataset(cfg, mode):
    assert mode in ['train', 'val', 'test']
    if cfg.dataset == 'ATARI':
        return Atari(cfg, mode)
    elif cfg.dataset == 'OBJ3D_SMALL':
        return Obj3D(cfg.dataset_roots.OBJ3D_SMALL, mode)
    elif cfg.dataset == 'OBJ3D_LARGE':
        return Obj3D(cfg.dataset_roots.OBJ3D_LARGE, mode)


def get_dataloader(cfg, mode):
    assert mode in ['train', 'val', 'test']

    batch_size = getattr(cfg, mode).batch_size
    shuffle = True if mode == 'train' else False
    num_workers = getattr(cfg, mode).num_workers
    dataset = get_dataset(cfg, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


def get_label_list(cfg):
    game = cfg.gamelist[0]
    if "MsPacman" in game:
        return label_list_pacman
    elif "Carnival" in game:
        return label_list_carnival
    elif "SpaceInvaders" in game:
        return label_list_space_invaders
    elif "Pong" in game:
        return label_list_pong
    elif "Boxing" in game:
        return label_list_boxing
    elif "Tennis" in game:
        return label_list_tennis
    else:
        raise ValueError(f"get_label_list failed for game {game}")
