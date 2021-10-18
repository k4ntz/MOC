from .space.space import Space
from .space.time_consistency import TcSpace

__all__ = ['get_model']

def get_model(cfg):
    """
    Also handles loading checkpoints, data parallel and so on
    :param cfg:
    :return:
    """
    model = None
    if cfg.model.lower() == 'tcspace':
        model = TcSpace()
    if cfg.model.lower() == 'space':
        model = Space()
    return model
