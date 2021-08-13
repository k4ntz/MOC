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
    if cfg.model == 'SPACE':
        model = Space()
    elif cfg.model == 'TcSPACE':
        model = TcSpace()

    return model
