from .space.space import Space
from .space.time_consistency import TcSpace
from .low_res_space.time_consistency import LrTcSpace

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
    if cfg.model.lower() in ['lrspace', "lrtcspace", "tclrspace"]:
        model = LrTcSpace()
    if cfg.model.lower() == 'space':
        model = Space()
    return model
