__all__ = ['get_vislogger']

from .space_vis import SpaceVis
def get_vislogger(cfg):
    return SpaceVis()
