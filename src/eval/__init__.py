__all__ = ['get_evaluator']

from .space_eval import SpaceEval
from .ap import convert_to_boxes, read_boxes

def get_evaluator(cfg):
    if cfg.model == 'SPACE':
        return SpaceEval()
    
    return None


