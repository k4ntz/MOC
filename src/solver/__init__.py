from torch.optim import Adam, RMSprop, AdamW

__all__ = ['get_optimizers']

def get_optimizers(cfg, model):
    fg_optimizer = get_optimizer(cfg.train.solver.fg.optim, cfg.train.solver.fg.lr, model.space.fg_module.parameters())
    bg_optimizer = get_optimizer(cfg.train.solver.bg.optim, cfg.train.solver.bg.lr, model.space.bg_module.parameters())
    return fg_optimizer, bg_optimizer
    
def get_optimizer(name, lr, param):
    optim_class = {
        'Adam': Adam,
        'AdamW': AdamW,
        'RMSprop': RMSprop
    }[name]
    
    return optim_class(param, lr=lr)

