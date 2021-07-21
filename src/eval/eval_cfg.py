from yacs.config import CfgNode
eval_cfg = CfgNode({
    # Evaluation during training
    'train': {
        # What to evaluate
        'metrics': ['mse', 'ap', 'cluster'],
        # Number of samples for evaluation
        'num_samples': {
            'mse': 240,
            'ap': 240,
            'cluster': 240,
        },
        
        # For dataloader
        'batch_size': 12,
        'num_workers': 4,
    },
    'test': {
        # For dataloader
        'batch_size': 12,
        'num_workers': 4,
    }
    
})
