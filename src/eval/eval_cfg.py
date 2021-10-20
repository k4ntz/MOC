from yacs.config import CfgNode
eval_cfg = CfgNode({
    # Evaluation during training
    'train': {
        # What to evaluate
        'metrics': ['ap', 'mse', 'cluster'],
        # Number of samples for evaluation
        'num_samples': {
            'mse': 512,
            'ap': 512,
            'cluster': 512,
        },
        
        # For dataloader
        'batch_size': 64,
        'num_workers': 4,
    },
    'test': {
        # For dataloader
        'batch_size': 32,
        'num_workers': 4,
    }
    
})
