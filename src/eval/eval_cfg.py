from yacs.config import CfgNode
eval_cfg = CfgNode({
    # Evaluation during training
    'train': {
        # What to evaluate
        'metrics': ['ap', 'cluster'],
        # Number of samples for evaluation
        'num_samples': {
            'mse': 640,
            'ap': 640,
            'cluster': 640,
        },
        
        # For dataloader
        'batch_size': 32,
        'num_workers': 4,
    },
    'test': {
        # For dataloader
        'batch_size': 32,
        'num_workers': 4,
    }
    
})
