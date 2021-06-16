from yacs.config import CfgNode

# dirname = os.checkpointdir.basename(os.checkpointdir.dirname(os.checkpointdir.abspath(__file__)))

cfg = CfgNode({
    # exp name
    'exp_name': '', 

    ### space stuff ###
    'model': 'SPACE',
    # exp model name for space
    'space_model_name': 'atari_ball_joint_v1',
    # For Atari and using matching SPACE model
    'gamelist': [
        'Pong-v0',
        'Tennis-v0',
    ],

    # gym env name
    'env_name': '',

    # Resume training or not
    'resume': True,
    # If resume is true, then we load this checkpoint. If '', we load the last checkpoint
    'resume_ckpt': '',
    # Whether to use multiple GPUs
    'parallel': False,
    # Device ids to use
    'device_ids': [0, 1, 2, 3],
    'device': 'cuda:0',
    'logdir': '/dqn/logs/',

    # space stuff
    'checkpointdir': '../output/checkpoints/',

    'video_steps': 10,

    'use_space': True,
    'liveplot': False,
    'debug': False,


    # For engine.train
    'train': {
        'batch_size': 128,
        'gamma': 0.97,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'eps_decay': 100000,
        'learning_rate': 0.00025,

        'memory_size': 50000,
        'memory_min_size': 25000,

        'num_episodes': 1000,
        'max_steps': 1000000,

        'skip_frames': 1,

        'log_steps': 500,
        # save every is for episodes
        'save_every': 5,

        'black_background': True,  
        'dilation': True,
    },
})

