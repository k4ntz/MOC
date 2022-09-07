from yacs.config import CfgNode

lr_arch = CfgNode({
    # TR
    'adjacent_consistency_weight': 0.0,
    'pres_inconsistency_weight': 0.0,
    'area_pool_weight': 0.0,
    'area_object_weight': 10.0,
    'cosine_sim': True,
    'object_threshold': 0.5,
    'z_cos_match_weight': 5.0,
    'full_object_weight': 3000,
    'motion_input': True,
    'motion': True,
    'motion_kind': 'mode',
    'motion_direct_weight': 1.0,
    'motion_loss_weight_z_pres': 10.0,
    'motion_loss_weight_z_where': 100.0,
    'motion_loss_weight_alpha': 1.0,
    'motion_weight': 1.0,
    'motion_sigmoid_steepen': 10000.0,  # Unused
    'motion_cooling_end_step': 1500,
    'dynamic_scheduling': True,
    'agree_sim': True,
    'dynamic_steepness': 2.0,
    'use_variance': True,
    'motion_underestimating': 2.0,
    'acceptable_non_moving': 8,  # Unused
    'variance_steps': 20,
    'motion_requirement': 2.0,  # Unused
    # SPACE-config
    'img_shape': (64, 64),
    'T': 4,
    
    # Grid size. There will be G*G slots
    'G': 16,
    
    # Foreground configurations
    # ==== START ====
    # Foreground likelihood sigma
    'fg_sigma': 0.15,
    # Size of the glimpse
    'glimpse_size': 8,
    # Encoded image feature channels
    'img_enc_dim_fg': 64,
    # Latent dimensions
    'z_pres_dim': 1,
    'z_depth_dim': 1,
    # (h, w)
    'z_where_scale_dim': 2,
    # (x, y)
    'z_where_shift_dim': 2,
    'z_what_dim': 16,
    
    # z_pres prior
    'z_pres_start_step': 4000,
    'z_pres_end_step': 10000,
    'z_pres_start_value': 0.1,
    'z_pres_end_value': 0.01,
    
    # z_scale prior
    'z_scale_mean_start_step': 10000,
    'z_scale_mean_end_step': 20000,
    'z_scale_mean_start_value': -1.0,
    'z_scale_mean_end_value': -2.0,
    'z_scale_std_value': 0.1,
    
    # Temperature for gumbel-softmax
    'tau_start_step': 0,
    'tau_end_step': 10000,
    'tau_start_value': 2.5,
    'tau_end_value': 2.5,

    # Fix alpha for the first N steps
    'fix_alpha_steps': 0,
    'fix_alpha_value': 0.1,
    # ==== END ====
    
    
    # Background configurations
    # ==== START ====
    # Number of background components. If you set this to one, you should use a strong decoder instead.
    'K': 3,
    # Background likelihood sigma
    'bg_sigma': 0.15,
    # Image encoding dimension
    'img_enc_dim_bg': 32,
    # Latent dimensions
    'z_mask_dim': 16,
    'z_comp_dim': 16,
    
    # (H, W)
    'rnn_mask_hidden_dim': 32,
    # This should be same as above
    'rnn_mask_prior_hidden_dim': 32,
    # Hidden layer dim for the network that computes q(z_c|z_m, x)
    'predict_comp_hidden_dim': 32,
    # ==== END ====
})
