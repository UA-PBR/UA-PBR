"""Default configuration for UA-PBR experiments"""

BASE_CONFIG = {
    'n_samples': 1000,
    'resolution': 32,
    'batch_size': 32,
    'ae_epochs': 20,
    'cnn_epochs': 20,
    'lambda_cost': 0.3,
    'n_mc_samples': 30,
    'n_bootstrap': 1000,
    'seed': 42,
    'device': 'cuda'  # Will be auto-detected
}

EXPERIMENT_CONFIGS = {
    'full_experiment': BASE_CONFIG,
    'quick_test': {**BASE_CONFIG, 'n_samples': 100, 'ae_epochs': 5, 'cnn_epochs': 5},
    'high_res': {**BASE_CONFIG, 'resolution': 64, 'batch_size': 16},
    'real_data': {**BASE_CONFIG, 'use_real_data': True}
}

def get_config(name='full_experiment'):
    """Get configuration by name"""
    return EXPERIMENT_CONFIGS.get(name, BASE_CONFIG)
