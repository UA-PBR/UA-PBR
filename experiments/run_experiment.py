"""Main experiment runner for UA-PBR"""
import torch
import numpy as np
from uapbr.data.dataset import RockDataset
from uapbr.models.physics_ae import PhysicsInformedAutoencoder
from uapbr.models.bayesian_cnn import BayesianCNN
from uapbr.models.baselines import StandardCNN
from uapbr.training.train_ae import train_physics_ae
from uapbr.training.train_bcnn import train_bayesian_cnn
from uapbr.training.train_baselines import train_standard_cnn
from uapbr.utils.metrics import EvaluationMetrics
from uapbr.utils.threshold_optimizer import ThresholdOptimizer
from uapbr.utils.visualization import plot_all_results

def run_uapbr_experiment(config):
    """Run complete UA-PBR experiment"""
    
    # Set seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    dataset = RockDataset(
        n_samples=config['n_samples'],
        resolution=config['resolution']
    )
    
    # Create loaders
    train_loader, val_loader, test_loader = dataset.get_data_loaders(
        batch_size=config['batch_size']
    )
    
    # Train models
    phy_ae = PhysicsInformedAutoencoder().to(device)
    phy_ae, ae_history = train_physics_ae(
        phy_ae, train_loader, val_loader, 
        epochs=config['ae_epochs']
    )
    
    bcnn = BayesianCNN().to(device)
    bcnn, cnn_history = train_bayesian_cnn(
        bcnn, train_loader, val_loader,
        epochs=config['cnn_epochs']
    )
    
    std_cnn = StandardCNN().to(device)
    std_cnn = train_standard_cnn(
        std_cnn, train_loader, val_loader,
        epochs=config['cnn_epochs']
    )
    
    # Optimize thresholds
    val_scores = collect_validation_scores(phy_ae, bcnn, val_loader, config)
    optimizer = ThresholdOptimizer(lambda_cost=config['lambda_cost'])
    best_tau_phy, best_tau_unc, _, _, _ = optimizer.optimize(*val_scores)
    
    # Test
    test_scores = collect_test_scores(phy_ae, bcnn, test_loader, config)
    metrics = EvaluationMetrics.rejection_quality(
        *test_scores, best_tau_phy, best_tau_unc, config['lambda_cost']
    )
    
    # Baselines
    std_preds = get_standard_cnn_predictions(std_cnn, test_loader)
    std_acc = (std_preds == test_scores[2]).mean()
    
    # Statistical tests
    risks = bootstrap_risk_analysis(
        test_scores, best_tau_phy, best_tau_unc, config
    )
    
    # Visualize
    plot_all_results(
        ae_history, test_scores, best_tau_phy, best_tau_unc,
        risks, val_scores, optimizer
    )
    
    return {
        'metrics': metrics,
        'std_acc': std_acc,
        'risks': risks,
        'thresholds': (best_tau_phy, best_tau_unc)
    }

if __name__ == "__main__":
    from uapbr.config.default_config import get_config
    config = get_config('full_experiment')
    results = run_uapbr_experiment(config)
    print("Experiment complete!")
