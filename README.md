# UA-PBR: Uncertainty-Aware Classifier with Physics-Based Rejection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A unified framework combining physics-informed filtering with Bayesian uncertainty quantification for robust scientific machine learning.

## 📋 Overview

Deep learning classifiers deployed in scientific settings cannot distinguish between clean inputs and corrupted data that violates physical laws. UA-PBR solves this by:

1. **Physics-Based Filtering**: Physics-informed autoencoder detects inputs violating PDEs
2. **Bayesian Uncertainty**: Monte Carlo Dropout quantifies predictive entropy
3. **Decision-Theoretic Rejection**: Joint rejection rule minimizes expected risk

## 🔬 Key Results (10 seeds)

| Condition | UA-PBR Risk | Std CNN Risk | Improvement |
|-----------|-------------|--------------|-------------|
| Clean | 0.0310 ± 0.0021 | 0.0021 ± 0.0016 | — |
| Gaussian (0.9) | 0.0393 ± 0.0042 | 0.5005 ± 0.0136 | 92.1% |
| Salt-Pepper (0.9) | 0.0598 ± 0.0157 | 0.5005 ± 0.0136 | 88.0% |
| Physics-Violating (0.9) | 0.0338 ± 0.0040 | 0.5005 ± 0.0137 | 93.2% |

**Acceptance Rate (Clean)**: 89.7% ± 0.7%
**Accuracy on Accepted**: 99.99%
**Statistical Significance**: p < 0.0001

## 📦 Installation

```python
git clone https://github.com/UA-PBR/UA-PBR.git
cd UA-PBR
pip install -r requirements.txt
pip install -e .
```

🚀 Quick Start
```python
from uapbr import UA_PBR
from uapbr.data import load_darcy_dataset

# Load data
data = load_darcy_dataset(n_samples=1000)

# Initialize model
model = UA_PBR(
    latent_dim=256,
    dropout_rate=0.3,
    lambda_cost=0.3
)

# Train
model.fit(data.train_loader, data.val_loader)

# Evaluate
results = model.evaluate(data.test_loader)
print(results)
```
📊 Running Experiments
```python
# Production run (10 seeds)
python experiments/run_production_10seeds.py

# Ablation study
python experiments/run_ablation.py

# Single seed quick test
python experiments/run_experiment.py --seed 42
```
📄 Citation
```python
@article{mostafa2026uapbr,
  title={Uncertainty-Aware Classifier with Physics-Based Rejection: 
         A Unified Framework for Robust Scientific Machine Learning},
  author={Mostafa, Mohsen},
  journal={Under Review},
  year={2026}
}
```
