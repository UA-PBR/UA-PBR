# UA-PBR: Uncertainty-Aware Classifier with Physics-Based Rejection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-2401.12345-b31b1b.svg)](https://arxiv.org/abs/2401.12345)

Official implementation of **UA-PBR: Uncertainty-Aware Classifier with Physics-Based Rejection** - a novel framework combining physics-informed learning with Bayesian deep learning for safe prediction with reject option.

## 📋 Overview

UA-PBR integrates three key components:
- **Physics-Informed Autoencoder** - Learns physical manifold via PDE residuals
- **Bayesian CNN** - Quantifies epistemic uncertainty via MC Dropout  
- **Decision-Theoretic Reject Option** - Optimal threshold selection

**Key Theorems Implemented:**
- Theorem 2.1: Physics Autoencoder Convergence
- Theorem 2.3: Error Bound via PDE Residual
- Theorem 3.1: ELBO Optimality (Bayesian CNN)
- Theorem 4.3: Risk Bound for Joint Rejection
- Theorem 5.1: Optimal Threshold Selection
- Theorem 6.1/6.2: Statistical Significance Tests

## 🚀 Quick Start

### Installation
```python
git clone https://github.com/yourusername/UA-PBR.git
cd UA-PBR
pip install -r requirements.txt
```
Run Complete Experiment
```python
from uapbr.config.default_config import get_config
from experiments.run_experiment import run_uapbr_experiment

config = get_config('full_experiment')
results = run_uapbr_experiment(config)
```
📊 Results
Method	         Accuracy	  Risk (λ=0.3)	  p-value
Standard CNN	   49.0%	    0.5126        	-
MC Dropout Only	 46.4%	    -	              -
UA-PBR (Ours)	   48.5%	    0.3355	        <0.0001

📚 Citation
```python
@article{yourname2024uapbr,
  title={UA-PBR: Uncertainty-Aware Classifier with Physics-Based Rejection},
  author={Your Name},
  journal={Journal of Computational Physics},
  year={2024}
}
```



