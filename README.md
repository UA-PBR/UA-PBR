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

| Condition       | UA-PBR Risk    | Std CNN Risk     | Improvement |
|-----------      |----------------|--------------|-------------|
| Clean           |0.0310 ± 0.0021 | 0.0021 ± 0.0016 | — |
| Gaussian (0.9)  | 0.0393 ± 0.0042 | 0.5005 ± 0.0136 | 92.1% |
| Salt-Pepper (0.9) | 0.0598 ± 0.0157 | 0.5005 ± 0.0136 | 88.0% |
| Physics-Violating (0.9) | 0.0338 ± 0.0040 | 0.5005 ± 0.0137 | 93.2% |

**Acceptance Rate (Clean)**: 89.7% ± 0.7%
**Accuracy on Accepted**: 99.99%
**Statistical Significance**: p < 0.0001

## 📦 Installation

```bash
git clone https://github.com/UA-PBR/UA-PBR.git
cd UA-PBR
pip install -r requirements.txt
pip install -e .
