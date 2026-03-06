# Theoretical Foundation of UA-PBR

## Overview

UA-PBR combines three key theoretical components:

1. **Physics-Informed Filtering** (Theorem 2.3)
2. **Bayesian Uncertainty Quantification** (Theorem 3.1)
3. **Decision-Theoretic Rejection** (Theorem 5.7)

## Theorem 2.3: Error Bound via PDE Residual

For any observed $u_{\text{obs}}$ with physics score $S_{\text{phy}}(u_{\text{obs}}) \le \tau$:

$$\|u_{\text{obs}} - u^*\|_{L^2} \le \frac{\tau}{\alpha} + \|\phi\|_{L^2} + \gamma_n$$

where $\alpha$ is the coercivity constant and $\gamma_n$ is the autoencoder approximation error.

## Theorem 3.1: ELBO Optimality

Maximizing the Evidence Lower Bound (ELBO) minimizes the KL divergence to the true posterior:

$$\mathcal{L}_{\text{ELBO}}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D}\mid \omega)] - \mathrm{KL}(q_\phi \parallel p(\omega))$$

## Theorem 5.7: Risk Bound

Under Lipschitz continuity ($L$) and clean data error ($\epsilon_0$), the UA-PBR risk is bounded by:

$$R_\lambda(q) \le \lambda + \epsilon_0 + L\delta$$

where $\delta = \tau_{\text{phy}}/\alpha$ is the reconstruction error bound.
