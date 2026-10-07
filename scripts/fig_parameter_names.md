# Parameter symbols in the tables

The tables print each parameter as `f(raw)`, where `raw` is the value stored in the bundle (`output/bundles/*.json`).

| Symbol | JSON key | Transform |
|---|---|---|
| $\eta_0$ | `alpha0` | raw |
| $\eta_1$ | `log_alpha1` | $e^{x}$ |
| $\mu_0, \mu_1, \mu_2$ | `mu0`, `mu1`, `mu2` | raw |
| $\sigma_0, \sigma_1, \sigma_2$ | `sigma0`, `sigma1`, `sigma2` | raw |
| $\alpha_1$ | `z1_log_std` | $\log(1+e^{x})$ |
| $\alpha_2$ | `z1_log_tail` | $e^{-x}$ |
| $\alpha_3$ | `z1_skew` | raw |
| $\lambda_0, \lambda_1$ | `beta_a0`, `beta_a1` | raw |
| $\sigma_a$ | `log_sigma_a_cond` | $e^{x}$ |
| $\gamma_1$ | `log_sigma_eps` | $e^{x}$ |
| $\gamma_2$ | `log_beta` | $e^{-x}$ |
| $\gamma_3$ | `alpha_eps` | raw |
| $\zeta$ | `theta` | raw |

In the linear Gaussian model (Table 1), $\sigma_{z_1}$ and $\sigma_e$ are read directly from `parameters_derived`.
