# Scaling Laws for Neural Language Models

## Problem

Replace intuition about "bigger is better" with a predictive quantitative theory: how does Transformer test loss depend on non-embedding parameter count $N$, dataset size $D$, compute $C$, steps $S$, and batch $B$ — accurately enough to extrapolate and to decide the compute-optimal allocation before training.

## Key idea

Loss is nearly independent of architecture *shape* at fixed $N$ (a few percent over wide depth/width/head ranges), so a single scale $N$ summarizes a model. Define it cleanly as the non-embedding count $N \approx 12\, n_{\text{layer}} d_{\text{model}}^2$, with training compute $C \approx 6ND$ FLOPs.

**Single-factor power laws** (each holds when the other factors are abundant; fit by log-log regression):
$$L(N) = (N_c/N)^{\alpha_N},\ \alpha_N\approx0.076; \quad L(D) = (D_c/D)^{\alpha_D},\ \alpha_D\approx0.095; \quad L(C_{\min}) = (C_c^{\min}/C_{\min})^{\alpha_C^{\min}},\ \alpha_C^{\min}\approx0.05.$$

**Joint law**, with form forced by three principles — vocabulary only rescales loss (so $N_c,D_c$ are non-fundamental); the limits $N\to\infty$, $D\to\infty$ must recover $L(D)$, $L(N)$; and overfitting must admit a clean integer $1/D$ expansion:
$$L(N,D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D},\quad \alpha_N=0.076,\ \alpha_D=0.103,\ N_c=6.4\times10^{13},\ D_c=1.8\times10^{13}.$$
Overfitting is governed by $N^{\alpha_N/\alpha_D}/D$, so to stay data-constrained data grows sublinearly: $D\propto N^{0.74}$.

**Training-time law** (infinite data, additive — a capacity floor plus an optimization gap):
$$L(N,S_{\min}) = (N_c/N)^{\alpha_N} + (S_c/S_{\min})^{\alpha_S},\quad \alpha_S\approx0.76.$$
Batch effects are standardized through the critical batch size $B_{\text{crit}}(L)=B_*/L^{1/\alpha_B}$ ($\alpha_B\approx0.21$), with $S_{\min}=S/(1+B_{\text{crit}}/B)$ and $C_{\min}=C/(1+B/B_{\text{crit}})$.

**Compute-optimal allocation.** Minimizing $L(N,S_{\min})$ at fixed $C_{\min}=6NB\,S_{\min}$:
$$\alpha_C^{\min} = \frac{1}{1/\alpha_N + 1/\alpha_S + 1/\alpha_B}\approx0.05,\qquad N\propto C^{\alpha_C^{\min}/\alpha_N}\approx C^{0.71}\ (\text{empirically } C^{0.73}),$$
with $B\propto C^{0.24}$, $S\propto C^{0.03}$ (nearly flat), and $D=BS\propto C^{0.27}$. Extra compute should go predominantly into a **bigger model**; data and serial steps grow slowly. (The $L(C_{\min})$ and $L(D)$ extrapolations eventually cross, marking where the simple picture must break down.)

## Code

```python
import numpy as np


def transformer_param_count(n_layer, d_model, d_ff=None, d_attn=None):
    d_ff = 4 * d_model if d_ff is None else d_ff
    d_attn = d_model if d_attn is None else d_attn
    return 2 * d_model * n_layer * (2 * d_attn + d_ff)      # ~ 12 n_layer d_model^2


def forward_flops_per_token(N, n_layer, d_model, n_ctx):
    return 2 * N + 2 * n_layer * n_ctx * d_model            # training ~ 3x -> 6N


def fit_power_law(X, L):
    # L = (X_c / X) ** alpha
    slope, intercept = np.polyfit(np.log(X), np.log(L), 1)
    alpha = -slope
    X_c = np.exp(intercept / alpha)
    return X_c, alpha


def joint_loss(N, D, params):
    # L(N,D) = [ (N_c/N)^(alpha_N/alpha_D) + D_c/D ]^alpha_D
    alpha_N, alpha_D, N_c, D_c = params
    return ((N_c / N) ** (alpha_N / alpha_D) + D_c / D) ** alpha_D


def fit_joint_loss(runs):
    from scipy.optimize import curve_fit
    N, D, L = runs[:, 0], runs[:, 1], runs[:, 2]

    def model(ND, alpha_N, alpha_D, log_Nc, log_Dc):
        Nv, Dv = ND
        return np.log(joint_loss(Nv, Dv, (alpha_N, alpha_D, np.exp(log_Nc), np.exp(log_Dc))))

    p0 = (0.076, 0.103, np.log(6.4e13), np.log(1.8e13))
    popt, _ = curve_fit(model, (N, D), np.log(L), p0=p0, maxfev=100000)
    a_N, a_D, lNc, lDc = popt
    return a_N, a_D, np.exp(lNc), np.exp(lDc)


def compute_optimal_exponents(alpha_N, alpha_S, alpha_B):
    alpha_C = 1.0 / (1.0 / alpha_N + 1.0 / alpha_S + 1.0 / alpha_B)
    return {
        "alpha_C_min": alpha_C,                    # ~0.05
        "N_exp": alpha_C / alpha_N,                 # ~0.71
        "B_exp": alpha_C / alpha_B,                 # ~0.24
        "S_exp": alpha_C / alpha_S,                 # ~0.03
        "D_exp": alpha_C / alpha_B + alpha_C / alpha_S,  # ~0.27
    }
```
