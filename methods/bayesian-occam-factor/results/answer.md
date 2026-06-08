# The Bayesian Occam factor

## The problem it solves

Given a data set and several candidate models of differing complexity, rank the models. Best-fit (maximum-likelihood) ranking fails: a more flexible model can always fit at least as well, so it drives selection toward over-parameterised models that overfit. Penalised-likelihood criteria (AIC, BIC), VC capacity, MDL code-lengths, and cross-validation all add a complexity penalty from outside the inference. The Bayesian Occam factor shows the penalty is already present in probability theory — it is the width of the marginal-likelihood integral — so no ad-hoc complexity term is needed.

## The key idea

Bayes operates at two levels. Fitting: P(w|D,H) = P(D|w,H)P(w|H)/P(D|H). Comparison: P(H|D) ∝ P(D|H)P(H). The normalising constant of the first level,

  **P(D|H) = ∫ P(D|w,H) P(w|H) dw**   (the *evidence* / marginal likelihood),

is the data-dependent term of the second. With equal model priors, rank models by their evidence. As a normalised distribution over data space, the evidence cannot be large everywhere: a model that spreads its predictive probability over many possible datasets must assign less to each, so it is automatically penalised for the data it does *not* predict. That is Occam's razor, with no added term.

## The decomposition (Laplace approximation)

The integrand P(D|w,H)P(w|H) ∝ posterior, which peaks sharply at w_MP. Laplace's method approximates the integral by peak height × Gaussian accessible width/volume.

One parameter, with a roughly uniform prior of width σ_w (so P(w_MP|H) ≈ 1/σ_w), using σ_{w|D} for the posterior accessible width √(2π)A^{-1/2} rather than for the bare standard deviation A^{-1/2}:

  P(D|H) ≈ P(D|w_MP,H) · P(w_MP|H) · σ_{w|D} = **P(D|w_MP,H) · (σ_{w|D}/σ_w)**.
  └ evidence ┘   └ best-fit likelihood ┘   └─ Occam factor ─┘

The **Occam factor = σ_{w|D}/σ_w** is the posterior width over the prior width: the factor (< 1) by which the model's parameter space collapses when the data arrive, equivalently the inverse number of distinguishable sub-models, of which one survives. Its log is the information gained about the parameter. It penalises both having many wide-ranging parameters and needing fine tuning (tiny σ_{w|D}).

k parameters, Gaussian posterior with Hessian A = −∇∇ log P(w|D,H) (the same matrix that gives the error bars A⁻¹), via the multivariate Gaussian integral ∫ exp(−½Δwᵀ A Δw) dᵏw = (2π)^{k/2}|A|^{−1/2}:

  **P(D|H) ≈ P(D|w_MP,H) · P(w_MP|H) · (2π)^{k/2} |A|^{−1/2}**.

Equivalently the Occam factor is P(w_MP|H) det⁻¹ᐟ²(A/2π) = (posterior volume)/(prior volume). Exact for a linear model with Gaussian noise and a quadratic regulariser; asymptotically exact otherwise (central limit theorem). Bayesian model comparison is thus maximum-likelihood comparison times one extra factor that the fit already provides.

## Why marginalize, not maximize

Comparing models requires the *integral* of the likelihood over the parameters (which carries the width / Occam factor), not its peak (which only ever grows with complexity). At the level of the regulariser strength α and noise level β, one likewise marginalises rather than jointly maximising over (w, α, β): the joint likelihood has a skew peak away from the probability mass. The canonical case: for a Gaussian with unknown μ, maximum likelihood gives σ_N but marginalising μ out gives the σ_{N−1} correction. The bias correction comes from integration, not from the prior (which is flat).

## The interpolation instantiation

Likelihood P(D|w,β) = exp(−βE_n)/Z_n, E_n = ½Σ(y(x_m)−t_m)², Z_n = (2π/β)^{N/2}, β = 1/σ_ν². Prior P(w|α) = exp(−αE_w)/Z_w, E_w quadratic, Z_w = ∫dᵏw exp(−αE_w). Posterior ∝ exp(−M), M = αE_w + βE_n, Hessian A = αC + βB, minimiser w_MP = βA⁻¹B w_ML, and Z_M = exp(−M(w_MP))(2π)^{k/2}|A|^{−1/2}. Then

  **Evidence for α,β:  P(D|α,β,H) = Z_M(α,β) / (Z_w(α) Z_n(β))**,
  log P(D|α,β,H) = −αE_w^MP − βE_n^MP − ½log det A − log Z_w(α) − log Z_n(β) + (k/2)log 2π.

In the prior-whitened basis C = I, maximising over α, β with a locally flat density over log α and log β gives the conditions

  2αE_w^MP = γ,  2βE_n^MP = N − γ,  where  **γ = k − α Tr A⁻¹ = Σ_a λ_a/(λ_a+α)**

is the **effective number of well-measured parameters** (λ_a the eigenvalues of βB; each term ∈ (0,1), ≈1 where data dominate the prior, ≈0 where they don't), and w_MP,a = (λ_a/(λ_a+α)) w_ML,a. So the Bayesian noise estimate satisfies χ²_D = N − γ — not N (discrepancy principle) nor N − k — because each well-measured parameter fits ≈1 unit of noise. A trace-based fixed point α := γ/(2E_w), β := (N−γ)/(2E_n) replaces the determinant search.

Ranking whole models integrates once more: P(D|H) = ∫ P(D|α,β,H) P(α,β) dα dβ ≈ P(D|α̂,β̂,H) P(α̂,β̂|H) · 2π Δlog α Δlog β, with (Δlog α)² ≈ 2/γ, (Δlog β)² ≈ 2/(N−1). Plotting log evidence versus model complexity traces the "Occam hill": a steep misfit-driven climb (penalty scaling as N) and a gentle complexity-driven descent (the accumulated Occam factors), with the adequate model at the peak. The maximum-likelihood fit term alone climbs monotonically; only the Occam factor bends the curve over.

## Worked numerical check

```python
import numpy as np

def design_matrix(x, centres, width):
    # linear-in-parameters model y(x) = sum_h w_h phi_h(x), Gaussian RBFs
    return np.exp(-0.5 * ((x[:, None] - centres[None, :]) / width) ** 2)

def log_evidence(Phi, t, alpha, beta):
    """Exact evidence for a linear model: quadratic E_w = (1/2)||w||^2,
    Gaussian noise E_n = (1/2)||Phi w - t||^2.  Returns the evidence and its
    decomposition into fit log-likelihood at w_MP + log Occam factor, plus gamma."""
    N, k = Phi.shape
    A = alpha * np.eye(k) + beta * (Phi.T @ Phi)          # Hessian A = alpha C + beta B
    w_mp = beta * np.linalg.solve(A, Phi.T @ t)           # w_MP = beta A^-1 B w_ML
    E_w = 0.5 * w_mp @ w_mp
    E_n = 0.5 * np.sum((Phi @ w_mp - t) ** 2)
    M_mp = alpha * E_w + beta * E_n
    sign, logdetA = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("Hessian must be positive definite")

    log_Zm = -M_mp + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdetA
    log_Zw = 0.5 * k * np.log(2 * np.pi / alpha)
    log_Zn = 0.5 * N * np.log(2 * np.pi / beta)
    log_ev = log_Zm - log_Zw - log_Zn                 # == log[Z_M / (Z_w Z_n)]

    fit_loglik = -beta * E_n - 0.5 * N * np.log(2 * np.pi / beta)             # log P(D|w_MP,beta)
    log_occam = 0.5 * k * np.log(2 * np.pi) - 0.5 * logdetA - alpha * E_w - log_Zw
    assert np.isclose(log_ev, fit_loglik + log_occam)
    gamma = k - alpha * np.trace(np.linalg.inv(A))        # effective # of parameters
    return log_ev, fit_loglik, log_occam, gamma

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-1, 1, size=40))
    t = np.sin(3 * x) + 0.1 * rng.standard_normal(x.size)
    beta = 1.0 / 0.1 ** 2
    alpha = 1.0
    for k in [2, 4, 6, 8, 12, 16, 24, 32]:
        centres = np.linspace(-1, 1, k)
        Phi = design_matrix(x, centres, width=2.0 / k)
        le, fit, oc, g = log_evidence(Phi, t, alpha, beta)
        print(f"k={k:2d}  log_ev={le:8.2f}  fit={fit:8.2f}  log_occam={oc:8.2f}  gamma={g:5.1f}")
    # Inspect the fit term and accumulated Occam factors separately; their sum is
    # the evidence used to rank the candidate bases, with no ad-hoc penalty.
```
