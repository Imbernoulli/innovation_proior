# Benign Overfitting in Linear Regression

## Problem

Overparameterized models trained to interpolate noisy data — zero training error, label noise and all —
often still generalize. In the simplest setting where this can be studied exactly (linear regression,
squared loss, parameter dimension large enough that a perfect fit is forced), when is fitting the noise
exactly compatible with near-optimal prediction? Benign overfitting characterizes precisely when the
minimum-norm interpolant has near-optimal excess risk, in terms of the spectrum of the covariate
covariance Σ.

## Setup

Well-specified linear model: x ∈ ℍ (a separable Hilbert space), y = x^⊤θ\* + ε, E[ε | x] = 0, conditional
noise variance ≥ σ². Covariance Σ = E[xx^⊤] = VΛV^⊤ with eigenvalues λ_1 ≥ λ_2 ≥ ⋯, and x = VΛ^{1/2}z with
z having independent σ_x²-subgaussian, unit-variance coordinates. Sample (x_i, y_i)_{i≤n}; X is the map
θ ↦ (x_i^⊤θ)_i. The dimension is large enough (Σ has rank > n, generic data) that interpolation is forced.

**Minimum-norm interpolant.** Among all θ with Xθ = y, the least-norm one (ridgeless limit of ridge):
θ̂ = X^⊤(XX^⊤)^{-1} y  (= (X^⊤X)^† X^⊤ y), which satisfies Xθ̂ = y.

**Excess risk.** R(θ̂) = E_{x,y}[(y − x^⊤θ̂)² − (y − x^⊤θ\*)²] = (θ\* − θ̂)^⊤ Σ (θ\* − θ̂).

## Key idea

The excess risk is the Σ-weighted squared parameter error, so an error in a low-variance direction is
nearly free. Interpolation embeds the label noise into θ̂; if there are many low-variance directions, the
noise spreads thinly across them and each picks up a negligible, Σ-weighted share. Overparameterization is
therefore not the disease but the cure — *provided* the small-eigenvalue tail of Σ is large in number and
balanced in size. Two effective ranks make this exact: one locates the harmless reservoir, the other
grades how evenly the noise dilutes.

## Effective ranks

For eigenvalues λ_i = μ_i(Σ) with Σ_i λ_i < ∞:
  r_k(Σ) = (Σ_{i>k} λ_i) / λ_{k+1},   R_k(Σ) = (Σ_{i>k} λ_i)² / (Σ_{i>k} λ_i²).
They satisfy r_k ≥ 1 and r_k(Σ²) ≤ r_k(Σ) ≤ R_k(Σ) ≤ r_k²(Σ), with equality r_k = R_k when the tail
eigenvalues are equal (e.g. r_0(I_p) = R_0(I_p) = p). r_k counts the tail directions (sets where the Gram
matrix turns isotropic); R_k measures their balance (an inverse participation ratio).

## Main result (two-sided risk bound)

Fix σ_x; there exist constants b, c, c_1 > 1. Let k\* = min{k ≥ 0 : r_k(Σ) ≥ b n} (min ∅ = ∞), and take
δ < 1 with log(1/δ) < n/c.

- If k\* ≥ n/c_1, then E R(θ̂) ≥ σ²/c  (harmful overfitting).
- Otherwise, with probability ≥ 1 − δ,
    R(θ̂) ≤ c ‖θ\*‖² ‖Σ‖ · max{ √(r_0(Σ)/n), r_0(Σ)/n, √(log(1/δ)/n) }
            + c log(1/δ) σ_y² ( k\*/n + n/R_{k\*}(Σ) ),
  and  E R(θ̂) ≥ (σ²/c) ( k\*/n + n/R_{k\*}(Σ) ).
- Necessity of the scale condition: for all large n, all Σ, all t ≥ 0, there is θ\* with ‖θ\*‖ = t such that
  with probability ≥ 1/4, R(θ̂) ≥ (1/a_1) ‖θ\*‖² ‖Σ‖ · 1[ r_0(Σ)/(n log(1 + r_0(Σ))) ≥ a_2 ].

**Benign sequence.** With ‖Σ‖ = 1 WLOG, the covariance condition called benign is
  r_0(Σ_n)/n → 0,  k\*_n/n → 0,  n/R_{k\*_n}(Σ_n) → 0.

## Proof (the derivation)

1. **Decomposition.** With y = Xθ\* + ε and P = X^⊤(XX^⊤)^{-1}X (projection onto X's row space),
   θ\* − θ̂ = (I − P)θ\* − X^⊤(XX^⊤)^{-1}ε, so
   R(θ̂) ≤ 2 θ\*^⊤Bθ\* + 2 ε^⊤Cε,  E_ε R(θ̂) ≥ θ\*^⊤Bθ\* + σ² tr(C),
   B = (I − P)Σ(I − P),  C = (XX^⊤)^{-1}XΣX^⊤(XX^⊤)^{-1}.
   (For the high-probability upper bound, ε^⊤Cε ≤ (4t+2)σ_y²tr(C) via a subgaussian quadratic-form bound,
   using ‖C‖ ≤ tr(C) and tr(C²) ≤ tr(C)².)

2. **Bias term.** Since (I − P)X^⊤ = 0, θ\*^⊤Bθ\* = θ\*^⊤(I − P)(Σ − (1/n)X^⊤X)(I − P)θ\* ≤
   ‖Σ − (1/n)X^⊤X‖ ‖θ\*‖². Sample-covariance concentration (effective rank r(Σ) ≤ r_0(Σ)) gives the
   max{√(r_0/n), r_0/n, √(t/n)} bound. Small when r_0(Σ) ≪ n.

3. **Variance term — eigenbasis.** With z_i = Xv_i/√λ_i independent isotropic subgaussian,
   A := XX^⊤ = Σ_i λ_i z_i z_i^⊤, XΣX^⊤ = Σ_i λ_i² z_i z_i^⊤, so tr(C) = Σ_i λ_i² z_i^⊤A^{-2}z_i.
   Sherman–Morrison (A = λ_i z_i z_i^⊤ + A_{−i}):
   λ_i² z_i^⊤A^{-2}z_i = λ_i² z_i^⊤A_{−i}^{-2}z_i / (1 + λ_i z_i^⊤A_{−i}^{-1}z_i)², with z_i ⟂ A_{−i}.

4. **Tail isotropy.** For A_k = Σ_{i>k}λ_i z_i z_i^⊤ (ε-net + Bernstein):
   (1/c)Σ_{i>k}λ_i − cλ_{k+1}n ≤ μ_n(A_k) ≤ μ_1(A_k) ≤ c(Σ_{i>k}λ_i + λ_{k+1}n).
   If r_k(Σ) ≥ bn, the mass Σ_{i>k}λ_i = λ_{k+1}r_k(Σ) dominates λ_{k+1}n, so all n eigenvalues of A_k lie
   within a constant of λ_{k+1}r_k(Σ): the tail Gram matrix is isotropic.

5. **Upper bound on tr(C).** Split at l ≤ k with r_k ≥ bn. The i ≤ l terms each ≲ ‖z_i‖²/‖Πz_i‖⁴ ≈ 1/n
   (so ≲ l/n); the i > l terms ≲ nΣ_{i>l}λ_i²/(λ_{k+1}r_k(Σ))². Hence
   tr(C) ≲ l/n + nΣ_{i>l}λ_i²/(λ_{k+1}r_k(Σ))². Taking k = k\* and minimizing over l, the minimizer is l = k\*, and since
   λ_{k\*+1}r_{k\*}(Σ) = Σ_{i>k\*}λ_i, the tail term = nΣ_{i>k\*}λ_i²/(Σ_{i>k\*}λ_i)² = n/R_{k\*}(Σ). So
   tr(C) ≲ k\*/n + n/R_{k\*}(Σ).

6. **Lower bound on tr(C).** A per-term lower bound
   λ_i²z_i^⊤A_{−i}^{-2}z_i/(1+λ_i z_i^⊤A_{−i}^{-1}z_i)² ≥ (1/cn)(1 + (Σ_{j>k}λ_j + nλ_{k+1})/(nλ_i))^{-2},
   plus a sum-of-positives lemma, gives tr(C) ≳ (1/cb²)min_{l≤k}(l/n + b²nΣ_{i>l}λ_i²/(λ_{k+1}r_k)²) when
   r_k ≥ bn; taking k = k\* and using the best-k step gives const·(k\*/n + n/R_{k\*}). When r_k < bn,
   tr(C) ≳ (k+1)/(cb²n) —
   so tr(C) = Ω(1) when no k ≤ n/c qualifies (k\* ≥ n/c).

7. **Necessity of r_0.** A packing/fat-shattering reduction (least-norm interpolation → learning a quantized
   weight vector; pack well-separated θ\* in the ρ(u,v) = √((u−v)^⊤Σ(u−v)) metric) forces
   r_0(Σ)/(n log(1+r_0)) → 0.

## Eigenvalue examples

- **Infinite dimension, fixed Σ.** λ_k = k^{-α} ln^{-β}(k+1) is benign iff α = 1 and β > 1 — a razor's
  edge: just barely summable, decaying no faster. Benign overfitting is fragile in infinite dimensions.
- **Large finite dimension + isotropic floor (the generic case).** λ_k = γ_k + ε_n for k ≤ p_n (else 0),
  γ_k = Θ(e^{-k/τ}). Benign iff p_n = ω(n) and n e^{-o(n)} = ε_n p_n = o(n). The floor builds a flat
  reservoir of ≈ p_n harmless directions; the decay of the real features γ_k is then irrelevant. For
  p_n = Ω(n), ε_n p_n = n e^{-o(n)}: R(θ̂) = O( (ε_n p_n + 1)/n + ln(n/(ε_n p_n))/n + max{1/n, n/p_n} ).

## A tiny spectral example (numpy)

```python
import numpy as np

def min_norm_interpolant(X, y):
    # theta_hat = X^T (X X^T)^{-1} y  — least-norm interpolant (ridgeless limit)
    return X.T @ np.linalg.solve(X @ X.T, y)

def excess_risk(theta_hat, theta_star, eigs):
    d = theta_star - theta_hat
    return float(d @ (eigs * d))                       # (theta*-theta_hat)^T Sigma (.)

def effective_ranks(eigs, k):
    tail = eigs[k:]
    s, s2 = tail.sum(), (tail**2).sum()
    r_k = s / eigs[k]                                   # r_k = sum_{i>k} lam_i / lam_{k+1}
    R_k = s**2 / s2                                     # R_k = (sum)^2 / sum-of-squares
    return r_k, R_k

def predicted_variance_term(eigs, n, b=2.0):
    # k* = min{k : r_k >= b n}, variance penalty ~ k*/n + n/R_{k*}
    for k in range(len(eigs) - 1):
        r_k, R_k = effective_ranks(eigs, k)
        if r_k >= b * n:
            return k / n + n / R_k
    return 1.0                                          # no reservoir -> harmful, Omega(1)

def experiment(eigs, n, theta_star, noise=0.5, trials=200, seed=0):
    rng = np.random.default_rng(seed)
    p = len(eigs)
    risks = []
    for _ in range(trials):
        Z = rng.standard_normal((n, p))
        X = Z * np.sqrt(eigs)[None, :]                  # x_i = Lambda^{1/2} z_i
        y = X @ theta_star + noise * rng.standard_normal(n)
        risks.append(excess_risk(min_norm_interpolant(X, y), theta_star, eigs))
    return np.mean(risks)

if __name__ == "__main__":
    n = 50

    # Benign spectrum: one strong signal direction + a vast flat reservoir of p-1 tiny equal
    # directions (p >> n). r_1 >> b n at k*=1, R_{k*} ~ p-1 huge -> noise dilutes harmlessly.
    p_b = 4000
    benign = np.concatenate([[1.0], np.full(p_b - 1, 1.0 / (p_b - 1))])
    theta_b = np.zeros(p_b); theta_b[0] = 1.0

    # Harmful spectrum: isotropic with only p ~ n directions, a stable version of the p=n,
    # Sigma=I disaster. For the theorem-threshold constant used here, every r_k stays < b n,
    # so the noise has no large flat reservoir and the variance scale remains order one.
    p_h = 60
    harmful = np.ones(p_h)
    theta_h = np.zeros(p_h); theta_h[0] = 1.0

    print("benign : empirical excess risk =", round(experiment(benign, n, theta_b), 4),
          " predicted variance scale =", round(predicted_variance_term(benign, n), 4))
    print("harmful: empirical excess risk =", round(experiment(harmful, n, theta_h), 4),
          " predicted variance scale =", round(predicted_variance_term(harmful, n), 4))
    # benign: small empirical risk, small predicted penalty (reservoir absorbs the noise);
    # harmful: order-1 empirical risk, predicted penalty ~ 1 (no flat reservoir, k* >= n/c).
```
