# Importance Sampling for Rare-Event Estimation

## Problem

Estimate a very small failure probability p_f = P(A) = E_p[1_A(X)] of a simulated system — a bit-error rate, word-error rate, or outage probability of 10⁻⁶ to 10⁻⁹ — when the failure set A is so rare that direct simulation almost never observes it.

The obstacle is the *relative* error of the crude estimator. Drawing X_i ∼ p and averaging the indicator, p̂ = (1/n)Σ 1_A(X_i), gives an unbiased estimate whose relative standard error is

    σ/μ = √((1 − p_f)/p_f) ≈ p_f^(−1/2).

Holding that to ~10% needs n ≈ 100/p_f trials — about 10¹¹ runs of the decoder for a 10⁻⁹ event. Adding samples only buys √n and cannot change the 1/p_f scaling, so the variance itself must be attacked.

## Key idea

Sample from a different distribution q that makes the failure event common, and reweight each sample by the **likelihood ratio** w = p/q to stay unbiased:

    p_f = ∫ 1_A(x) p(x) dx = ∫ 1_A(x) (p(x)/q(x)) q(x) dx = E_q[ 1_A(X) · p(X)/q(X) ].

The importance-sampling estimator is

    p̂_q = (1/n) Σ_i 1_A(X_i) w(X_i),   w = p/q,   X_i ∼ q,

valid for any q with q(x) > 0 wherever 1_A(x) p(x) ≠ 0. It is unbiased for every such q; q affects only the variance,

    Var_q(p̂_q) = σ_q²/n,   σ_q² = ∫ (1_A p − p_f q)² / q  dx.

**Optimal sampler.** Minimizing σ_q² (or by Cauchy–Schwarz) gives the zero-variance density

    q*(x) = 1_A(x) p(x) / p_f,

i.e. p restricted to A and renormalized. It is unusable — its normalizer is the unknown p_f — but it dictates the design: q should concentrate on A and be shaped like p there.

**Parametric tilt (mean translation).** For light-tailed (Gaussian) noise the way to reach that shape cheaply is exponential tilting, which for a Gaussian is exactly a mean shift. With nominal p(y) = N(y; x, σ²) and error set A = {y < 0} (BPSK, x = +1 sent), take

    q(y) = N(y; x + θ, σ²),   θ < 0  (shift the mean toward / across the decision boundary),

with closed-form likelihood ratio

    w(y) = p(y)/q(y) = exp( [ −2(y − x)θ + θ² ] / (2σ²) ).

**Choosing θ.** The first moment is fixed (= p_f for all θ), so minimizing variance is the same as minimizing the second moment

    M(θ) = E_{q_θ}[1_A w_θ²] = E_p[1_A w_θ]
         = E_p[1_A exp((−2(Y − x)θ + θ²)/(2σ²))].

Each term is the exponential of a quadratic with positive curvature, so M is convex in θ. Its stationary point obeys

    θ̂ = Σ 1_A(y_i)(y_i − x) e^{−(y_i−x)θ̂/σ²} / Σ 1_A(y_i) e^{−(y_i−x)θ̂/σ²}

because the common e^{θ̂²/(2σ²)} factor cancels. The fixed point is the failure-weighted mean noise excursion. A safe, asymptotically-optimal default is **drift reversal**, θ = −x, putting the tilted mean on the decision boundary so A becomes roughly a coin flip; θ can then be refined adaptively from the failures the tilted sampler produces, with the appropriate likelihood factor if those samples came from an earlier tilt.

**Pitfall.** Over-tilting (|θ| too large, in the direction of A) makes q lighter-tailed than p over part of A, so the few relevant samples carry enormous weights p/q and the variance can exceed crude MC, even diverge. Keep q's support and tails covering all of {1_A p ≠ 0}. When p/q is known only up to a constant, use the self-normalized estimator p̂ = Σ 1_A w / Σ w (small O(1/n) bias, more stable).

## Algorithm

1. Draw X_i ∼ q (the tilted sampler), for i = 1..n.
2. Compute the likelihood ratio w_i = p(X_i)/q(X_i).
3. Return p̂ = (1/n) Σ 1_A(X_i) w_i (or the self-normalized form if w is unnormalized).
4. Report the empirical variance to certify the relative accuracy; optionally refine the tilt θ from the observed failures and repeat.

## Code (BPSK over AWGN: crude MC vs. importance sampling)

```python
import numpy as np
from scipy.special import erfc

def ebn0_to_sigma(ebn0_db):
    """Noise std for unit-energy BPSK at Eb/N0 (dB): N0 = 1/(Eb/N0), sigma^2 = N0/2."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return np.sqrt(1.0 / (2.0 * ebn0))

def theoretical_ber(ebn0_db):
    """Ground-truth BPSK BER = Q(sqrt(2 Eb/N0)) = 0.5 erfc(sqrt(Eb/N0))."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0))

def channel(mean, sigma, n, rng):
    """Return n scalar AWGN samples y = mean + noise."""
    return mean + sigma * rng.standard_normal(n)

def is_error(y):
    """Failure set A for BPSK with x = +1: a sign flip."""
    return y < 0.0

def crude_mc_ber(ebn0_db, n, rng):
    """Plain Monte Carlo: send +1, error if the noise flips the sign."""
    sigma = ebn0_to_sigma(ebn0_db)
    x = 1.0
    y = channel(x, sigma, n, rng)                       # nominal y ~ N(x, sigma^2)
    errors = is_error(y).astype(float)                  # rare set A = {y < 0}
    return errors.mean(), errors.var(ddof=1) / n

def rare_event_estimator(ebn0_db, n, rng, theta=None):
    """Importance sampling by translating the noise mean toward the boundary."""
    sigma = ebn0_to_sigma(ebn0_db)
    x = 1.0
    if theta is None:
        theta = -x                                      # drift reversal: tilted mean -> 0
    y = channel(x + theta, sigma, n, rng)               # tilted y ~ N(x + theta, sigma^2)
    in_a = is_error(y)
    log_w = (-2.0 * (y - x) * theta + theta**2) / (2.0 * sigma**2)
    contrib = np.zeros(n)
    contrib[in_a] = np.exp(log_w[in_a])                 # 1_A * likelihood ratio p/q
    return contrib.mean(), contrib.var(ddof=1) / n

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for ebn0_db in [6, 8, 10]:
        n = 200_000
        p_true = theoretical_ber(ebn0_db)
        p_mc, v_mc = crude_mc_ber(ebn0_db, n, rng)
        p_is, v_is = rare_event_estimator(ebn0_db, n, rng)
        print(f"Eb/N0={ebn0_db:2d} dB  true={p_true:.3e}  "
              f"MC={p_mc:.3e}(var {v_mc:.1e})  IS={p_is:.3e}(var {v_is:.1e})  "
              f"var-ratio={v_mc / max(v_is, 1e-300):.0f}x")
```

The same recipe extends beyond BPSK: for a negative-drift random walk that must climb to a far level (queue overflow, insurance ruin) the tilt is the drift-reversing exponential change of measure set by the Lundberg constant E[e^{γΔ}] = 1; for Rayleigh deep fades or fibre PMD outage one tilts the fading/delay law toward the rare excursion. In every case the three ingredients are the same: change the measure to make the rare event typical, reweight by the likelihood ratio to stay unbiased, and choose the tilt to minimize the (convex) variance while keeping q's tails covering the failure set.
