# The Dammann grating

A **Dammann grating** is a binary-phase diffractive element that splits one input beam into a regular array
of equal-intensity beams (a "fan-out" or beam splitter), using a single thin transparent surface with only
two phase levels, 0 and π.

## Problem it solves

Take one collimated beam and produce a chosen symmetric set of equally bright copies on a regular grid about
the optical axis, in a single passive transmissive pass, wasting as little light as possible — and with a
layout simple enough to etch reliably (one step, two depths).

## Key idea

A periodic transmittance has a far field that is a comb of diffraction orders at sinθ_m = mλ/d, and the
amplitude in order m equals the m-th Fourier-series coefficient of one period. So equalizing a symmetric
target set of orders is equalizing Fourier coefficients. To avoid wasting light, use a **pure phase** element
(|t| = 1, lossless); to make it fabricable in one etch, restrict the phase to **two levels 0 and π** (so
t ∈ {+1, −1}); and make the period **even** so the coefficients are real and the fan-out is symmetric. The
period is then fully described by its **sign-flip transition points**, and the order amplitudes are
closed-form sine sums of those points. The design reduces to: solve for the transition positions that make
the target-order magnitudes equal while keeping the most power in them.

## Final design

Normalize the period to 1; work on the half-period [0, 1/2]. Interior transitions
0 = x₀ < x₁ < … < x_K < x_{K+1} = 1/2 define an even ±1 profile (sign (−1)^j on [x_j, x_{j+1})). The
diffraction-order amplitudes are

- a₀ = 2 Σ_{j=0}^{K} (−1)^j (x_{j+1} − x_j)   (signed length balance — the central spot),
- a_m = (2/(πm)) Σ_{j=1}^{K} (−1)^{j-1} sin(2πm x_j),  m ≥ 1.

Order intensity is |a_m|²; by evenness a_{−m} = a_m. For a 2Q−1 spot fan-out, choose K = Q−1 transitions
and solve |a₀| = |a₁| = … = |a_{Q−1}|. The captured efficiency is
η = a₀² + 2 Σ_{m=1}^{Q−1} a_m², the power in the 2Q−1 target orders.

Closed-form smallest case (single transition → 1×3): a₀ = 4x₁ − 1, a₁ = (2/π) sin(2πx₁). Setting a₀² = a₁²
gives two equivalent intensity branches; one is x₁ ≈ 0.36763, with I₀ = I₁ ≈ 0.2214 and η ≈ 66.4%. Two
transitions (1×5) reach uniform orders at η ≈ 77.4%. Binary 0/π caps achievable efficiency below a multilevel
phase element. The transition *pattern* is wavelength-independent; only the period d (spot spacing) and the
half-wave etch depth d_etch = λ/(2(n−1)) scale with λ.

## Code

```python
import numpy as np
from scipy.optimize import brentq, minimize

def binary_profile(x, transitions, start=+1.0):
    """Render the even +/-1 period at coordinates x; map +1/-1 to phase 0/pi for the mask."""
    transitions = np.sort(np.asarray(transitions, float))
    edges = np.concatenate(([0.0], transitions, [0.5]))
    idx = np.searchsorted(edges, np.abs(x), side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return start * (-1.0) ** idx

def order_amplitudes(transitions, M):
    """Diffraction-order amplitudes a_0..a_M for an even binary +/-1 period."""
    t = np.sort(np.asarray(transitions, float))
    a = np.zeros(M + 1)
    edges = np.concatenate(([0.0], t, [0.5]))
    a[0] = 2.0 * np.sum(((-1.0) ** np.arange(len(edges) - 1)) * np.diff(edges))
    if M >= 1 and len(t):
        m = np.arange(1, M + 1)
        sign = (-1.0) ** np.arange(len(t))
        S = (sign[None, :] * np.sin(2 * np.pi * np.outer(m, t))).sum(axis=1)
        a[1:] = (2.0 / (np.pi * m)) * S
    return a

def design_cost(transitions, num_nonnegative, efficiency_weight=0.05):
    """Equalize m=0..num_nonnegative-1 and prefer higher captured power."""
    t = np.sort(np.clip(transitions, 1e-4, 0.5 - 1e-4))
    I = order_amplitudes(t, num_nonnegative - 1) ** 2
    target = I[:num_nonnegative]
    eta = I[0] + 2.0 * np.sum(I[1:num_nonnegative])
    uniformity = (target.max() - target.min()) / (target.max() + target.min() + 1e-12)
    return uniformity + efficiency_weight * (1.0 - eta)

def design(num_nonnegative, num_transitions=None, restarts=400, seed=0):
    """Multistart search for a 2*num_nonnegative-1 order splitter."""
    K = num_nonnegative - 1 if num_transitions is None else num_transitions
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        t0 = np.sort(rng.uniform(0.01, 0.49, K))
        r = minimize(design_cost, t0, args=(num_nonnegative,), method="Nelder-Mead",
                     options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
        if best is None or r.fun < best.fun:
            best = r
    t = np.sort(np.clip(best.x, 1e-4, 0.5 - 1e-4))
    I = order_amplitudes(t, num_nonnegative - 1) ** 2
    eta = I[0] + 2.0 * np.sum(I[1:num_nonnegative])
    target = I[:num_nonnegative]
    uniformity = (target.max() - target.min()) / (target.max() + target.min() + 1e-12)
    return t, eta, uniformity, I

if __name__ == "__main__":
    for q in (2, 3):                          # q=2 -> 1x3, q=3 -> 1x5
        t, eta, uniformity, I = design(q, restarts=400)
        print(f"{2*q-1} equal orders: transitions={np.round(t, 4)}  "
              f"eff={eta*100:.2f}%  non-unif={uniformity*100:.3f}%")

    f = lambda t1: (4 * t1 - 1) ** 2 - (2 / np.pi * np.sin(2 * np.pi * t1)) ** 2
    t1 = brentq(f, 0.30, 0.39)
    a0, a1 = 4 * t1 - 1, 2 / np.pi * np.sin(2 * np.pi * t1)
    print(f"[1x3] t1={t1:.5f}  I0=I1={a0**2:.5f}  eff={(a0**2 + 2*a1**2)*100:.3f}%")
```

Typical output: `1×3 → eff≈66.4%`; `1×5 → eff≈77.4%, non-uniformity ≈ 0`. The two-dimensional fan-out is
the outer product of two such 1-D gratings, one per axis.
