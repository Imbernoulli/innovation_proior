# The Dammann grating

A **Dammann grating** is a binary-phase diffractive element that splits one input beam into a regular array
of equal-intensity beams (a "fan-out" or beam splitter), using a single thin transparent surface with only
two phase levels, 0 and π.

## Problem it solves

Take one collimated beam and produce N equally bright copies on a regular grid about the optical axis, in a
single passive transmissive pass, wasting as little light as possible — and with a layout simple enough to
etch reliably (one step, two depths).

## Key idea

A periodic transmittance has a far field that is a comb of diffraction orders at sinθ_m = mλ/d, and the
amplitude in order m equals the m-th Fourier-series coefficient of one period. So equalizing the first N
orders is equalizing Fourier coefficients. To avoid wasting light, use a **pure phase** element (|t| = 1,
lossless); to make it fabricable in one etch, restrict the phase to **two levels 0 and π** (so t ∈ {+1, −1});
and make the period **even** so the coefficients are real and the fan-out is symmetric. The period is then
fully described by its **sign-flip transition points**, and the order amplitudes are closed-form sine sums of
those points. The design reduces to: solve for the transition positions that make the target-order
magnitudes equal while keeping the most power in them.

## Final design

Normalize the period to 1; work on the half-period [0, 1/2]. Interior transitions
0 = x₀ < x₁ < … < x_K < x_{K+1} = 1/2 define an even ±1 profile (sign (−1)^j on [x_j, x_{j+1})). The
diffraction-order amplitudes are

- a₀ = 2 Σ_{j=0}^{K} (−1)^j (x_{j+1} − x_j)   (signed length balance — the central spot),
- a_m = (2/(πm)) Σ_{j=1}^{K} (−1)^{j-1} sin(2πm x_j),  m ≥ 1.

Order intensity is |a_m|²; by evenness a_{−m} = a_m. **Design**: choose the transitions so that
|a₀| = |a₁| = … = |a_{N−1}| (the first N orders equal) and the efficiency η = a₀² + 2 Σ_{m=1}^{N−1} a_m²
(power in the 2N−1 target orders) is maximized; K transitions yield 2K+1 equal orders.

Closed-form smallest case (single transition → 1×3): a₀ = 4x₁ − 1, a₁ = (2/π) sin(2πx₁). Setting a₀² = a₁²
gives x₁ ≈ 0.36763, I₀ = I₁ ≈ 0.2214, efficiency η ≈ 66.4%. Two transitions (1×5) reach uniform orders at
η ≈ 77.4%. Binary 0/π caps achievable efficiency (commonly quoted below ~86% for fan-outs). The transition
*pattern* is wavelength-independent; only the period d (spot spacing) and the half-wave etch depth
d_etch = λ/(2(n−1)) scale with λ.

## Code

```python
import numpy as np
from scipy.optimize import minimize, brentq

def order_amplitudes(t, M):
    """Diffraction-order amplitudes a_0..a_M for an even binary +/-1 period with interior
    transitions t (ascending, in (0,1/2)).
        a_0 = 2*sum_j (-1)^j (x_{j+1}-x_j)
        a_m = (2/(pi m)) * sum_j (-1)^{j-1} sin(2 pi m t_j),  m>=1."""
    t = np.asarray(t, float)
    m = np.arange(0, M + 1)
    sign = (-1.0) ** np.arange(len(t))
    a = np.zeros(M + 1)
    edges = np.concatenate(([0.0], t, [0.5]))
    a[0] = 2 * np.sum(((-1.0) ** np.arange(len(edges) - 1)) * np.diff(edges))
    mm = m[1:]
    S = (sign[None, :] * np.sin(2 * np.pi * np.outer(mm, t))).sum(axis=1)
    a[1:] = (2 / (np.pi * mm)) * S
    return a

def cost(t, N):
    """N equal target orders (m = 0..N-1, i.e. 2N-1 spots): uniformity first, efficiency as a nudge."""
    t = np.sort(np.clip(t, 1e-4, 0.5 - 1e-4))
    I = order_amplitudes(t, N - 1) ** 2
    Itar = I[:N]
    eta = I[0] + 2 * np.sum(I[1:N])
    unif = (Itar.max() - Itar.min()) / (Itar.max() + Itar.min() + 1e-12)
    return unif + 0.1 * (1 - eta / (eta + 1e-9))

def design(N, K, restarts=400, seed=0):
    """Multistart search over K transition positions for the 2N-1 equal-order splitter."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        t0 = np.sort(rng.uniform(0.01, 0.49, K))
        r = minimize(cost, t0, args=(N,), method='Nelder-Mead',
                     options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
        if best is None or r.fun < best.fun:
            best = r
    t = np.sort(np.clip(best.x, 1e-4, 0.5 - 1e-4))
    I = order_amplitudes(t, N - 1) ** 2
    eta = I[0] + 2 * np.sum(I[1:N])
    Itar = I[:N]
    unif = (Itar.max() - Itar.min()) / (Itar.max() + Itar.min())
    return t, eta, unif, I

def grating_profile(x, transitions, start=+1.0):
    """Render the even +/-1 period at coordinates x (|x| in [0, 0.5]); map +1/-1 -> phase 0/pi for the mask."""
    y = np.full_like(x, start)
    edges = np.concatenate(([0.0], np.asarray(transitions, float), [0.5]))
    idx = np.searchsorted(edges, np.abs(x), side='right') - 1
    y = start * (-1.0) ** idx
    return y                                  # phase = np.where(y > 0, 0.0, np.pi)

if __name__ == "__main__":
    for N in (2, 3):                          # N=2 -> 1x3,  N=3 -> 1x5
        t, eta, unif, I = design(N, K=N, restarts=400)
        print(f"{2*N-1} equal orders: transitions={np.round(t, 4)}  "
              f"eff={eta*100:.2f}%  non-unif={unif*100:.3f}%")

    f = lambda t1: (4 * t1 - 1) ** 2 - (2 / np.pi * np.sin(2 * np.pi * t1)) ** 2
    t1 = brentq(f, 0.30, 0.39)
    a0, a1 = 4 * t1 - 1, 2 / np.pi * np.sin(2 * np.pi * t1)
    print(f"[1x3] t1={t1:.5f}  I0=I1={a0**2:.5f}  eff={(a0**2 + 2*a1**2)*100:.3f}%")
```

Typical output: `1×3 → t₁≈0.3676, eff≈66.4%`; `1×5 → eff≈77.4%, non-uniformity ≈ 0`. The two-dimensional
fan-out (an N×N spot array) is the outer product of two such 1-D gratings, one per axis.
