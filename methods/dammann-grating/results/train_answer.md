The problem is to take one collimated beam and turn it into a regular array of equally bright copies with a single passive transmissive element. Ordinary approaches all hit the same wall. A stack of partially reflecting plates is bulky, alignment-sensitive, and does not scale to a two-dimensional grid. An absorbing amplitude mask can imprint a Fourier pattern, but it throws away most of the light to absorption and leaves a bright unwanted zero order. A sinusoidal or continuous phase grating is lossless, yet its diffraction-order weights are fixed special-function values that cannot be tuned independently, and a gray-scale surface relief is hard to fabricate faithfully. What is needed is a thin, fully transparent element with only two etch depths whose periodic profile can be optimized to put equal power into a chosen symmetric set of diffraction orders.

The solution is a Dammann grating. It is a binary-phase diffractive element whose phase takes only two values, 0 and π, on a periodic surface relief. Because |t| = 1 everywhere, the element absorbs nothing; by Parseval the total incident power is conserved and merely redistributed among the diffraction orders. The grating period is made even, so the Fourier coefficients are real and the fan-out is symmetric. The entire design is then specified by the sign-flip transition points inside the half-period. For an even ±1 profile with transitions x_1, …, x_K on [0, 1/2], the order amplitudes are a_0 = 2 Σ_j (−1)^j (x_{j+1} − x_j) and a_m = (2/(πm)) Σ_{j=1}^K (−1)^{j−1} sin(2πm x_j) for m ≥ 1. The design task is to choose the transitions so that |a_0| = |a_1| = … = |a_{Q−1}| for the desired 2Q−1 equal spots, while maximizing the captured efficiency η = a_0^2 + 2 Σ_{m=1}^{Q−1} a_m^2.

The simplest case uses one transition to make three equal beams. Setting a_0^2 = a_1^2 gives x_1 ≈ 0.36763, with about 66% of the incident power landing in the three target orders. Two transitions give five equal beams at roughly 77% efficiency. The transition positions are pure numbers on the normalized period, so the same lithographic pattern works for any wavelength once the physical period and the half-wave etch depth λ/(2(n−1)) are scaled appropriately. A two-dimensional fan-out is obtained by using one such grating along each axis.

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
