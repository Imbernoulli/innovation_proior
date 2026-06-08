# The Feynman path integral (sum over histories)

## Problem

Ordinary quantum mechanics is built only on the Hamiltonian side of classical mechanics
(canonical `(q,p)`, `[q,p]=iℏ`, evolution by `Ĥ`). It gives no operational role to the classical
*action* `S = ∫ L dt` or the least-action principle, and it singles out a time axis — making it
awkward for relativistic problems and unusable for systems defined only by an action with a
time delay (e.g. direct action-at-a-distance electrodynamics), which have no Hamiltonian to
quantize. Goal: formulate quantum mechanics *directly in terms of the action*, recovering both
the classical limit `δS=0` and Schrödinger's equation.

## Key idea

The quantum amplitude (propagator) to go from one space-time point to another is a **sum over
all paths** connecting them. Each path `x(t)` contributes with **equal magnitude** and a phase
equal to its **classical action in units of ℏ**:

> contribution of a path `= const · exp[ (i/ℏ) S[x(t)] ]`, `S[x(t)] = ∫ L(ẋ, x) dt`.

Amplitudes **multiply along** a path (the action of the whole path is the sum of slice actions,
so the phases add in the exponent) and **add across** paths (integrate over intermediate
positions). This is the third formulation of quantum mechanics, equivalent to Schrödinger's and
Heisenberg's. It grows directly from Dirac's 1933 remark that `⟨q_{t+dt}|q_t⟩` *corresponds to*
`exp[(i/ℏ)L dt]` — promoted here from a correspondence to an exact equality with a definite
normalization, and then summed over paths.

## The propagator (final form)

For a non-relativistic particle of mass `m` in a potential `V(x)` (`L = ½mẋ² − V`), the kernel
from `(x_a,t_a)` to `(x_b,t_b)`:

```
K(x_b,t_b ; x_a,t_a) = ∫ exp[ (i/ℏ) S[x(t)] ] 𝒟x(t)      (paths with x(t_a)=x_a, x(t_b)=x_b)

  = lim_{N→∞} (1/A^N) ∫···∫ exp[ (i/ℏ) Σ_{i=0}^{N-1} S(x_{i+1},x_i) ] dx_1···dx_{N-1}

  ε = (t_b - t_a)/N ,   A = (2π i ℏ ε / m)^{1/2}   ⇔ each slice carries (m / 2π i ℏ ε)^{1/2}
  S(x_{i+1},x_i) = ε[ (m/2)((x_{i+1}-x_i)/ε)² − V(x_{i+1}) ]
```

For a scalar potential, evaluating `V` at the start, end, or midpoint changes only higher-order
terms in the `ε→0` limit. In `k` dimensions each short-time factor carries `A^{-k}`. The
normalization `A` is *not* free: it is forced by the zeroth-order identity matching below.

## Classical limit

As `ℏ→0`, `S/ℏ` is large and the phase `e^{iS/ℏ}` oscillates rapidly, so neighbouring paths
cancel — except near a path where `S` is **stationary**, `δS = 0`. The sum over histories
collapses onto the classical (least-action) path. Paths within `ΔS ∼ ℏ` of it interfere
constructively — the quantum fuzz around the classical trajectory.

## Recovery of the Schrödinger equation (the load-bearing derivation)

The single-slice kernel advancing `ψ` by `ε`, with `ξ = x − x_{old}`:

```
ψ(x,t+ε) = (1/A) ∫ exp[ i m ξ²/(2ℏε) ] exp[ −i ε V(x)/ℏ ] ψ(x−ξ, t) dξ.
```

`exp(imξ²/2ℏε)` oscillates fast except for `ξ ∼ (ℏε/m)^{1/2}`, so expand
`ψ(x−ξ) = ψ − ξψ_x + ½ξ²ψ_xx − ⋯` and `exp(−iεV/ℏ)=1−iεV/ℏ`. Fresnel moments:

```
∫ exp(imξ²/2ℏε) dξ      = (2π i ℏ ε / m)^{1/2}
∫ ξ  exp(imξ²/2ℏε) dξ   = 0
∫ ξ² exp(imξ²/2ℏε) dξ   = (iℏε/m)(2π i ℏ ε / m)^{1/2}   ⇒  ⟨ξ²⟩ = iℏε/m.
```

Zeroth order in `ε` ⇒ `(1/A)(2π i ℏ ε/m)^{1/2} = 1`, i.e. **`A = (2π i ℏ ε/m)^{1/2}`**.
Then to first order in `ε`:

```
ψ + ε ψ_t = (1 − iεV/ℏ)( ψ + (iℏε/2m) ψ_xx )
          = ψ + (iℏε/2m) ψ_xx − (iε/ℏ) V ψ + O(ε²).
```

Cancel `ψ`, divide by `ε`, multiply by `iℏ`:

```
iℏ ∂ψ/∂t = −(ℏ²/2m) ∂²ψ/∂x² + V(x) ψ.     ✓  (Schrödinger's equation)
```

This proves the path-integral formulation is equivalent to ordinary quantum mechanics for the
class of Lagrangians quadratic (possibly inhomogeneous) in the velocity — the class for which
Schrödinger's equation is established.

## Why the wave function emerges

Because `L` couples only neighbouring instants, the action sum factorizes at any time `t_k`:
`φ = ∫ χ*(x,t) ψ(x,t) dx`, where `ψ(x_k,t)` integrates `e^{(i/ℏ)ΣS}` over the *past* region only.
`ψ` depends solely on the past and carries everything needed to predict the future — it *is* the
quantum state, the same `ψ` that obeys Schrödinger's equation above.

## Notes on the construction

- **Equal magnitude per path**: the discriminating power is entirely in the phase; interference
  among paths reproduces all wave phenomena.
- **Dominant paths are non-differentiable**: typical slice velocity `(x_{i+1}-x_i)/ε ∼ (ℏ/mε)^{1/2}
  → ∞`; the important paths are continuous but nowhere differentiable (Brownian-like).
- **Midpoint rule for velocity-linear terms**: for a magnetic term `(e/c)A(x)·ẋ`, endpoint vs.
  midpoint evaluation of the slice action differs at `O(ε)` (since `(x_{i+1}-x_i)² ∼ ε`) and
  shifts the Hamiltonian by `(ℏe/2imc)∇·A`; use
  `S = ε L((x_{i+1}-x_i)/ε, (x_{i+1}+x_i)/2)`. A pure potential `V(x)` is insensitive at first
  order (difference in the exponent is `O(ε^{3/2})`).

## Minimal numerical illustration (1-D short-time evolution)

```python
import numpy as np

hbar = 1.0

def slice_action(x_next, x, eps, m, V):
    # ε[ (m/2)((x_next-x)/eps)^2 - V(x_next) ]
    return eps * (0.5 * m * ((x_next - x) / eps) ** 2 - V(x_next))

def normalization(eps, m):
    # A = (2π i ℏ ε / m)^{1/2}, forced by zeroth-order identity matching.
    return np.sqrt(2j * np.pi * hbar * eps / m)

def short_time_kernel(x_next, x, eps, m, V):
    return np.exp(1j * slice_action(x_next, x, eps, m, V) / hbar) / normalization(eps, m)

def evolve_one_slice(psi, x_grid, eps, m, V):
    """psi(x', t+eps) = ∫ K(x', x) psi(x, t) dx  on a uniform grid."""
    dx = x_grid[1] - x_grid[0]
    K = short_time_kernel(x_grid[:, None], x_grid[None, :], eps, m, V)  # K[i,j] = K(x_i, x_j)
    return (K @ psi) * dx

def propagator(x_final, x_initial, t, N, m, V, x_grid):
    """Compose N short-time kernels by integrating over N-1 intermediate coordinates:
    the discretized sum over paths. Returns K(x_final, x_initial; t)."""
    eps = t / N
    dx = x_grid[1] - x_grid[0]
    # start from a grid delta at the nearest x_initial point, propagate N slices, read off at x_final
    psi = np.zeros_like(x_grid, dtype=complex)
    psi[np.argmin(np.abs(x_grid - x_initial))] = 1.0 / dx
    for _ in range(N):
        psi = evolve_one_slice(psi, x_grid, eps, m, V)
    return psi[np.argmin(np.abs(x_grid - x_final))]
```
