# Particle Mesh Ewald (PME)

## Problem

Evaluate the exact Ewald electrostatic energy and the force on every atom of a
large periodic unit cell (N ≳ 10⁴ charges) at every MD step. The Ewald
reciprocal-space sum is the bottleneck: O(N²) at fixed splitting parameter,
O(N^{3/2}) with it optimized. We want near-linear cost, accuracy tunable to any
tolerance independent of N, and energy/forces that are continuous in the atomic
positions.

## Key idea

Choose the Ewald splitting parameter β large enough that the direct (screened)
sum is short-ranged — O(N) with a cutoff — so only the reciprocal sum is
expensive. The reciprocal pair potential Φ_rec depends only on the difference of
fractional coordinates and is fixed throughout the simulation, so **precompute it
(and its gradient) once on a regular K₁×K₂×K₃ grid.** Approximate Φ_rec at
arbitrary atomic separations by **centred, high-order Lagrange interpolation**
whose order p sets the accuracy. Because the interpolation weights factorize over
the three lattice axes, the interpolated pairwise reciprocal sum collapses into a
**discrete convolution** of a gridded charge array with the precomputed Φ_rec
grid — and a periodic convolution is evaluated by **FFT**. The result is O(N log
N) with the error controlled by the grid spacing to the power 2p.

## The Ewald decomposition

For a neutral cell U with charges q_i at r_i, lattice volume V, reciprocal
vectors m = m₁a₁*+m₂a₂*+m₃a₃*:

  E = Σ_{i<j} q_i q_j erfc(βr_ij)/r_ij                          (direct, O(N) with cutoff)
      − (β/√π) Σ_i q_i²                                          (self-energy, O(N))
      + ½ Σ_{i,j} q_i q_j Φ_rec(r_j − r_i;β)                     (reciprocal — the target)
      + J(D),                                                    (surface/dipole, O(N))

  Φ_rec(r;β) = (1/πV) Σ_{m≠0} [exp(−π²m²/β²)/m²] exp(2πi m·r).

In fractional coordinates f_k = a_k*·r:

  Φ_rec(f₁,f₂,f₃;β) = (1/πV) Σ_{m≠0} [exp(−π²m²/β²)/m²]
                        exp[2πi(m₁f₁ + m₂f₂ + m₃f₃)].

Total E is invariant to β (β only shifts work between direct and reciprocal sums).

## Centred Lagrange interpolation weights

For real x, [x] = floor(x), and offset k_{p,K}(x) = [Kx] − p + 1. With C(n,k) the
binomial coefficient, define on the canonical window 0 ≤ k ≤ 2p−1

  φ_{p,K}(x,k) = [ (−1)^k C(2p−1,k) /(x − k/K) ]
                 / Σ_{l=0}^{2p−1} (−1)^l C(2p−1,l) /(x − l/K),   (0 otherwise)

and shift it to centre on x:

  θ_{p,K}(x,k) = φ_{p,K}[ x − k_{p,K}(x), k − k_{p,K}(x) ].

Properties: nonzero for only 2p values of k (local stencil); Σ_k θ_{p,K}(x,k)=1
(partition of unity); θ_{p,K}(k/K,k)=1, θ_{p,K}(l/K,k)=0 for l≠k, so θ is
continuous in x — hence the interpolated energy and forces are continuous in the
atomic positions. Keeping x in the central interval avoids the ill-conditioning
of high-order polynomial interpolation.

## Gridded charge, convolution, energy, forces

Spread the charges onto the grid:

  Q(l₁/K₁,l₂/K₂,l₃/K₃) = Σ_{j} q_j θ_{p,K₁}(f_{j1},l₁) θ_{p,K₂}(f_{j2},l₂)
                            θ_{p,K₃}(f_{j3},l₃).        [O((2p)²N)]

Interpolated reciprocal energy as a discrete convolution Φ_rec*Q:

  Ê_{rec,p}(i) = ½ Σ_{k₁,k₂,k₃} q_i θ_{p,K₁}(f_{i1},k₁) θ_{p,K₂}(f_{i2},k₂)
                   θ_{p,K₃}(f_{i3},k₃) · (Φ_rec * Q)(k₁/K₁,k₂/K₂,k₃/K₃),
  E_rec = Σ_i Ê_{rec,p}(i) = ½ Σ_grid Q · (Φ_rec * Q).

Evaluate Φ_rec*Q by FFT (precompute FFT(Φ_rec) once at start):

  spread → FFT(Q) → multiply by FFT(Φ_rec) pointwise → inverse FFT → gather.

Cost: convolution O(K₁K₂K₃ log K₁K₂K₃) + spread/gather O((2p)²N). Forces come
from interpolating Φ_rec's grid gradient the same way (analytic, continuous);
interpolating energy and forces symmetrically makes the reciprocal forces sum to
zero to machine precision (Newton's third law exact).

## Accuracy and complexity

Lagrange interpolation error of the complex exponential (per axis):

  | exp(2πi m x) − Σ_k exp(2πi m k/K) θ_{p,K}(x,k) | < 2 C(2p,p) (m/4K)^{2p},
  C(2p,p) = (2p)!/(p!)².

Propagated through Φ_rec and bounded by an integral (change of variables to
x_k = a*_{1k}m₁+a*_{2k}m₂+a*_{3k}m₃, spherical coordinates, Cauchy–Schwarz on m_k,
Gaussian moments):

  | Φ̂_{rec,p} − Φ_{rec,p} | ≤ 8 C(2p,p) ((2p)!/p!) (β/√π) (β/8π)^{2p}
        · [ (a₁/K₁)^{2p} + (a₂/K₂)^{2p} + (a₃/K₃)^{2p} ].

The error scales as (grid spacing a_k/K_k)^{2p}. Fix a_k/K_k < 1 Å and choose p:
the error → 0 geometrically for any tolerance, while K₁K₂K₃ ∝ cell volume ∝ N.
Hence cost = **O(N log N)** at any fixed accuracy. Accuracy and cost are
decoupled: the order p (with each atom touching (2p)³ grid points) is an accuracy
knob set independently of N, so a modest order on a sub-Ångström grid suffices for
MD-grade forces while a couple of extra orders drive the error toward near-exact.
The method is fully general for non-orthogonal cells and slots into a Verlet-list
MD code.

## Implementation

The reciprocal-space evaluator (spread → FFT → influence → inverse FFT → gather).
Direct, self, and surface terms are the standard O(N) Ewald bookkeeping.

```python
import numpy as np
from math import comb

def lagrange_weights(u, p):
    """Centred (2p-1)th-order Lagrange weights theta_{p,K} at scaled coord u=K*f,
    on the 2p-point window. Returns (left_index, weights[2p])."""
    floor_u = int(np.floor(u))
    left = floor_u - p + 1                       # k_{p,K}(x) = [Ku] - p + 1
    nodes = np.arange(2 * p)
    x = u - left
    bary = np.array([(-1) ** k * comb(2 * p - 1, k) for k in nodes], float)
    diff = x - nodes
    on_node = np.isclose(diff, 0.0)
    if on_node.any():
        w = np.where(on_node, 1.0, 0.0)
    else:
        terms = bary / diff
        w = terms / terms.sum()                  # partition of unity
    return left, w

class PME:
    def __init__(self, cell, beta, grid_dims, order):
        self.cell = np.asarray(cell, float)      # 3x3, lattice vectors as columns
        self.V = abs(np.linalg.det(self.cell))
        self.a_star = np.linalg.inv(self.cell).T # rows a_k*: f = a_star @ r
        self.beta = beta
        self.K = tuple(grid_dims)
        self.p = order
        self.M = self.K[0] * self.K[1] * self.K[2]
        self._setup()

    def _setup(self):
        K1, K2, K3 = self.K
        m1 = np.fft.fftfreq(K1, d=1.0 / K1)
        m2 = np.fft.fftfreq(K2, d=1.0 / K2)
        m3 = np.fft.fftfreq(K3, d=1.0 / K3)
        M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing="ij")
        A = self.a_star
        mx = M1 * A[0, 0] + M2 * A[1, 0] + M3 * A[2, 0]
        my = M1 * A[0, 1] + M2 * A[1, 1] + M3 * A[2, 1]
        mz = M1 * A[0, 2] + M2 * A[1, 2] + M3 * A[2, 2]
        m2sq = mx ** 2 + my ** 2 + mz ** 2
        kern = np.zeros_like(m2sq)
        nz = m2sq > 0
        kern[nz] = np.exp(-(np.pi ** 2) * m2sq[nz] / self.beta ** 2) \
                   / (np.pi * self.V * m2sq[nz])
        self.kern = kern                         # reciprocal Ewald kernel in k-space
        self.m_vec = (mx, my, mz)

    def _spread(self, positions, charges):
        K1, K2, K3 = self.K
        Q = np.zeros((K1, K2, K3), dtype=complex)
        info = []
        for r, q in zip(positions, charges):
            f = self.a_star @ r
            f = f - np.floor(f)
            l1, w1 = lagrange_weights(f[0] * K1, self.p)
            l2, w2 = lagrange_weights(f[1] * K2, self.p)
            l3, w3 = lagrange_weights(f[2] * K3, self.p)
            i1 = (l1 + np.arange(2 * self.p)) % K1
            i2 = (l2 + np.arange(2 * self.p)) % K2
            i3 = (l3 + np.arange(2 * self.p)) % K3
            Q[np.ix_(i1, i2, i3)] += q * w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
            info.append((i1, i2, i3, w1, w2, w3, q))
        return Q, info

    def energy_and_forces(self, positions, charges):
        Q, info = self._spread(positions, charges)
        Qhat = np.fft.fftn(Q)                     # F(Q)(m) = approximate structure factor S(m)
        # E_rec = 1/2 sum Q.(Phi_rec*Q) = 1/(2 pi V) sum kernel |S(m)|^2  (Parseval)
        E_rec = 0.5 * np.sum(self.kern * np.abs(Qhat) ** 2)
        pot_hat = self.kern * Qhat               # F(Phi_rec * Q)
        mx, my, mz = self.m_vec                   # *M restores numpy ifft normalization
        gx = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * mx)).real * self.M
        gy = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * my)).real * self.M
        gz = np.fft.ifftn(pot_hat * (1j * 2 * np.pi * mz)).real * self.M
        F = np.zeros_like(positions)
        for k, (i1, i2, i3, w1, w2, w3, q) in enumerate(info):
            wc = w1[:, None, None] * w2[None, :, None] * w3[None, None, :]
            b = np.ix_(i1, i2, i3)
            F[k] = -q * np.array([np.sum(wc * gx[b]),
                                  np.sum(wc * gy[b]),
                                  np.sum(wc * gz[b])])
        return E_rec, F
```

## Worked numerical check

A small sanity check that the gridded convolution reproduces the brute-force
reciprocal Ewald sum E_rec = (1/2πV) Σ_{m≠0} [exp(−π²m²/β²)/m²] |S(m)|², and that
the error shrinks geometrically with the interpolation order p.

```python
import numpy as np

def ewald_recip_bruteforce(pos, q, cell, beta, mmax):
    """Exact reciprocal-space Ewald energy via the structure factor."""
    V = abs(np.linalg.det(cell)); a_star = np.linalg.inv(cell).T
    E = 0.0
    for m1 in range(-mmax, mmax + 1):
      for m2 in range(-mmax, mmax + 1):
        for m3 in range(-mmax, mmax + 1):
            if m1 == m2 == m3 == 0: continue
            m = m1 * a_star[0] + m2 * a_star[1] + m3 * a_star[2]
            m2sq = m @ m
            S = np.sum(q * np.exp(2j * np.pi * (pos @ m)))
            E += np.exp(-(np.pi ** 2) * m2sq / beta ** 2) / m2sq * abs(S) ** 2
    return E / (2 * np.pi * V)

rng = np.random.default_rng(0)
L = 10.0; cell = np.eye(3) * L
N = 40
pos = rng.uniform(0, L, size=(N, 3))
q = rng.uniform(-1, 1, size=N); q -= q.mean()        # neutral cell
beta = 0.35

E_exact = ewald_recip_bruteforce(pos, q, cell, beta, mmax=12)
for p in (2, 4, 6):
    pme = PME(cell, beta, grid_dims=(32, 32, 32), order=p)
    E_pme, _ = pme.energy_and_forces(pos, q)
    print(f"p={p}: E_rec(PME)={E_pme:.6f}  E_rec(exact)={E_exact:.6f}  "
          f"rel.err={abs(E_pme - E_exact)/abs(E_exact):.2e}")
```

The PME reciprocal energy converges to the brute-force value, with the relative
error falling as the interpolation order p (hence the grid spacing exponent 2p)
increases — exactly the (a_k/K_k)^{2p} behaviour of the error bound — while the
cost stays O(N log N) in the number of charges.
