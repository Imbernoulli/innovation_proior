# The Mermin-Wagner theorem: no continuous-symmetry breaking in 1D or 2D

## Problem

For the isotropic Heisenberg model with short-range exchange, does a one- or two-dimensional
system develop spontaneous (ferro- or antiferro-) magnetic order at a finite temperature? The
naive spin-wave count diverges in d ≤ 2 but assumes the ordered state, so it only hints. The goal
is a rigorous statement that bounds the order parameter without presupposing order, valid for
quantum spin S at any T > 0.

## Key idea

The spontaneous magnetization is a quasi-average — apply a small ordering field b, take N → ∞,
then send b → 0 — so it suffices to **upper-bound** the magnetization m(T,b) by a function that
vanishes as b → 0. The Bogoliubov inequality supplies such a bound from exact thermal averages.
Choosing the spin raising/lowering Fourier components places the magnetization in its commutator
slot; the spin-length sum rule caps the anticommutator; the energy-cost double commutator is
bounded by a term ∝ |bm| plus a term ∝ k² (the k² being a rigorous consequence of
1 − cos(k·R) ≤ ½k²R², not a spin-wave dispersion). Dividing and integrating over k yields an
integral whose small-k behavior is ∫ k^{d-3} dk: convergent for d = 3, divergent for d ≤ 2. The
divergence forces m → 0 as b → 0 in one and two dimensions, independent of the ordering
wavevector, so both ferromagnetism and antiferromagnetism are excluded.

## Setup

Isotropic Heisenberg Hamiltonian with finite-range exchange and an ordering field of wavevector
**K** (K = 0 ferromagnetic, K = zone boundary staggered/antiferromagnetic):

  H = − Σ_{ij} J_{ij} **S**_i · **S**_j − b Σ_i e^{−i**K**·**R**_i} S_i^z .

Spin algebra: [S^z, S^±] = ±ℏ S^±, [S^+, S^-] = 2ℏ S^z. Fourier components
S^±(**k**) = Σ_i e^{i**k**·**R**_i} S_i^±. Order parameter (staggered magnetization)
m = (1/N)|⟨Σ_i e^{−i**K**·**R**_i} S_i^z⟩|.

Hypotheses (each load-bearing): **isotropic** exchange (no spin anisotropy → no gap), and
**short-range** exchange (finite second moment J̄ ∝ Σ_j |J_{ij}|(R_i − R_j)²).

## The Bogoliubov inequality

For any operators A, C and Hamiltonian H in thermal equilibrium at T = 1/(k_B β):

  ½ β ⟨{A, A⁺}⟩ · ⟨[[C, H], C⁺]⟩  ≥  |⟨[A, C]⟩|² .

*Proof.* On operators define the positive-semidefinite form
B(X,Y) = Σ_{E_n≠E_m} w_n ⟨n|X⁺|m⟩⟨m|Y|n⟩/(E_n − E_m), with w_n = e^{−βE_n}/Z; positivity follows
from (w_n − w_m)/(E_n − E_m) ≥ 0, so Schwarz gives |B(A,Y)|² ≤ B(A,A) B(Y,Y). Take Y = [C⁺,H];
the identity ⟨m|[C⁺,H]|n⟩ = (E_n − E_m)⟨m|C⁺|n⟩ cancels the denominator, giving
B(A,[C⁺,H]) = ⟨[C⁺,A⁺]⟩ (so |B| = |⟨[A,C]⟩|) and B([C⁺,H],[C⁺,H]) = ⟨[[C,H],C⁺]⟩ ≥ 0. For the
remaining norm, (w_n − w_m)/(E_n − E_m) = [(w_n+w_m)/(E_n−E_m)]·tanh(β(E_m−E_n)/2) < (β/2)(w_n+w_m)
since |tanh x| < |x|, whence B(A,A) < (β/2)⟨{A,A⁺}⟩. Combining gives the inequality. ∎

## The three building blocks (for the Heisenberg model)

Choose C = S^+(**k**), A = S^-(−**k** − **K**).

1. **Commutator = order parameter.** [A,C] = −2ℏ S^z(−**K**), so
   |⟨[A,C]⟩|² = 4ℏ² N² m².

2. **Anticommutator sum rule.** Σ_**k** ⟨{A,A⁺}⟩ = N Σ_i ⟨2(S_i^x)² + 2(S_i^y)²⟩
   ≤ 2ℏ² S(S+1) N² (since (S^x)² + (S^y)² = **S**² − (S^z)² ≤ ℏ²S(S+1)).

3. **Double-commutator bound.** Splitting H into exchange + field and using
   1 − cos(**k**·(**R**_m−**R**_p)) ≤ ½ k²(R_m−R_p)²,
   ⟨[[C,H],C⁺]⟩ ≤ 2Nℏ² ( |b m| + S(S+1) J̄ k² ),
   where the |bm| term comes from the field and the k² term from the exchange (finite ⟺ short range,
   gapless ⟺ isotropic). The bound is independent of **K**.

## From the inequality to the bound

Divide the per-**k** Bogoliubov inequality by the double commutator and sum over **k**; use block 2
on the right and pull the k-independent block 1 out on the left:

  (2 N m²) Σ_**k** 1/(|bm| + S(S+1) J̄ k²)  ≤  β S(S+1) N² .

Replace Σ_**k** → [N V_d/(2π)^d] ∫ d^d k (V_d the volume per spin; the N cancels one power of N on
each side) and restrict to the inscribed sphere of radius k̄ (which only strengthens the inequality):

  S(S+1)  >  [2 V_d Ω_d m² T /(2π)^d] ∫₀^{k̄} dk · k^{d-1} / ( |b m| + S(S+1) J̄ k² ) ,

with Ω_d the surface of the unit d-sphere. The dimension enters only through k^{d-1} dk.

- **d = 1:** ∫ dk/(bm+Jk²) = (1/√(bmJ)) arctan(k̄√(J/bm)) → (π/2)/√(bmJ) ∝ 1/√(bm) (diverges).
- **d = 2:** ∫ k dk/(bm+Jk²) = (1/2J) ln(1 + Jk̄²/(bm)) → ∞ (diverges logarithmically).
- **d = 3:** ∫ k² dk/(bm+Jk²) → finite as b → 0 (integrand → 1/J; no divergence).

The small-k integrand is k^{d-1}/k² = k^{d-3}, whose integral diverges at the origin iff d ≤ 2.

## Theorem and explicit bounds

**Theorem (absence of spontaneous magnetization in d ≤ 2).** For the isotropic spin-S Heisenberg
model with finite-range exchange on a one- or two-dimensional lattice, at any nonzero temperature
T, the (staggered) magnetization in an ordering field b obeys

  d = 1:   m < c₁ · b^{1/3} / T^{2/3},     c₁ = S(S+1) J̄^{1/3},
  d = 2:   m < c₂ · 1 / ( √T · √|ln b| ),   c₂ = √(2π) S(S+1) J̄^{1/2}.

Both right-hand sides → 0 as b → 0 for every finite T. The bound is independent of the ordering
wavevector **K**, so it forbids both ferromagnetic (K = 0) and antiferromagnetic/staggered (K ≠ 0)
spontaneous order. In three dimensions the controlling integral converges, the argument gives no
vanishing ceiling on m, and ordinary long-range order is not excluded. The two hypotheses are
necessary: an easy-axis anisotropy adds a gap Δ to the denominator (bm + Δ + Jk²), cutting off the
divergence; a long-range exchange tail makes J̄ diverge and changes the small-k power, either of
which can restore order. ∎

## Numerical illustration of the controlling integral

```python
import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy renamed trapz -> trapezoid

def infrared_integral(d, bm, kbar=1.0, J=1.0, n=200000):
    """I_d(bm) = ∫_0^kbar k^{d-1}/(bm + J k^2) dk. The bound reads
    S(S+1) > const * m^2 * T * I_d(bm); a diverging I_d as bm->0 forces m->0."""
    k = np.linspace(0.0, kbar, n)
    return _trapz(k**(d - 1) / (bm + J * k**2), k)

for d in (1, 2, 3):
    vals = [infrared_integral(d, bm) for bm in (1e-1, 1e-3, 1e-5, 1e-7)]
    print(f"d={d}:  I(bm) for bm=1e-1..1e-7 ->", [f"{v:.3f}" for v in vals])

# d=1:  ~ 1/sqrt(bm)      diverges (x10 per 100x drop)  -> m < c1 b^{1/3}/T^{2/3} -> 0
# d=2:  ~ (1/2) ln(1/bm)  diverges logarithmically       -> m < c2 /(sqrt(T) sqrt(|ln b|)) -> 0
# d=3:  -> finite (~1.0)  converges                       -> no vanishing ceiling; order survives
```

Running it reproduces the theorem: in d = 1 the integral grows like 1/√(bm), in d = 2 like a
logarithm, and in d = 3 it saturates to a constant — divergence (no order) in one and two
dimensions, convergence (order permitted) in three.
