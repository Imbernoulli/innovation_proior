# The Ginzburg–Landau theory of superconductivity

## Problem

A field-expulsion (London) description of a superconductor fixes the density of
superconducting electrons as a temperature-only constant. It therefore cannot describe a
field/current destroying superconductivity, gives the wrong (negative) sign for the surface
energy between coexisting normal and superconducting regions of the same metal, and cannot
treat a spatially varying degree of superconductivity at all. The aim is a phenomenological,
gauge-invariant theory valid near T_c that produces all of this from ordinary material
parameters.

## Key idea

Promote the degree of superconductivity to a spatially varying **complex order parameter**
Ψ(r) — complex because a supercurrent requires a phase — and write the free energy as a
**functional** of Ψ and the vector potential A. Fix the uniform part by the Landau
second-order-transition expansion in |Ψ|²; add the magnetic field energy and a
**gauge-covariant kinetic (gradient) term**, the covariant form being *forced* by requiring
invariance under Ψ → Ψ e^{iχ}, A → A + (ℏc/e*)∇χ. Minimizing the functional yields the
governing equations, the supercurrent, two intrinsic lengths, and their ratio κ, which
splits superconductors into two classes at κ = 1/√2.

## Free energy functional

In a magnetic field, per unit volume,

    F = F_n + α|Ψ|² + (β/2)|Ψ|⁴ + (1/2m*)|(−iℏ∇ − (e*/c)A)Ψ|² + H²/8π ,

with β > 0; α(T) = a'(T − T_c) crosses zero at T_c, so α < 0 for T < T_c. The mass m* is not
observable (absorbable into the normalization of Ψ); e* is a charge of order e, fixed in
principle by experiment.

Uniform zero-field minimization gives |Ψ_∞|² = −α/β and the condensation energy
−α²/2β = −H_c²/8π, hence the anchoring relation

    α²/β = H_c²/4π .

## The Ginzburg–Landau equations

Varying F with respect to Ψ* and to A:

    (1/2m*)(−iℏ∇ − (e*/c)A)² Ψ + αΨ + β|Ψ|²Ψ = 0          (1st GL equation)

    ∇×H = (4π/c) j_s ,
      j_s = −(ie*ℏ/2m*)(Ψ*∇Ψ − Ψ∇Ψ*) − (e*²/m*c)|Ψ|² A    (2nd GL equation)
          = −(e*²/m*c)|Ψ|² ( A − (ℏc/e*)∇θ ),  Ψ = |Ψ|e^{iθ}.

Natural boundary condition at a free surface (Ψ is an *average*, not a true wave function, so
Ψ = 0 there is wrong and over-constrains the plate problem):

    n·(−iℏ∇ − (e*/c)A) Ψ = 0      ⟹      n·j_s = 0 .

The rigid-|Ψ| limit (|Ψ| = |Ψ_∞|) reduces the 2nd equation to London's, j_s = −(e*²/m*c)|Ψ_∞|²A,
recovering ∇²H = H/δ² — but now with a penetration depth that depends on |Ψ|² and hence on the
field.

## Characteristic lengths and the GL parameter

    coherence length    ξ = ℏ / sqrt(2 m* |α|)             (decay length of Ψ)
    penetration depth   δ = sqrt( m* c² β / (4π e*² |α|) ) (decay length of H)
    GL parameter        κ = δ/ξ = (m* c / ℏ e*) sqrt(β/2π)   (temperature-INDEPENDENT)

Both lengths diverge as (T_c − T)^{−1/2}; their ratio κ is a pure material number.

## Order-parameter wall and surface energy

In dimensionless units (length in ξ, Ψ in units of Ψ_∞, field in H_c√2 so that H_c = 1/√2):

    (1/κ²) Ψ'' = −(1 − A²)Ψ + Ψ³ ,        A'' = Ψ² A .

Zero-field s/n wall: −ξ² f'' − f + f³ = 0, with first integral 2ξ²f'² + 2f² − f⁴ = 1, giving

    f(x) = tanh( x / (√2 ξ) ) .

Surface energy at the n/s boundary, for small κ:

    σ_ns = (H_c²/8π) · δ_wall ,   δ_wall = ∫_0^∞ (1 − f⁴) dx = (4√2/3) ξ ≈ 1.886 ξ  > 0 ,

i.e. **positive**, produced from ordinary parameters — the main aim. The surface energy
changes sign with κ:

    σ_ns > 0  for  κ < 1/√2  (the ordinary regime) ;
    σ_ns < 0  for  κ > 1/√2  (normal phase unstable to fine superconducting layering) ;
    σ_ns = 0  exactly at  κ = 1/√2 .

## The upper critical / nucleation field

Linearizing the 1st GL equation about Ψ = 0 in a uniform field (Landau gauge) gives a
harmonic-oscillator eigenproblem; the largest field at which a superconducting nucleus first
appears from the normal state is

    H_{c2} = ℏc / (e* ξ²) ,        H_{c2}/H_c = √2 κ      ⟹   H_{c2} = √2 κ H_c .

H_{c2} = H_c precisely at κ = 1/√2, consistent with the surface-energy sign change: for
κ > 1/√2 there is a range H_c < H < H_{c2} where the normal phase is unstable to forming thin
superconducting layers. (A single flux quantum hc/e* threads the nucleus area 2πξ² at H_{c2}.)

## Worked numerical check (the landing artifact)

```python
import numpy as np
from scipy.integrate import solve_ivp, quad

# Static 1-D GL wall, zero field, lengths in units of the coherence length xi.
# First GL equation reduces to  -xi^2 f'' - f + f^3 = 0,  f real, 0 <= f <= 1,
# f -> 0 (normal), f -> 1 (superconductor).  In units x -> x/xi:  -f'' - f + f^3 = 0.
# First integral (x f' and integrate): f'^2 + f^2 - f^4/2 = C; deep inside f->1, f'->0
# => C = 1/2, so f'^2 = (1 - f^2)^2/2 and on the rising side  f'(x) = (1 - f^2)/sqrt(2).

def gl_wall(L=14.0, n=4001):
    x = np.linspace(-L, L, n)

    def rhs(x, f):
        return (1.0 - f[0] ** 2) / np.sqrt(2.0)        # GL first integral, rising branch

    f_left = np.tanh(-L / np.sqrt(2.0))                # analytic value at the left end as IC
    sol = solve_ivp(rhs, (-L, L), [f_left], t_eval=x, rtol=1e-11, atol=1e-13)
    assert sol.success, sol.message
    return x, sol.y[0]


if __name__ == "__main__":
    x, f = gl_wall()

    # (1) wall profile is f = tanh(x / sqrt(2))  (units of xi)
    f_exact = np.tanh(x / np.sqrt(2.0))
    m = np.abs(x) < 12.0
    print("max |f_num - tanh(x/sqrt2)| =", np.max(np.abs(f[m] - f_exact[m])))   # ~3e-5

    # (2) positive length carrying the small-kappa surface energy:
    #     sigma_ns = (Hc^2/8pi) * delta,  delta = int_0^inf (1 - f^4) dx = (4 sqrt2 / 3) xi.
    delta_num, _ = quad(lambda t: 1.0 - np.tanh(t / np.sqrt(2.0)) ** 4, 0, 60)
    print("delta/xi numeric  =", delta_num)                 # 1.885618...
    print("delta/xi analytic =", 4.0 * np.sqrt(2.0) / 3.0)  # 1.885618...  (= 4 sqrt2/3)
    print("=> sigma_ns = (Hc^2/8pi) * 1.886 xi > 0  (type-I, positive surface energy)")
```

Output: profile matches tanh(x/√2) to ≈3×10⁻⁵; ∫_0^∞(1 − tanh⁴(x/√2))dx = 1.885618 = 4√2/3,
confirming the positive type-I surface energy length.

## Summary of the causal chain

Spatially varying complex order parameter Ψ → Landau expansion fixes the |Ψ|²,|Ψ|⁴ terms and
ties α²/β to H_c² → gauge invariance forces the covariant kinetic term, which produces the
supercurrent → minimization gives the two GL equations + natural BC → coherence length ξ and
penetration depth δ, ratio κ = δ/ξ (T-independent) → positive surface energy for small κ;
sign flips at κ = 1/√2, the same κ where H_{c2} = √2 κ H_c crosses H_c, separating type-I
(κ < 1/√2) from type-II (κ > 1/√2).
