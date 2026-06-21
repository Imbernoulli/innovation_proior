# The Coleman-Weinberg potential: radiative corrections as the origin of spontaneous symmetry breaking

## The problem

A classically scale-invariant theory — a massless charged scalar in electrodynamics, with couplings `e` and `λ` and no dimensionful constant — has a classical potential `V_cl = (λ/4!)φ⁴` whose only minimum is the symmetric point `φ=0`. The tree analysis says: no symmetry breaking. But the tree potential ignores the zero-point energy of the field fluctuations, which depends on the background. The question is whether the *quantum* corrections, with no asymmetric input, move the true vacuum off the origin.

## The key idea

Work with the **effective potential** `V(φc)` — the non-derivative part of the effective action `Γ[φc] = W[J] − ∫Jφc` (Legendre transform of Schwinger's generating functional, after Jona-Lasinio). Its minima are the true quantum vacua; spontaneous breaking means the minimum is at `φc = ⟨φ⟩ ≠ 0`. Crucially `V` surveys all candidate vacua at once, and its `n`-th derivative is the sum of all 1PI graphs with `n` zero-momentum external legs, so tree-level `V = V_cl` and every quantum effect is in the closed-loop graphs.

Truncate by the **loop expansion**, not by powers of the coupling: putting `a` in front of `L` (`L → a⁻¹L`), a graph carries `a^{I−V} = a^{L−1}`, so loop order is purely topological — invariant under field shifts and under how `L` is split into free + interacting. That invariance is what makes `V` evaluable impartially at every `φc`.

At one loop, the sum of polygon graphs resums into a single integral; renormalizing the coupling at an *arbitrary* off-singularity scale `M` (the fourth derivative at the origin doesn't exist — log infrared singularity), the cutoff drops out. In a theory with a *single* coupling the loop-induced minimum requires `ln(φc/M) ∼ 1/λ`, i.e. `φc/M ∼ e^{1/λ}`, outside one-loop validity — an artifact. With a *second* independent coupling, the loop term balances the tree term at comparable magnitude with **no large logarithm**, so the minimum is genuine. The minimum condition then *determines* the quartic coupling in terms of the gauge coupling — a dimensionless coupling is traded for the dimensionful vacuum value `⟨φ⟩` (**dimensional transmutation**).

## The derivation

**Effective-potential machinery.** With `e^{iW[J]} = ⟨0⁺|0⁻⟩_J`, `φc = δW/δJ`, and `Γ = W − ∫Jφc`, one has `δΓ/δφc = −J`; for a translation-invariant vacuum the condition `δΓ/δφc=0` at `J=0` is `dV/dφc = 0`, and stability requires a minimum.

**Self-interacting scalar (template).** `L = ½(∂φ)² − (λ/4!)φ⁴ + ½A(∂φ)² − ½Bφ² − (1/4!)Cφ⁴`. The one-loop polygon graphs give

```
V = (λ/4!)φc⁴ + ½Bφc² − (1/4!)Cφc⁴ + i∫(d⁴k/(2π)⁴) Σ_{n≥1} (1/2n)[½λφc²/(k²+iε)]ⁿ .
```

Summing `Σ xⁿ/2n = −½ln(1−x)` and Wick-rotating turns the (term-by-term divergent) series into `½∫d⁴k_E/(2π)⁴ ln(1 + λφc²/2k_E²)` — the per-graph infrared divergences become a harmless log singularity at `φc=0`. With a cutoff `k²=Λ²`,

```
V = (λ/4!)φc⁴ + ½Bφc² − (1/4!)Cφc⁴ + λΛ²φc²/64π² + (λ²φc⁴/256π²)[ln(λφc²/2Λ²) − ½] .
```

Impose `d²V/dφc²|₀ = 0` (renormalized mass zero → fixes `B`, absorbs the `Λ²`) and `d⁴V/dφc⁴|_{φc=M} = λ` (coupling defined at the arbitrary scale `M`, since the fourth derivative at the origin doesn't exist → fixes `C`). The cutoff cancels and

```
V = (λ/4!)φc⁴ + (λ²φc⁴/256π²)[ ln(φc²/M²) − 25/6 ] .       (single scalar)
```

Changing `M → M'` is compensated by `λ' = λ + (3λ²/32π²)ln(M'²/M²)`: a reparametrization. The loop-induced minimum here needs `ln(φc/M) ∼ 1/λ` → rejected as outside the one-loop domain.

**Massless scalar electrodynamics.** `L = −¼F² + ½(∂φ₁−eAφ₂)² + ½(∂φ₂+eAφ₁)² − (λ/4!)(φ₁²+φ₂²)² + c.t.` In Landau gauge only three polygon classes survive (`φ₁`, `φ₂`, photon). The radial mode has mass² `λφc²/2`, the would-be Goldstone `λφc²/6`, the photon `e²φc²`:

```
V = (λ/4!)φc⁴ + (5λ²/1152π² + 3e⁴/64π²) φc⁴ [ ln(φc²/M²) − 25/6 ] .
```

Renormalizability forces `λ ≳ O(e⁴)` (the quartic cancels the order-`e⁴` Coulomb-scattering divergence); then `5λ²/1152π² ∼ e⁸` is dropped:

```
V = (λ/4!)φc⁴ + (3e⁴/64π²) φc⁴ [ ln(φc²/M²) − 25/6 ] .
```

Minimum (`dV/dφc=0`, choose `M=⟨φ⟩` so the log vanishes at the vacuum):

```
λ = 33 e⁴ / 8π² .          (dimensional transmutation: λ determined by e)
```

Final potential, parametrized by `e` and `⟨φ⟩` alone:

```
V = (3e⁴/64π²) φc⁴ [ ln(φc²/⟨φ⟩²) − ½ ] .
```

**Spectrum.** Scalar mass from the curvature, vector mass from the eaten Goldstone (minimal coupling):

```
m²(S) = V''(⟨φ⟩) = 3e⁴⟨φ⟩²/8π² ,   m²(V) = e²⟨φ⟩² ,
m²(S)/m²(V) = 3e²/8π² = (3/2π)(e²/4π) ,   V(⟨φ⟩) = −3e⁴⟨φ⟩⁴/128π² < 0 .
```

A small mass *ratio* from a small coupling, and a vacuum strictly below the symmetric one.

**Renormalization-group control.** With `t = ln(φc/M)` and one-loop `β̄ = 3λ²/16π²` (scalar; `γ̄=0`), the running coupling `λ'(t) = λ/(1 − 3λt/16π²)` stays small for all `t<0`, so the improved `V⁗ = λ'(t)` is reliable down to the origin and confirms the origin is a genuine maximum (not a hidden minimum); `V(0)=0` is fixed exactly. For scalar QED, `γ̄ = 3e²/16π²`, `e'² = e²/(1−e²t/24π²)`; the RG removes the `λ ∼ O(e⁴)` restriction — breaking occurs for arbitrary small `e, λ`.

**All-orders structure (why the one-loop surprises are not accidents).** Classify graphs by vertex type; `L = ½V₃ + V₄ + 1`, so for fixed `L` only finitely many prototype graphs (no type-2 vertices) exist, and all others come from inserting type-2 vertices. Summing the insertions is a geometric series that shifts every internal propagator `i/(k²+iε) → i/(k² − ½λφc² + iε)` (and `e²φc²` for photon lines): the infrared divergences become a singularity only at `φc=0`, to all orders. Rescaling loop momenta `k = λ^{1/2}k'` shows the `L`-loop contribution is `∝ φc⁴ f(φc/M) λ^{L+1}` — a pure power, no `ln λ`, to all orders (legitimate because the coupling is subtracted at a field-space point, not a fixed momentum); for scalar QED, `∝ (e²)^{L+1}`.

**Massive limits (consistency).** Taking a mass `μ → 0` reproduces the massless `V` smoothly. For *negative* `μ²`, `V` develops an imaginary part `Im V = −(1/64π²)[(μ²+½λφc²)²θ(−μ²−½λφc²) − μ⁴]`, uncancellable by the real counterterms — the field-theory analogue of the Euler-Heisenberg/Schwinger imaginary part for a constant electric field, signalling a kinematically unstable (symmetric) vacuum; it vanishes at the real asymmetric vacuum. Renormalization condition becomes `Re V''|₀ = μ²`.

## The general one-loop master formula

For any massless renormalizable gauge theory (scalars, fermions, gauge bosons), in Landau gauge, the one-loop effective potential is the trace-log of the field-dependent fluctuation operator:

```
V₁(φc) = (1/64π²) [ Tr W²(φc) ln W(φc)  +  3 Tr M⁴(φc) ln M²(φc)  −  Tr (mm†)²(φc) ln (mm†)(φc) ] ,
```

with `W = ∂²V₀/∂φ_a∂φ_b` (scalar mass² matrix), `M²_ab = g_a g_b (T_a φ, T_b φ)` (vector mass² matrix from the gauge generators), `mm†` (fermion mass² from the Yukawa matrix). Each field contributes `± (2J+1)/(64π²) · (mass²)² ln(mass²)`: `+` for bosons, `−` for fermions, weighted by polarizations (the `3` is the Landau-gauge vector count). Massless scalar QED is the special case `W = {λφc²/2, λφc²/6}`, `M² = e²φc²`.

Applied to an `SU(2)` gauge triplet + scalar isovector: radiative breaking, an unbroken `U(1)` (massless photon) survives, charged vectors eat scalars, `m²(S)/m²(V) = (3/π)(e²/4π)` (twice Abelian), `λ = 33e⁴/4π²`. Applied to the `SU(2)×U(1)` lepton model: `m²(S) = (3/32π²)[2g²m²(W) + (g²+g'²)m²(Z)]` — the scalar mass predicted from the vector masses.

## Worked verification

```python
import sympy as sp
phi, M, lam, e, vev = sp.symbols('phi M lambda e v', positive=True)
pi = sp.pi

# Massless scalar electrodynamics, one loop (scalar-loop term dropped at order e^8):
V = lam/sp.factorial(4)*phi**4 + 3*e**4/(64*pi**2)*phi**4*(sp.log(phi**2/M**2) - sp.Rational(25,6))

# Minimum with M = <phi>: the log vanishes, giving lambda(e):
lam_min = sp.solve(sp.Eq(lam/6 - 11*e**4/(16*pi**2), 0), lam)[0]
assert lam_min == 33*e**4/(8*pi**2)

# Final potential parametrized by e and <phi>:
V_final = sp.simplify(V.subs({lam: lam_min, M: vev}))
assert sp.simplify(V_final - 3*e**4/(64*pi**2)*phi**4*(sp.log(phi**2/vev**2) - sp.Rational(1,2))) == 0

# Spectrum at the broken vacuum:
mS2   = sp.simplify(sp.diff(V_final, phi, 2).subs(phi, vev))   # 3 e^4 v^2 / (8 pi^2)
mV2   = e**2*vev**2                                            # e^2 v^2
ratio = sp.simplify(mS2/mV2)                                   # 3 e^2 / (8 pi^2)
depth = sp.simplify(V_final.subs(phi, vev))                   # -3 e^4 v^4 / (128 pi^2)
assert ratio == 3*e**2/(8*pi**2)               # = (3/2pi)(e^2/4pi)
assert depth == -3*e**4*vev**4/(128*pi**2)

# Cross-check coefficients against the master formula (1/64pi^2) Sigma (mass^2)^2:
assert sp.simplify(sp.Rational(1,64)/pi**2*((lam*phi**2/2)**2 + (lam*phi**2/6)**2)
                   - 5*lam**2/(1152*pi**2)*phi**4) == 0     # scalar loops -> 5 lam^2/1152pi^2
assert sp.simplify(3*sp.Rational(1,64)/pi**2*(e**2*phi**2)**2
                   - 3*e**4/(64*pi**2)*phi**4) == 0          # photon loop -> 3 e^4/64pi^2
```
