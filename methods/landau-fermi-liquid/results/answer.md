# The Theory of a Fermi Liquid

## Problem

A degenerate system of strongly interacting fermions (liquid helium-3, conduction electrons in a
metal) reproduces the *qualitative* low-temperature laws of an ideal Fermi gas — a linear-in-T heat
capacity above all — while renormalizing the prefactors. The interaction is of order the kinetic
energy, so the many-body problem is unsolvable, yet the gas-like behaviour persists. The task is a
low-temperature theory that reproduces the gas-like thermodynamics with measured constants, says
exactly which quantities are renormalized and by how much, fixes stability, and predicts new
collective phenomena — without solving the Schrödinger equation.

## Key idea

Turn the interaction on adiabatically from the gas. Assume the classification of levels is invariant
(valid until a phase transition opens a gap). Then each gas excitation maps to a **quasiparticle** —
fermionic, definite momentum, number equal to the real particles, long-lived near the Fermi surface
because the decay rate is Pauli-suppressed as ∝ ω². Because the dressing depends on the
surroundings, the energy is a **functional of the quasiparticle distribution**; its first variation
defines the quasiparticle energy and its second variation defines the **Landau interaction
function** f. All low-temperature physics — mass renormalization, compressibility, susceptibility,
stability, and a new collisionless sound — follows from f.

## The construction

**Energy functional.** With `dτ = d³p/(2πℏ)³` and the spin spur `Sp_σ`, the energy is a functional
of the distribution `n`, expanded to second order in the deviation `δn` from the ground-state Fermi
step:

```text
E - E0 = Sp_σ ∫ ε(p) δn dτ  +  (1/2) Sp_σ Sp_σ' ∫∫ f(p,p') δn δn' dτ dτ' + O(δ³)
ε(p)   = δE/δn(p)                          # quasiparticle energy (1st variation)
f(p,p')= δ²E/δn(p)δn(p')                   # Landau interaction function (2nd variation)
```

**Equilibrium and effective mass.** Maximizing the fermionic entropy
`S = -Sp_σ ∫ {n ln n + (1-n)ln(1-n)} dτ` at fixed `N`, `E` gives the Fermi form (now with ε a
functional of n), and the dispersion is linearized at the limiting momentum `p0`:

```text
n(ε) = [ exp((ε-μ)/θ) + 1 ]⁻¹
m*   = p / (∂ε/∂p) |_{p=p0}
```

**Landau function on the Fermi surface.** Only `f` with both momenta on the surface matters, so it
depends on the angle θ between p and p' and on spin; for spin ½, time-reversal + reflection allow

```text
f(p,p') = f^s(θ) + (σ·σ') f^a(θ),     f^{s,a}(θ) = Σ_l f_l^{s,a} P_l(cos θ)
F_l^{s,a} = D(ε_F) f_l^{s,a},         D(ε_F) = V m* p0 / (π² ℏ³)   # dimensionless Landau parameters
```

**Effective mass from Galilean invariance.** Momentum density = mass current,
`Sp ∫ p n dτ = Sp ∫ m (∂ε/∂p) n dτ`; varying in n and using `∂n/∂p = -(p/p)δ(p-p0)`:

```text
1/m = 1/m* + (p0 / 2(2πℏ)³) Sp_σ Sp_σ' ∫ f cos θ dΩ
⇒   m*/m = 1 + F_1^s / 3
```

(Valid where p is the true momentum, i.e. helium-3; not for crystal quasi-momentum in a metal.)

**Compressibility / first sound (l = 0 symmetric).** From `μ = ε(p0)` and propagating `δN` through
both the Fermi-level shift and the f-feedback:

```text
∂μ/∂N = Sp_σ Sp_σ' ∫ f do / (16π V) + (2πℏ)³ / (8π p0 m* V)
c²    = p0²/3m*² + (1/6m)(p0/2πℏ)³ Sp_σσ' ∫ f (1 - cos θ) do
⇒   c_s = (v_F*/√3) √[ (1 + F_0^s)(1 + F_1^s/3) ],   v_F* = p0/m*
    n² κ = D(ε_F) / (1 + F_0^s)
```

**Spin susceptibility (l = 0 antisymmetric).** A field both Zeeman-shifts ε and repolarizes n; the
spin part of f feeds back, giving a self-consistent effective moment γ:

```text
δε = -β(σ·H) + Sp_σ' ∫ f δn' dτ'  →  δε = -γ(σ·H)
1/χ = β⁻² { 2π²k²/(3α) + ψ̄0 }
⇒   χ/χ0 = (1 + F_1^s/3) / (1 + F_0^a)        # Stoner-like enhancement, F_0^a < 0 in He³
```

For helium-3 the exchange parameter is negative (about −2/3 of the heat-capacity term): the simple
gas relation between heat capacity and susceptibility no longer holds.

**Stability (Pomeranchuk).** The quadratic form `½ f δn δn` must be positive for every Fermi-surface
deformation:

```text
1 + F_l^{s,a} / (2l+1) > 0   for all l, both channels
```

A violation (e.g. F_1^s < −3 ⇒ m* < 0; F_0^s = −1 ⇒ κ → ∞) is a spontaneous deformation of the
Fermi surface — exactly where adiabatic continuity must fail.

**Kinetic equation and conservation laws.** Since ε depends on r through n(r), quasiparticles feel a
self-consistent force:

```text
∂n/∂t + (∂n/∂r)·(∂ε/∂p) - (∂n/∂p)·(∂ε/∂r) = I(n)
∂/∂t Sp∫ p_i n dτ + ∂Π_{ik}/∂x_k = 0,   Π_{ik} = Sp∫ p_i (∂ε/∂p_k) n dτ + δ_{ik}[Sp∫ ε n dτ - E]
∂E/∂t + div Q = 0,                       Q = Sp∫ n ε (∂ε/∂p) dτ
```

**Zero sound (the new mode).** At T = 0 the collision rate ∝ T² vanishes, so I(n) = 0; ordinary
sound (needing ωτ ≪ 1) is killed by the diverging mean free path, but a collisionless,
self-consistent oscillation of the *shape* of the Fermi surface survives. Linearizing
`δn, δε ∝ e^{i(k·r-ωt)}` with `η = u/v`, `v = ∂ε0/∂p`, surface displacement `ν(p̂)`, and
`F(χ) = Sp_σ' f · 4πp²dp/(2πℏ)³ dε`:

```text
(η - cos θ) ν(θ,φ) = cos θ ∫ F(χ) ν(θ',φ') do'/4π
```

For `F = F0` (constant): `ν ∝ cos θ/(η - cos θ)`, and the dispersion is the transcendental equation

```text
φ(η) ≡ (η/2) ln[(η+1)/(η-1)] - 1 = 1/F0
```

- Real undamped mode requires η > 1 (u > v): the wave outruns the quasiparticles, escaping Landau
  damping. φ(η) decreases monotonically from +∞ to 0, so a root exists only for F0 > 0 (repulsive).
- Strong coupling F0 → ∞: φ ≈ 1/3η² ⇒ η = √(F0/3). Weak coupling F0 → 0⁺: η - 1 ~ exp(-2 - 2/F0).
- Zero sound deforms the Fermi surface (ν ∝ cos θ/(η-cos θ)); ordinary sound is a rigid shift
  (ν ∝ cos θ). Zero sound is faster: u > c ≈ v/√3.
- Asymmetric (m = ±1) zero sound from `F = F0 + F1 cos χ` requires F1 > 6:
  `∫₀^π sin³θ cos θ/(η-cos θ) dθ = 4/F1`, the left side ≤ 2/3 at η = 1.
- Spin waves: with `K = ½F(χ) + ½G(χ)(σ·σ')`, the same equation holds with F → G/4.

**Helium-3 numbers.** Read parameters off measurements via `F1/3 = m*/m - 1`,
`F0 = 3 m m* c²/p0² - 1`. With m* ≈ 1.43 m, ordinary sound c ≈ 195 m/s, p0/ℏ ≈ 0.76×10⁸ cm⁻¹:

```text
F0 ≈ 5.4,   F1 ≈ 1.3
φ(η) = 1/F0 ⇒ η ≈ 1.83  ⇒  zero-sound speed u = η v = 1.83 p0/m* ≈ 206 m/s   (> c, as required)
```

## Why it works

The theory replaces the intractable many-body wavefunction with a *functional of the quasiparticle
distribution*. The first variation is the renormalized single-particle spectrum (gas thermodynamics
with m → m*); the second variation, the Landau function f, packages the entire residual interaction
into a few dimensionless numbers F_l^{s,a} on the Fermi surface. Every static response is the gas
answer renormalized by one symmetry-selected Landau parameter; the same f provides the molecular
field that, in the collisionless limit, sustains zero sound — a qualitatively new mode with no
hydrodynamic analogue. Stability and the limits of validity are read off the same parameters
(1 + F_l/(2l+1) > 0), and the breakdown points are precisely the phase transitions where adiabatic
continuity fails.
