# B3 / B3LYP: Hybrid Exchange-Correlation by Mixing Exact Exchange via the Adiabatic Connection

## Problem

Kohn–Sham DFT with semilocal (LSDA + gradient) exchange-correlation functionals
gives good but not chemically accurate thermochemistry: a *systematic*
overbinding, worst for non-hydride molecules, that does not shrink with further
density-and-gradient refinement. The goal is to reach ~2 kcal/mol average error
on standard thermochemical tests without leaving the cheap Kohn–Sham framework.

## Key idea

Read the exact exchange-correlation energy as the adiabatic-connection
(coupling-strength) integral

  E_xc = ∫₀¹ U_xc^λ dλ,

where λ scales the electron–electron repulsion (0 = non-interacting Kohn–Sham
reference, 1 = real system), the density being held fixed at its physical value.
The integrand's λ = 0 end is *exact exchange* — the exchange energy of the
Kohn–Sham determinant, with no correlation. Local/semilocal functionals replace
the integrand by a uniform-gas model at *every* λ, including λ = 0, where the
gas hole "follows" its electron and so manufactures fake left–right correlation
(visible in H₂, whose exact exchange hole is static). This λ = 0-end error is the
structural overbinding. The cure: inject a fraction of *exact* exchange at the
weak-coupling end, while keeping the semilocal model at strong coupling (where its
following hole correctly mimics correlation and the X–C error cancellation lives).

Modeling the integrand's correction as decaying like (1 − λ)^{n−1} gives, after
integration, an exact-exchange fraction a₀ = 1/n; for ordinary molecules
(MP4-accurate) n ≈ 4, so a₀ ≈ 1/4. Mixing in exact exchange also removes a
fraction of the one-electron self-interaction error of semilocal exchange; the
fraction must stay small (~0.2) to preserve the static-correlation imitation that
makes semilocal X–C cancel.

## The functional

**Original three-parameter hybrid (B3), based on LSDA plus gradient corrections:**

  E_xc = E_xc^{LSDA} + a₀ (E_x^exact − E_x^{LSDA}) + a_x ΔE_x^{B88} + a_c ΔE_c^{PW91}.

- E_xc^{LSDA}: uniform-gas exchange-correlation (base term — guarantees the exact
  uniform-gas limit, since all corrections vanish for constant density).
- a₀ (E_x^exact − E_x^{LSDA}): swaps a fraction a₀ of LSDA exchange for exact
  (Fock) exchange, fixing the λ = 0 end.
- ΔE_x^{B88}: Becke-1988 gradient correction to LSDA exchange.
- ΔE_c^{PW91}: Perdew–Wang gradient correction to LSDA correlation.

Fitting the three coefficients by linear least-squares to a thermochemical set
(56 atomization energies, 42 ionization potentials, 8 proton affinities, 10
first-row total atomic energies) gives

  **a₀ = 0.20,  a_x = 0.72,  a_c = 0.81.**

a₀ = 0.20 ≈ 1/4 as the integrand argument predicts; a_x = 0.72 < 1 because exact
exchange already supplies part of the exchange physics; a_c = 0.81.

**B3LYP** replaces the correlation *gradient correction* by the *complete*
Lee–Yang–Parr correlation functional (a Colle–Salvetti-derived functional, not
built on the uniform gas), mixed with LSDA correlation in the proportion
c : (1 − c). Collecting the LSDA-exchange coefficient (1 − a − b), the functional
in terms of complete pieces is

  **E_xc^{B3LYP} = a E_x^exact + b E_x^{B88} + (1 − a − b) E_x^{LSDA}
                 + c E_c^{LYP} + (1 − c) E_c^{LSDA},**

with **a = 0.20, b = 0.72, c = 0.81** (so the LSDA-exchange weight is
1 − a − b = 0.08 and the LSDA-correlation weight is 1 − c = 0.19). The LSDA
correlation is a Vosko–Wilk–Nusair uniform-gas parametrization (historically the
RPA-based VWN variant).

Ingredient forms used:
- LSDA exchange: E_x^{LSDA} = −(3/4)(3/π)^{1/3} ∫ (n↑^{4/3} + n↓^{4/3}) dr.
- B88 exchange: ΔE_x^{B88} = −β ∫ Σ_σ n_σ^{4/3} x_σ²/(1 + 6β x_σ sinh⁻¹ x_σ) dr,
  x_σ = |∇n_σ|/n_σ^{4/3}, β = 0.0042.
- LYP correlation: complete functional E_c^{LYP}(n↑, n↓, ∇n↑, ∇n↓) from the
  Colle–Salvetti formula (constants a = 0.04918, b = 0.132, c = 0.2533,
  d = 0.349).
- Exact exchange: E_x^exact = −½ Σ_σ Σ_{i,j} ∫∫ φ_iσ*(r₁)φ_jσ(r₁)
  φ_jσ*(r₂)φ_iσ(r₂)/|r₁ − r₂| dr₁dr₂ (nonlocal; solved in the generalized
  Kohn–Sham scheme).

## Worked recipe (the mixing a code performs)

On a converged density n_σ and Kohn–Sham orbitals, B3LYP assembles the XC energy
from precomputed standard ingredients with fixed weights — the exact-exchange
weight being the only term that introduces nonlocality:

```python
# B3LYP = 0.2*HF + 0.08*Slater + 0.72*B88  (exchange)
#       + 0.81*LYP + 0.19*VWN              (correlation)
# weights: a=0.20 exact exchange, b=0.72 B88, 1-a-b=0.08 LSDA exchange;
#          c=0.81 LYP, 1-c=0.19 LSDA(VWN) correlation.
a, b, c = 0.20, 0.72, 0.81

def E_xc_B3LYP(n_up, n_dn, grad_up, grad_dn, orbitals, grid):
    # exchange:  a*HF + b*B88 + (1-a-b)*Slater
    Ex = ( a       * E_x_exact(orbitals, grid)                       # nonlocal Fock exchange
         + b       * E_x_B88(n_up, n_dn, grad_up, grad_dn, grid)     # complete B88 exchange
         + (1-a-b) * E_x_LSDA(n_up, n_dn, grid) )                    # Slater (uniform-gas) exchange
    # correlation:  c*LYP + (1-c)*LSDA(VWN)
    Ec = ( c       * E_c_LYP(n_up, n_dn, grad_up, grad_dn, grid)     # complete LYP correlation
         + (1-c)   * E_c_LSDA(n_up, n_dn, grid) )                    # uniform-gas (VWN) correlation
    return Ex + Ec

def total_energy(system):
    n_up, n_dn, grad_up, grad_dn, orbitals, grid = system.density_and_orbitals()
    return system.core_energy() + system.hartree_energy() \
         + E_xc_B3LYP(n_up, n_dn, grad_up, grad_dn, orbitals, grid)
```

## Special cases

- a₀ = ½, a_x = a_c = 0: the half-and-half (linear-integrand / two-point)
  approximation — but now with the LSDA base it also keeps the full gas-limit
  correlation that half-and-half dropped.
- a₀ = 0, a_x = a_c = 1: pure gradient-corrected (B88 + GGA) DFT.
- a₀ = 1, a_x = a_c = 0: full exact exchange plus a correlation functional (the
  n = 1, exchange-dominated case).
- One-parameter hybrid: set a_x = 1 − a₀, a_c = 1, giving
  E_xc = E_xc^{DFA} + a₀(E_x^exact − E_x^{DFA}) with a₀ ≈ 0.25.

## Why it works

The exact-exchange admixture repairs the weak-coupling (λ = 0) end of the
adiabatic-connection integrand, removing the structural overbinding and a fraction
of the self-interaction error; basing the functional on LSDA plus
gradient corrections that vanish for the uniform gas preserves the exact gas
limit; and three independent coefficients let it fit atomization energies,
ionization potentials, and proton affinities simultaneously — reaching average
errors near 2 kcal/mol, the chemical-accuracy target previously met only by far
more expensive composite *ab initio* procedures.
