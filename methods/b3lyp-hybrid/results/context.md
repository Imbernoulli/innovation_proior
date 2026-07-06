# Context: Pushing Kohn–Sham Density-Functional Thermochemistry Toward Chemical Accuracy

## Research question

For a molecule with nuclei clamped at fixed positions (Born–Oppenheimer), the
quantity chemists want is the total electronic energy and its differences:
atomization energies, ionization potentials, proton affinities, reaction
barriers. The accepted target — "chemical accuracy" — is an average error of
about 2 kcal/mol (≈ 0.1 eV, ≈ 10 kJ/mol) against experiment, the threshold below
which computed thermochemistry becomes predictively useful. Expensive composite
*ab initio* wavefunction recipes (the Gaussian-2 procedure of Pople and
co-workers) reach this on small first- and second-row systems, but at a cost that
scales steeply with system size.

Kohn–Sham density-functional theory (KS-DFT) is enormously cheaper: it solves
self-consistent single-particle equations and needs only an approximate
exchange-correlation (XC) energy functional E_xc[n]. In Hartree atomic units
(ℏ = m = e = 1) the KS equations are

  (−½∇² + v_eff) φ_i = ε_i φ_i,  n(r) = Σ_i |φ_i(r)|²,
  v_eff = v(r) + ∫ n(r′)/|r − r′| dr′ + v_xc(r),  v_xc = δE_xc/δn,

and the total energy follows once E_xc[n] is specified. The local-spin-density and
gradient-corrected approximations for E_xc made KS-DFT a workhorse for
thermochemistry — but a stubborn, *systematic* error remained, and it did not
shrink with further refinement of the density-and-gradient model. The question:
is there a structural reason these functionals miss chemical accuracy, and is
there an extra ingredient — knowable from the exact theory — that would remove the
error without leaving the cheap KS framework?

## Background

**The exact XC energy is a coupling-constant integral (the adiabatic
connection).** A central, rigorous result of KS-DFT expresses E_xc not as a
single object but as an integral over an interaction-strength parameter λ. Scale
the electron–electron repulsion as λ/|r − r′| with 0 ≤ λ ≤ 1, and at each λ adjust
the external potential v_λ(r) so that the density stays pinned at its *physical*
(λ = 1) value n(r). At λ = 0 the electrons are non-interacting (the KS reference
system); at λ = 1 they are the real, fully interacting system; in between is a
continuum of partially interacting systems all sharing the same n. Differentiating
the energy with respect to λ (Hellmann–Feynman) and integrating gives

  E_xc = ∫₀¹ U_xc^λ dλ,

where U_xc^λ is the XC *potential* energy at coupling λ. Although the integrand
refers only to potential energy, the λ-integration generates the kinetic part of
the correlation energy as well. This formula was developed by Harris & Jones
(1974), Gunnarsson & Lundqvist (1976), and Langreth & Perdew (1977); a short
first-order-perturbation-theory derivation is standard. The two ends of the
integrand are physically different objects: at λ = 0, U_xc^λ is the pure exchange
energy of the Slater determinant of the KS orbitals (no correlation at all);
at λ = 1 it is the deep, localized XC hole of the fully interacting system.

**The XC hole and the sum rule.** E_xc can be written as the electrostatic
interaction of each electron with its coupling-averaged XC hole n̄_xc(r, r′):
E_xc = ½ ∫∫ n(r) n̄_xc(r, r′)/|r − r′| dr dr′. The hole obeys the exact sum rule
∫ n̄_xc(r, r′) dr′ = −1 ("one electron is missing"). The reason local
approximations work as well as they do is that E_xc depends mainly on the
spherical average and the normalization of the hole, not its detailed shape.

**Local and semilocal approximations and their diagnostic failure.** The
local-spin-density approximation (LSDA) replaces U_xc^λ at *every* λ by the
corresponding uniform-electron-gas value, with the exchange part known
analytically,

  E_x^LSDA = −(3/4)(3/π)^{1/3} ∫ (n↑^{4/3} + n↓^{4/3}) dr,

and the correlation part taken from uniform-gas parametrizations (Vosko–Wilk–Nusair
1980; Perdew–Wang 1992). LSDA exhibits a well-documented, systematic *overbinding*:
atomization energies come out too large. The diagnostic reason is visible in the
simplest molecule, H₂: its exact (restricted) exchange hole is the negative of the
σ_g orbital density, static and reference-point independent, implying a complete
*absence* of left–right correlation in the exchange limit. The uniform-gas model
hole, by contrast, is reference-point-centered, relatively localized, and
"follows" its reference electron around — a crude simulation of left–right
correlation. That simulated correlation is *desirable* in the strongly
interacting (large-λ) regime but *misrepresents* the non-interacting (λ = 0)
limit. Gunnarsson & Jones (1985) further observed that DFT energy differences err
conspicuously when orbital nodes are created or lost (e.g. occupying antibonding
orbitals), and that local models describe such exchange-energy differences
poorly. Both pathologies are sharpest at λ = 0.

**One-electron self-interaction.** For a one-electron density the Hartree energy
E_H[n] must be cancelled exactly by exchange; Hartree–Fock (HF) exchange does this
by construction, but any semilocal E_x leaves a residual self-interaction error.
This error favors over-delocalized densities and over-stabilizes stretched bonds
and charge-transfer states.

## Baselines

**LSDA exchange-correlation.** E_xc^LSDA from the uniform gas (analytic exchange
above; VWN/PW correlation). Cheap, exact for the uniform gas. Limitation: the
systematic overbinding described above; the uniform-gas hole is the wrong object
near the λ = 0 exchange limit, and the overbinding does not vanish with finer
gas parametrizations.

**Gradient-corrected exchange (Becke 1988, "B88").** Adds a reduced-gradient term
to LSDA exchange,

  ΔE_x^B88 = −β ∫ Σ_σ n_σ^{4/3} x_σ²/(1 + 6β x_σ sinh⁻¹ x_σ) dr,
  x_σ = |∇n_σ|/n_σ^{4/3},  β = 0.0042,

constructed to be quadratic in x_σ at small gradient (matching the gradient
expansion) and to recover the correct −n/2r asymptotic decay of the exchange
energy density at large r in atoms; the single parameter β was fit to noble-gas HF
exchange. Limitation: exchange-only B88 gives extremely poor ionization potentials
and, paired with a correlation gradient correction, still overbinds non-hydride
molecules.

**Gradient-corrected correlation (Perdew–Wang 1991; Lee–Yang–Parr 1988).**
ΔE_c^PW91 is a |∇n|-dependent correction to LSDA correlation. The Lee–Yang–Parr
(LYP) functional is a *complete* correlation functional (not built on LDA): it
descends from the Colle–Salvetti (1975) formula for the correlation energy in
terms of the curvature of the HF hole, turned into a density functional by a
gradient expansion,

  E_c^LYP = −a ∫ [ (n↑n↓ + …) / (1 + d n^{−1/3}) … ] dr,
  a = 0.04918, b = 0.132, c = 0.2533, d = 0.349,

with constants fit to helium. Combined B88 + GGA-correlation defines the best
purely semilocal DFT of the day, reaching ~5–6 kcal/mol average error on standard
atomization tests. Limitation: a residual, structural overbinding of non-hydrides
that resists further density-and-gradient refinement; chemical accuracy stays out
of reach for any functional built only from n and ∇n.

**Hartree–Fock exchange.** The exchange energy of a Slater determinant,

  E_x^HF = −½ Σ_σ Σ_{i,j} ∫∫ φ_iσ*(r₁)φ_jσ(r₁)φ_jσ*(r₂)φ_iσ(r₂)/|r₁ − r₂| dr₁dr₂,

is exact exchange and exactly self-interaction free, but contains *no* correlation
and produces a *nonlocal* exchange operator. HF underbinds badly. So HF and LSDA
err in opposite directions; neither alone is satisfactory.

**A half-and-half attempt.** A first effort to bring exact exchange into KS-DFT
approximates the coupling-strength integral as the average of its two ends —
exact exchange at λ = 0 and the LSDA XC potential energy at λ = 1,
E_xc ≈ ½(E_x^exact + E_xc^{LSDA}). On standard atomization tests it performs about
as well as gradient-corrected DFT (~6.5 kcal/mol). Limitation: total energies come
out poor; the uniform-electron-gas limit is *not* recovered (a formal failure — at
constant density the theory must reduce to the gas energy, and this fixed-average
form does not); and ionization potentials and proton affinities are poor. It does
atomization energies and little else, with no remaining freedom to retune.

## Evaluation settings

The natural yardstick is the Gaussian-1 (G1) database of Pople and co-workers (and
its extension): 56 molecular atomization energies, 42 ionization potentials, 8
proton affinities, and 10 first-row total atomic energies of first- and second-row
systems, with experimental (and exact) reference values. Energies are reported in
kcal/mol (atomization, proton affinities), eV (ionization potentials), and Hartree
(total atomic energies). Quality is summarized by average absolute deviation and
maximum absolute deviation from experiment. The standard of comparison is the
composite Gaussian-2 procedure (≈ 1.2 kcal/mol average on atomization) and the
prior gradient-corrected DFT (≈ 5.7 kcal/mol average). Electron affinities are
excluded because the LSDA XC potential does not bind negative ions. The
computation uses the basis-set-free, post-LSDA numerical procedure established for
this functional series.

## Code framework

The pre-method scaffold: a KS/post-LSDA engine already provides, on a given
converged density and set of orbitals, the standard energy components. We assemble
the XC energy as a combination of these known ingredients; the combination itself
is the open slot.

```python
import numpy as np

# --- known, pre-existing ingredients evaluated on the density / KS orbitals ---
def E_x_LSDA(n_up, n_dn, grid):
    """Uniform-gas (Slater) exchange energy.  -(3/4)(3/pi)^(1/3) ∫ (n_up^4/3 + n_dn^4/3)."""
    c = -(3.0/4.0)*(3.0/np.pi)**(1.0/3.0)
    return c*np.sum((n_up**(4.0/3.0) + n_dn**(4.0/3.0))*grid.w)

def E_c_LSDA(n_up, n_dn, grid):
    """Uniform-gas correlation energy (VWN / PW parametrization)."""
    ...  # standard uniform-gas correlation

def dE_x_B88(n_up, n_dn, grad_up, grad_dn, grid):
    """Becke-1988 gradient correction to LSDA exchange (the GGA exchange piece)."""
    ...  # -beta * sum_sigma ∫ n_s^4/3 x_s^2 / (1 + 6 beta x_s asinh(x_s))

def E_x_B88(n_up, n_dn, grad_up, grad_dn, grid):
    """Complete B88 exchange energy = E_x_LSDA + dE_x_B88."""
    ...

def dE_c_GGA(n_up, n_dn, grad_up, grad_dn, grid):
    """A gradient correction to LSDA correlation (the GGA correlation piece)."""
    ...

def E_c_LYP(n_up, n_dn, grad_up, grad_dn, grid):
    """A complete (non-LDA-based, Colle-Salvetti-derived) correlation energy."""
    ...

def E_x_exact(orbitals, grid):
    """Exchange energy of the Slater determinant of the orbitals: a (nonlocal)
    Fock exchange integral over the occupied orbitals."""
    ...

# --- the slot the method occupies ---
def E_xc(n_up, n_dn, grad_up, grad_dn, orbitals, grid):
    """TODO: combine the available exchange and correlation ingredients into the
    exchange-correlation energy we will design."""
    raise NotImplementedError

# --- driver (unchanged): plug E_xc into the post-LSDA energy evaluation ---
def total_energy(system):
    n_up, n_dn, grad_up, grad_dn, orbitals, grid = system.density_and_orbitals()
    return system.core_energy() + system.hartree_energy() \
         + E_xc(n_up, n_dn, grad_up, grad_dn, orbitals, grid)
```
