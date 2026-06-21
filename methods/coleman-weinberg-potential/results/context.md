# Context: a massless theory with no scale, and the question of where a vacuum could come from

## Research question

Take the electrodynamics of a massless charged scalar field — a complex scalar `φ` minimally coupled to the photon, with the usual quartic self-interaction but **no mass term at all**. Written in terms of two real fields `φ₁, φ₂`, the Lagrange density is

```
L = −¼ Fμν² + ½(∂μφ₁ − eAμφ₂)² + ½(∂μφ₂ + eAμφ₁)² − (λ/4!)(φ₁²+φ₂²)² .
```

This theory has two free dimensionless parameters, the gauge coupling `e` and the quartic coupling `λ`, and **not a single dimensionful constant**. Classically it is exactly scale-invariant.

The classical potential is `V_cl = (λ/4!)(φ₁²+φ₂²)²`. For `λ>0` its one and only stationary point is the origin `φ=0`, which is a minimum: the symmetric configuration. So at the classical (tree) level there is no spontaneous symmetry breaking — the field sits at zero, the scalar and the photon are both massless, and the U(1) symmetry is manifest.

The precise question: **the classical analysis says the symmetric point is the vacuum, working only from the classical potential. What is the true vacuum of the full quantum theory — the one that includes the quantum fluctuations of the fields — and what spectrum (the photon's fate, the scalar's mass) results?** The setting carries no dimensionful constant and no negative mass-squared; the determination must be a computation controlled by the small couplings `e` and `λ`.

## Background

**Mass is the curvature of the potential.** Given a potential `V(φ)`, the mass-squared of small oscillations about a configuration `φ₀` is `V''(φ₀)`, the local curvature. A free scalar `L = ½(∂φ)² − ½m²φ²` has `mass² = m²`; vanishing curvature means a flat direction and a massless mode. So "what are the masses" is the same question as "what is the shape of the potential, and about which point do we expand."

**Spontaneous symmetry breaking and the Goldstone theorem.** A theory whose Lagrangian is exactly invariant under a continuous symmetry can nonetheless have a ground state that is not invariant — the symmetry is *hidden in the vacuum*. The standard mechanism (Goldstone 1961; Nambu; Goldstone, Salam & Weinberg 1962) is a "Mexican-hat" potential `V = −μ²|φ|² + λ|φ|⁴` with `μ²>0`: the origin is a local maximum, and a whole circle of degenerate minima sits at `|φ|² = ...`. Picking one breaks the U(1) by the choice of vacuum. The Goldstone theorem then says that for every broken continuous generator there is a massless spin-0 boson — the flat (angular) direction along the trough. In this mechanism the symmetry breaking is driven by the negative mass-squared term `−μ²|φ|²` written into the Lagrangian.

**The Higgs mechanism / Abelian Higgs model.** When the broken symmetry is *gauged*, the would-be Goldstone boson is not physical: it is absorbed (eaten) by the gauge field, which acquires a mass and a longitudinal polarization (Englert & Brout 1964; Higgs 1964; Guralnik, Hagen & Kibble 1964; Kibble 1967). In the Abelian model — a charged scalar with a Mexican-hat potential coupled to a photon — after the scalar gets a vacuum expectation value `⟨φ⟩`, the photon acquires a mass `m²(V) = e²⟨φ⟩²` by the conventional minimal-coupling formula, the radial scalar fluctuation remains as a massive physical particle, and the angular mode disappears into the now-massive vector. No physical massless scalar survives. The breaking is set by the negative `μ²` in the Lagrangian.

**The effective potential and the generating functional.** The proper relativistic object for asking "which configuration is the true vacuum, once quantum fluctuations are included" is the effective potential. It is built from Schwinger's generating-functional formalism: couple the field linearly to a classical source `J(x)` via `L → L + J(x)φ(x)`, define the connected generating functional `W[J]` by `e^{iW[J]} = ⟨0⁺|0⁻⟩_J`, and the classical field `φc(x) = δW/δJ = ⟨0⁺|φ|0⁻⟩_J/⟨0⁺|0⁻⟩_J`. The Legendre transform `Γ[φc] = W[J] − ∫J φc` (Jona-Lasinio 1964) is the effective action, and it satisfies `δΓ/δφc = −J`. For a translation-invariant configuration `Γ` reduces to `−(volume)·V(φc)`, where `V(φc)` is the **effective potential**: an ordinary function whose stationary points at `J=0` are the candidate vacua, and whose minimum is the true vacuum. Two standard facts:
- Expanded in 1PI Green's functions, the `n`-th derivative of `V` at a point is the sum of all one-particle-irreducible Feynman graphs with `n` external lines carrying zero momentum.
- In the tree approximation (no closed loops), `V` is exactly the classical potential. The whole content of "quantum corrections to the vacuum" lives in the closed-loop graphs.

Jona-Lasinio introduced this effective potential to study spontaneous symmetry breaking, with the breaking produced by a Lagrangian mass term and the effective potential used to organize the corrections to a breaking already present at tree level.

**Renormalization of a massless theory.** A quartic scalar theory is renormalizable; the divergences are absorbed into wave-function, mass, and coupling-constant counterterms. In a *massless* theory the Green's functions have logarithmic (infrared) singularities as the external momenta vanish, so some of the usual on-shell conditions defined at zero momentum cannot be imposed at the origin. The standard momentum-space prescription defines the coupling not at zero momentum but at some off-mass-shell Euclidean point. There is a corresponding freedom in any subtraction scheme — the renormalization point is arbitrary.

**The renormalization group (Gell-Mann & Low 1954).** Because the renormalization point is arbitrary, physics must be independent of it: a change in the reference scale `M` can be compensated by a change in the renormalized coupling and a rescaling of the field. This invariance is the renormalization-group equation; its content is the running of the coupling `λ(M)` governed by a `β`-function and the running field normalization governed by an anomalous dimension `γ`. Gell-Mann and Low used this in QED to control the high-energy behavior; the same machinery applies to any renormalizable theory and, in particular, lets one resum the leading logarithms that appear in perturbation theory.

**The effective Lagrangian for constant background fields, and its imaginary part.** A foundational precedent for "compute the one-loop correction to an effective Lagrangian/potential in a constant background" is the Euler-Heisenberg calculation (Euler & Heisenberg 1936; Schwinger 1951) of the effective Lagrangian of QED in a constant electromagnetic field. For a constant magnetic field the result is real; for a constant electric field it develops an **imaginary part**, whose physical meaning is that the vacuum is unstable and decays into electron-positron pairs. The lesson carried forward: when the background makes the vacuum kinematically unstable, the one-loop effective Lagrangian acquires an imaginary part, and that imaginary part is a genuine physical signal, not a calculational error.

**Renormalizability of gauge theories.** The contemporaneous proofs that spontaneously broken gauge theories are renormalizable ('t Hooft; Lee & Zinn-Justin) make it legitimate to take seriously a gauge theory with a non-invariant vacuum and to compute its higher-order corrections, in a suitable gauge, as a consistent perturbation expansion. The Weinberg-Salam model of the weak and electromagnetic interactions (Weinberg 1967; Salam) is the concrete `SU(2)×U(1)` gauge theory of leptons in which a scalar doublet with a negative mass-squared term breaks the symmetry and gives the `W` and `Z` their masses.

## Baselines

**Tree-level (semiclassical) analysis.** Read off the vacuum from the minimum of the classical potential, and the spectrum from its second derivatives there. For the massless scalar QED above, `V_cl = (λ/4!)φ⁴` has its minimum at the origin, so the verdict is: symmetric vacuum, massless scalar, massless photon, no breaking. This analysis works entirely from the classical potential and discards the closed-loop graphs.

**Goldstone/Higgs symmetry breaking driven by a Lagrangian mass term.** Put `−μ²|φ|²` with `μ²>0` into the Lagrangian by hand; the tree potential is then a Mexican hat, breaking occurs at tree level, `⟨φ⟩² = 6μ²/λ`, the photon eats the Goldstone and gets mass `e²⟨φ⟩²`, and a massive radial scalar remains. This is the workhorse mechanism, with the breaking set by the dimensionful parameter `μ²` in the Lagrangian. The conventional reading in the literature is that scalar electrodynamics with `μ² ≥ 0` does not spontaneously break — the symmetric point is the vacuum.

**Effective potential used to dress an already-broken theory.** Jona-Lasinio's effective potential, and the surrounding work, compute loop corrections to a symmetry breaking already present at tree level — corrections to `⟨φ⟩`, to the masses, to the shape of an already-tipped Mexican hat. The apparatus is applied to theories whose tree potential already chooses an asymmetric vacuum.

**Order-by-order momentum-space perturbation theory.** Expand Green's functions in the coupling, renormalize at an off-shell point to handle the infrared singularities. This computes scattering amplitudes about a chosen expansion point; the expansion is organized around one configuration at a time.

## Evaluation settings

This is a theoretical question; the "yardstick" is internal consistency and the structure of the predictions, not a dataset. The relevant settings:

- **The theories under study.** Massless scalar electrodynamics (charged scalar + photon, couplings `e, λ`); the pure massless self-interacting scalar `(λ/4!)φ⁴` as the simplest template; and, as generalizations, massless non-Abelian gauge theories (e.g. an `SU(2)` triplet of gauge fields coupled to a scalar isovector or isotensor) and the massless version of the `SU(2)×U(1)` electroweak gauge theory of leptons.
- **The object computed.** The effective potential `V(φc)` as a function of the constant background field, to a controlled order; its stationary points and their nature (minimum vs maximum); the value `⟨φ⟩` of the field at the true vacuum; and the masses read from `V''` there together with the gauge-boson masses from minimal coupling.
- **Controls / consistency checks.** Independence of physical results from the arbitrary renormalization scale `M`; the renormalization-group resummation of leading logarithms to test the domain of validity of a one-loop result; the smooth limit as a mass parameter is taken to zero, connecting any massless result to the corresponding massive theory; and the requirement that the perturbative expansion (small couplings) be self-consistent at the configuration where a candidate vacuum sits (the logarithms that appear must stay `O(1)` for the truncation to be trusted there).
- **Spectral predictions used as the test of a candidate vacuum.** The masses of the vector and scalar excitations, and their ratio, expressed in terms of the surviving parameters; whether the spectrum is sensible (no leftover physical massless scalar where one is not wanted; a massive vector if a gauge symmetry is broken).

## Code framework

For this pure-theory derivation the natural "code" is a small symbolic/numerical harness that (a) carries the algebra of the candidate effective potential and (b) plots its shape to confirm where the minimum lies. The pre-existing primitives are a computer-algebra system (symbolic differentiation, series, logs) and a numerical plotting stack — both standard. What does not yet exist is the *form of the loop-corrected potential itself*; that is the slot to be filled.

```python
import sympy as sp
import numpy as np

# --- existing primitives ---
phi, M, lam, e = sp.symbols('phi M lambda e', positive=True)
pi = sp.pi

# The tree-level potential is known: the classical, scale-invariant quartic.
def V_tree(phi, lam):
    return lam/sp.factorial(4) * phi**4

# Standard tools that already exist:
#   sp.diff(V, phi, n)            -> the n-th derivative (curvature = mass^2 at a point)
#   sp.solve(sp.diff(V,phi), phi) -> stationary points of a candidate potential
#   sp.series / sp.logcombine     -> organize logs and small-coupling expansions

def one_loop_correction(phi, couplings, M):
    # TODO: the quantum (closed-loop) contribution to the effective potential
    #       that the derivation will construct, and the renormalization that
    #       removes its cutoff dependence.
    pass

def V_effective(phi, couplings, M):
    # tree piece is known; the quantum piece + counterterms go in the slot above
    return V_tree(phi, couplings['lam']) + one_loop_correction(phi, couplings, M)

def true_vacuum(V, phi):
    # given a candidate effective potential, locate and classify its minima
    stat = sp.solve(sp.diff(V, phi), phi)
    return stat  # then test V'' > 0 and compare V values

def spectrum(V, phi, vev, e):
    # masses from curvature at the vacuum, gauge-boson mass from minimal coupling
    mS2 = sp.diff(V, phi, 2).subs(phi, vev)
    mV2 = e**2 * vev**2
    return mS2, mV2

def plot_potential(Vnum, phi_grid):
    # numerical sanity check of the shape (where is the minimum?)
    import matplotlib.pyplot as plt
    plt.plot(phi_grid, [Vnum(x) for x in phi_grid])
```
