# Context: a strongly interacting degenerate Fermi system

## Research question

A degenerate gas of non-interacting fermions has a clean, complete low-temperature theory. But
the systems we actually care about are not gases: the conduction electrons in a metal repel each
other with a Coulomb energy comparable to their kinetic energy, and liquid helium-3 interacts
strongly enough to condense into a liquid that stays liquid down to the lowest temperatures. The
puzzle is sharp and quantitative. Despite interactions of order the kinetic energy, liquid He-3
measured at millikelvin temperatures has a heat capacity that is *linear* in temperature, exactly
the qualitative signature of a degenerate ideal Fermi gas — only the *slope* is wrong, larger than
the gas prediction by a factor of several. The same linear-in-T electronic heat capacity, far
smaller than the lattice (phonon) contribution, is seen in every normal metal.

So the question is: when the interaction is *not* weak, what survives of the Fermi-gas picture,
and what is renormalized? Can one build a low-temperature theory of a strongly interacting
degenerate Fermi system that works with the *measured* parameters rather than bare-particle
quantities — all without solving the intractable many-body Schrödinger equation?

## Background

**The ideal degenerate Fermi gas (Sommerfeld).** At T = 0 the lowest states up to the Fermi
momentum p₀ are filled; the density fixes p₀ through n = p₀³/3π²ℏ³ (per spin, times 2). The
abruptness of the Fermi step is what makes the gas special: only fermions within ~k_BT of the
surface can be thermally excited, so the heat capacity is linear, C ≈ (π²/3) k_B² T g(ε_F), with
g(ε_F) the density of states at the Fermi level — proportional to the particle mass m. The Pauli
spin susceptibility χ₀ and the n = p₀³/3π²ℏ³ relation are likewise controlled by the Fermi
surface.

**Dressing single particles: Hartree–Fock.** Dresses each fermion in the averaged field of the
others, giving renormalized single-particle energies. The treatment addresses the mean field
through a self-consistent procedure; for the electron gas it produces a single-particle group
velocity that diverges logarithmically at the Fermi surface.

**The "Fermi-type spectrum" idea.** It is known that the qualitative low-temperature behaviour of
a system is fixed by the *character of its energy spectrum* near the ground state, not by the
microscopic Hamiltonian. A Bose liquid such as He II has a phonon/roton spectrum and is
superfluid; a system of fermions should have a *different* type of spectrum. The relevant prior
results in this direction treat *weakly* non-ideal Fermi systems by perturbation theory: the
weakly-non-ideal Fermi gas (Klimontovich and Silin, J. Exptl. Theoret. Phys. 23, 151 (1952);
Silin, ibid. 23, 641 (1952); 27, 269 (1954)), and the charged electron gas with its plasma
oscillations (Gol'dman, ibid. 17, 681 (1947)). These give, in the weak-coupling limit, both the
renormalized single-particle energies and the collective modes — as an expansion in the small
interaction.

**Adiabatic switching as an available tool.** A standard device in quantum mechanics is to turn a
perturbation on slowly ("adiabatically"): a slow, smooth change cannot make the conserved
quantities — total momentum, total spin, particle number — jump, so the labels carried by a state
are protected as long as nothing singular happens to the spectrum. This device is well established
in the perturbative regime; its scope when the interaction is large remains to be characterized.

**Phase space near a Fermi surface.** The Pauli principle sharply restricts the final states
available to a fermion scattering near the Fermi surface: energy and momentum conservation together
with the exclusion of already-filled states confine the partners of a collision to a thin shell
whose thickness is set by how far the participating states sit from the surface. This is a generic
kinematic fact about degenerate Fermi systems, independent of the interaction's strength, and it is
the same counting that makes the low-T collision rate of a degenerate gas fall off as a power of the
temperature.

**The empirical anchors.** Liquid He-3 is the cleanest realization: charge-neutral (short-range
interaction, no Coulomb screening complications), translationally and rotationally invariant, spin
½, and it stays a normal liquid down to the millikelvin range. Its measured low-T heat capacity is
linear in T but with a slope several times larger than the ideal-gas value for atoms of the bare
He-3 mass; its compressibility and magnetic susceptibility are likewise measured. These are pre-existing facts
about the world (Fairbank, Ard and Walters, Phys. Rev. 95, 566 (1954); Walters and Fairbank,
Phys. Rev. 103, 263 (1956)) that fix the parameters a theory must reproduce. The
quadratic-in-T relaxation time underlies the observed sound-absorption ∝ 1/T² (Pomeranchuk, J.
Exptl. Theoret. Phys. 20, 919 (1950)).

## Baselines

**Sommerfeld ideal Fermi gas.** Energy E = Σ ε_p n_p with ε_p = p²/2m fixed; equilibrium is the
Fermi function n_p = [e^{(ε_p−μ)/θ}+1]^{−1}. Yields C ∝ T with slope ∝ m, Pauli χ₀ ∝ g(ε_F),
n = p₀³/3π²ℏ³, and a single sound speed from the gas pressure.

**Hartree–Fock / self-consistent field.** Dresses each fermion in the averaged field of the
others, giving renormalized single-particle energies through the mean-field self-consistency.

**Weakly non-ideal Fermi gas (Silin; Klimontovich–Silin; Gol'dman).** Perturbation theory in a
small interaction for a Fermi gas: gives the leading correction to single-particle energies and,
for the charged case, plasma oscillations.

**Bose-liquid theory of He II.** Shows that the low-temperature properties of a strongly
interacting *quantum liquid* follow from the *type* of its excitation spectrum (phonon/roton →
superfluidity), independent of microscopic detail.

## Evaluation settings

The natural testbed is liquid helium-3 at temperatures well below its Fermi temperature (a few
kelvin), in the millikelvin range where it is a normal degenerate liquid. The observables that
serve as yardsticks are the ones already measured for the ideal-gas comparison: the low-temperature
specific heat (linear coefficient → density of states at the Fermi level), the isothermal
compressibility (from the equation of state / sound velocity, e.g. ordinary sound at ~195 m/s from
Walters–Fairbank), the static magnetic (spin) susceptibility, and the temperature dependence of
sound absorption (∝ 1/T² for ordinary sound). The density fixes the Fermi momentum (p₀/ℏ of order
10⁸ cm⁻¹ for liquid He-3). Conduction electrons in normal metals are the second arena, with the
caveat that there the carrier's momentum is the crystal quasi-momentum rather than the true
momentum, and the lattice carries momentum too.

## Code framework

This is a theory problem; the "code" is the analytic machinery. The scaffold below holds only the
pre-method primitives — the ideal-gas relations and the equilibrium occupation — and one empty slot
for the low-temperature theory to be constructed.

```python
import numpy as np

# --- Known pre-method primitives ---------------------------------------------
HBAR = 1.0545718e-34          # J s
KB   = 1.380649e-23           # J / K

def fermi_momentum(n):
    """Ideal-gas relation between number density and Fermi momentum (per spin x2)."""
    return HBAR * (3.0 * np.pi**2 * n) ** (1.0 / 3.0)

def fermi_distribution(eps, mu, theta):
    """Equilibrium occupation of a single mode of energy eps."""
    return 1.0 / (np.exp((eps - mu) / theta) + 1.0)

def ideal_gas_dos(m, p0, V):
    """Density of states at the Fermi surface for a free gas of mass m."""
    return V * m * p0 / (np.pi**2 * HBAR**3)

# --- The slot the theory will fill -------------------------------------------
def low_temperature_theory(distribution):
    """Given a degenerate, strongly interacting Fermi system specified through its
    occupation, return the low-temperature observables (heat capacity, compressibility,
    magnetic susceptibility, any collective response) in terms of measurable parameters.
    For a non-interacting gas these reduce to the Sommerfeld results above."""
    # TODO: the theory we will construct here
    pass
```
