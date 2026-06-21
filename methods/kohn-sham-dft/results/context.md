# Context: Ground-State Electronic Structure of Many-Electron Systems

## Research question

Given a collection of atomic nuclei at fixed positions (the Born–Oppenheimer
picture), the electrons settle into a ground state whose density distribution and
total energy determine essentially all of chemistry and condensed-matter physics:
binding energies, bond lengths, lattice constants, reaction barriers, magnetic
moments. In Hartree atomic units (ℏ = m = e = 1) the electrons obey the
non-relativistic many-body Schrödinger equation

  H Ψ = E Ψ,  H = −½ Σ_i ∇_i² + Σ_i v(r_i) + ½ Σ_{i≠j} 1/|r_i − r_j|,

where v(r) is the external (nuclear + applied) potential and the last term is the
electron–electron Coulomb repulsion. The ground-state wavefunction
Ψ(r₁, …, r_N) and energy E are obtained, in principle, from the Rayleigh–Ritz
variational principle E = min_Ψ ⟨Ψ|H|Ψ⟩.

The problem is that Ψ lives in 3N-dimensional configuration space. To reach
"chemical accuracy" a general trial Ψ must be parameterized with on the order of
M ≈ p^{3N} numbers, where p (the number of parameters needed per continuous
variable) is empirically in the range 3–10. The number of electrons N for which
this minimization is feasible therefore grows only logarithmically with the
available computing power: with the most optimistic counts one reaches N ≈ 10,
and "being clever" pushes it to perhaps N ≈ 20. For N ≈ 100 one would need to
minimize in a space of dimension ~10^{150}; even *recording* such a Ψ to the
required accuracy would take ~q^{3N} bits, a number that dwarfs the count of
particles in the observable universe. This is an exponential wall: wavefunction
methods give superb results for small molecules (H₂, light atoms) and become
impossible for large molecules, solids, biomolecules.

The precise goal: find a *complete and in-principle-exact* reformulation of
ground-state electronic structure whose central variable has a number of
components that does **not** grow with N, together with a computational scheme
based on it that is accurate enough for real chemistry and materials — recovering
the binding and shell structure that the cheap density-only theories of the day
could not.

## Background

**The density as a candidate variable.** Almost every physically interesting
quantity — total energy E, the one-electron density n(r), the pair correlation
function — depends on only a few variables and can formally be obtained by
integrating |Ψ|² over all but a few coordinates,

  n(r) = N ∫ |Ψ(r, r₂, …, r_N)|² dr₂ … dr_N,  ∫ n(r) dr = N.

The density n(r) is a function of just three variables, independent of how many
electrons there are. If the ground-state energy could be expressed as a functional
of n(r) alone, the exponential wall would disappear. Whether this is even possible
in principle was an open question: the density obviously fixes N (by integration),
but does it fix the *whole* Hamiltonian?

**Why one suspects it might.** In the crudest density theory (below), the density
substituted into the model relations yields back the external potential up to a
constant, so there n(r) does specify the system. The same can be checked for any
one-particle system, and for a weakly perturbed uniform electron gas, where the
linear-response (susceptibility) of the gas lets one recover the perturbing
potential from the induced density. These special cases suggest a general
hypothesis: the ground-state density of *any* electronic system, interacting or
not, uniquely determines the system.

**The Coulomb pieces.** Two of the energy contributions are easy as density
functionals. The external energy is exactly ∫ v(r) n(r) dr. The classical
electrostatic (Hartree) self-energy of the charge cloud is
  U_H[n] = ½ ∫∫ n(r) n(r′)/|r − r′| dr dr′.
The hard parts are (i) the kinetic energy and (ii) everything in the
electron–electron interaction beyond the classical mean field — exchange (from
antisymmetry / the Pauli principle) and correlation.

**Motivating empirical facts about the cheap theories (knowable before any new
method).** Two diagnostic observations frame the problem:
- A theory that models the *kinetic* energy by a purely local function of the
  density gives qualitatively useful total energies for atoms but **predicts no
  chemical binding at all** — dissociated atoms always come out lower in energy
  than the molecule. The defect is traced specifically to the kinetic-energy term.
- The self-consistent single-particle theory that instead computes the kinetic
  energy from orbitals (Laplacians of one-electron wavefunctions) binds atoms far
  better and reproduces atomic shell structure. The only structural difference
  between the two is *how the kinetic energy is treated*.

These two facts together point at the kinetic energy as the term whose crude local
modeling is fatal — but they do not, by themselves, say how to repair it within a
density-only framework.

**The uniform electron gas as a reference.** The homogeneous interacting electron
gas of density n is the one many-body system whose energetics are essentially
solved. Its kinetic energy per volume is (3/10)(3π²)^{2/3} n^{5/3}; its exchange
energy per volume is known analytically (below); its correlation energy per
electron, e_c(n), is a hard but tractable many-body number, parameterized at the
time by interpolation (e.g. Wigner's e_c ≈ −0.44/(r_s + 7.8) in atomic units) and
characterized by the Wigner–Seitz radius r_s = (3/4πn)^{1/3}.

## Baselines

**Thomas–Fermi (Thomas 1927, Fermi 1928) and Thomas–Fermi–Dirac.** Write the
total energy entirely as a functional of n(r):
  E_TF[n] = C_F ∫ n^{5/3} dr + ∫ v n dr + ½ ∫∫ n n′/|r−r′| dr dr′,
with C_F = (3/10)(3π²)^{2/3} ≈ 2.871 (atomic units), the kinetic-energy density of
a *uniform* gas applied *locally*; Dirac adds a local exchange term −C_x ∫ n^{4/3}.
Minimizing over n at fixed N gives a single closed equation for n(r). Core idea:
the gas-kinetic-energy density evaluated at the local density. **Gap:** the local
n^{5/3} kinetic functional is badly wrong wherever the density varies on the scale
of the Fermi wavelength — i.e. everywhere chemistry happens. Concretely, within
Thomas–Fermi no molecule is stable: the energy of the assembled molecule never
falls below that of the separated atoms, so the theory describes no bonds. It is a
density-only theory that is exact in form but useless for valence physics, and the
breakdown sits in the kinetic term.

**Hartree (Hartree 1928).** Treat each electron as moving in the average potential
of the nucleus and of the mean density of the others:
  v_H(r) = −Z/r + ∫ n(r′)/|r − r′| dr′,  (−½∇² + v_H) φ_j = ε_j φ_j,
  n(r) = Σ_{j=1}^{N} |φ_j(r)|²  (sum over the N lowest states),
solved self-consistently: guess n, build v_H, solve for the φ_j, recompute n,
iterate. Core idea: a *self-consistent single-particle* picture in which the
kinetic energy is computed honestly from the orbitals. It binds atoms much better
than Thomas–Fermi precisely because of this honest kinetic energy. **Gap:** it
omits antisymmetry (no exchange) and correlation entirely, and it is introduced as
a physically reasonable *ansatz* with no derivation tying it to the exact
ground-state energy — there is no statement of what quantity it is the
approximation *to*.

**Hartree–Fock (Fock 1930, Slater 1930).** Restrict the trial Ψ to a single Slater
determinant of orbitals and minimize ⟨Φ|H|Φ⟩. This yields self-consistent
single-particle equations like Hartree's but with an extra **non-local** exchange
term — an integral operator with kernel built from Σ_k φ_k*(r′) φ_k(r)/|r − r′|.
Core idea: exact treatment of exchange via the determinant. **Gap:** (i) by
construction it contains *no correlation* — it is the best *single-determinant*
energy and nothing beyond; (ii) the exchange operator is non-local, making the
equations substantially more expensive to solve than Hartree's; (iii) it still
carries unfavorable scaling for large systems.

**Slater Xα (Slater 1951).** Replace Hartree–Fock's costly non-local exchange by a
*local* exchange potential obtained by averaging the exchange interaction over the
Fermi sphere of the local-density gas:
  v_x^{Xα}(r) = −3 α (3/8π · n(r))^{1/3}  (atomic units),
with α an adjustable parameter; Slater took α = 1, the value from averaging over
the *entire* Fermi sphere. Core idea: a cheap local stand-in for exchange. **Gap:**
the coefficient α is fixed by an averaging *choice* rather than by any variational
or energy-minimizing principle, so it is uncontrolled; and correlation is still
absent. Different reasonable averages give different α, with no internal criterion
to prefer one.

The common situation: there is an exact-in-form density-only theory whose kinetic
term is fatally crude (Thomas–Fermi), and there are orbital-based single-particle
schemes whose kinetic energy is good but which are either incomplete (Hartree),
non-local and correlation-free (Hartree–Fock), or rest on an unjustified constant
(Slater Xα). None of them is simultaneously exact in principle, cheap, and bound
to a variational statement of the true energy.

## Evaluation settings

The natural yardsticks for a new electronic-structure method, all available at the
time and all defined independently of the method:
- **Systems.** Light atoms and small molecules where accurate
  wavefunction/variational results exist (H₂: binding energy and bond length;
  closed-shell atoms such as He, Ne, Ar; alkali metals; simple crystals).
- **Reference data.** Experimental dissociation/atomization energies and bond
  lengths; high-accuracy variational (configuration-interaction-type) total
  energies for small systems; measured cohesive energies, lattice constants, and
  spin susceptibilities for solids.
- **Metrics.** Total ground-state energy; electron density and its shell structure;
  ionization energies; for molecules, binding energy and equilibrium geometry; for
  solids, lattice constant and cohesive energy. "Chemical accuracy" is roughly
  |ΔR| ≲ 0.01 Å in bond length and |ΔD| ≲ 0.1 eV in binding energy.
- **Protocol.** Fix the nuclei (Born–Oppenheimer), specify N and v(r), solve for
  the ground-state density and energy self-consistently, and compare the resulting
  energies/geometries against the reference data.

## Code framework

Numerical primitives for a self-consistent electronic-structure calculation on a
real-space grid. A real function is sampled on a uniform grid; a second-derivative
finite-difference operator gives the kinetic energy; a real symmetric
single-particle Hamiltonian is diagonalized for its lowest eigenpairs; an outer
loop updates the density toward self-consistency. The classical electrostatic
(Hartree) and external pieces are explicit; the treatment of the rest of the energy
is left as an open slot.

```python
import numpy as np

def build_kinetic(x):
    """-1/2 d^2/dx^2 as a finite-difference matrix on a uniform grid."""
    n = len(x); h = x[1] - x[0]
    lap = (np.diag(np.full(n, 2.0))
           + np.diag(np.full(n-1, -1.0), 1)
           + np.diag(np.full(n-1, -1.0), -1)) / h**2
    return 0.5 * lap

def density(psi_gn, occ, x):
    """n(x) = sum_n f_n |psi_n(x)|^2 from L2-normalized orbitals."""
    h = x[1] - x[0]; n = np.zeros_like(x)
    for i, f in enumerate(occ):
        if f:
            psi = psi_gn[:, i]
            psi = psi / np.sqrt(np.sum(psi**2) * h)
            n += f * psi**2
    return n

def hartree(n, x):
    """Classical electrostatic energy and potential of the charge cloud."""
    h = x[1] - x[0]
    K = 1.0 / np.sqrt((x[:, None] - x[None, :])**2 + 1.0)   # softened in 1D
    v_H = K @ n * h
    return 0.5 * np.sum(n * v_H) * h, v_H

def occupations(num_electrons, num_states):
    """Fill the lowest states, 2 electrons each."""
    occ = np.zeros(num_states); e = num_electrons; i = 0
    while e > 0 and i < num_states:
        occ[i] = min(2, e); e -= occ[i]; i += 1
    return occ

def solve(x, v_ext, num_electrons, iters=200, mix=0.3):
    T = build_kinetic(x); h = x[1] - x[0]
    n = np.zeros_like(x)
    occ = occupations(num_electrons, len(x))
    for _ in range(iters):
        E_H, v_H = hartree(n, x)
        # TODO: account for the remaining energy (everything past the classical
        # mean field, and the kinetic energy) and turn the result into whatever
        # enters the single-particle problem -- the form is what the derivation
        # must settle. Then build the Hamiltonian, diagonalize, rebuild n.
        n = (1 - mix) * n + mix * density(psi_gn, occ, x)   # psi_gn from the step above
    # TODO: assemble the total ground-state energy consistently with that choice
    return n
```
