# Context: closed-shell electronic structure of molecules, circa 1950

## Research question

Given a molecule with fixed nuclei (the Born–Oppenheimer clamped-nuclei picture), we want the
electronic ground-state energy and the one-electron wavefunctions (the orbitals each electron occupies)
to useful accuracy, **for an arbitrary molecule** — not just an atom. We want a procedure that a person,
and ideally a computing machine, can actually carry out: it must turn into a finite, well-posed numerical
task with a definite recipe, rather than an open-ended search over arbitrary functions of three dimensions.

The best available *ab initio* theory of many-electron systems — the self-consistent field with exchange —
is, for a molecule, a set of coupled three-dimensional nonlinear integro-differential equations. For a
single atom these collapse (by spherical symmetry) to one-dimensional radial equations that can be
integrated on a grid by hand or by a desk calculator; this is how atomic structure was being computed. A
molecule has no such symmetry: each orbital is a genuinely three-dimensional function with no privileged
coordinate system, and the equations contain a *nonlocal* operator (an integral kernel), and they must be
re-solved every cycle because the effective potential depends on the very orbitals being sought.

## Background

**The variational principle.** For any normalized trial wavefunction Ψ, ⟨Ψ|H|Ψ⟩ ≥ E₀, with equality only
for the true ground state. Minimizing the energy over a family of trial functions therefore gives the best
member of that family and an energy that is a rigorous upper bound. Restricting Ψ to a parameterized
family turns "solve the Schrödinger equation" into "minimize a function of the parameters" (the Ritz idea).

**Antisymmetry and the Slater determinant.** Electrons are identical fermions, so the total wavefunction
must change sign under exchange of any two. A determinant of one-electron spin-orbitals,
Ψ = (N!)^{-1/2} det[φ₁(1)φ̄₁(2)…], is antisymmetric by construction and so is the simplest admissible
many-electron ansatz built from orbitals.

**Hartree's self-consistent field (Hartree 1928).** Hartree replaced the mutual electron repulsion by an
*average* field: each electron moves in the electrostatic potential of the nuclei plus the smeared-out
charge of all the others. Because that average field is built from the orbitals, the equations are nonlinear
and are solved by iteration — guess the orbitals, build the field, re-solve for the orbitals, repeat until the
input and output orbitals agree ("self-consistent"). Hartree used a product wavefunction (no
antisymmetry) and solved the resulting one-electron radial equations numerically for atoms. The averaged
Coulomb interaction of one electron with the charge cloud of orbital j is the *Coulomb* term J.

**Fock's correction (Fock 1930; Slater).** Starting instead from an antisymmetric determinant adds a term
with no classical analogue — the *exchange* term K — coupling each electron to the others through the
antisymmetry of the wavefunction. The resulting one-electron operator (call it the Fock operator F)
contains the core one-electron part H (kinetic energy plus attraction to the nuclei), the averaged Coulomb
field built from the occupied orbitals (J), and the exchange piece (K). The exchange operator is *nonlocal*:
acting on an orbital it produces an integral over all space weighted by the other orbitals, i.e. it is an
integral kernel, not a multiplication by a function. The stationarity condition of the determinant's energy
is a one-electron eigenvalue problem **F φ_i = ε_i φ_i** — but with F depending on the {φ_i}, so it too is
solved self-consistently.

**Closed-shell structure.** For the ground state of most stable molecules the electrons pair up: each
spatial orbital φ_i holds one spin-up and one spin-down electron, and there are n doubly-occupied orbitals
for N = 2n electrons (a closed-shell singlet). Carrying out the spin sums in the energy of such a
determinant leaves a purely spatial expression in which each orbital appears with occupancy two and the
two-electron part is a combination of Coulomb and exchange contributions over the occupied orbitals.

**The atoms-only situation.** It is an established, practical observation of the time that F φ_i = ε_i φ_i is
tractable for atoms: the central field lets the angular variables separate analytically and leaves a
one-dimensional radial equation for grid integration. A molecule offers no coordinate system in which the
three-dimensional nonlinear integro-differential eigenproblem separates, so it cannot be put on a grid and
integrated in the same way.

**The chemist's orbital picture (Hund–Mulliken MO; LCAO heuristic).** In the molecular-orbital view each
electron occupies a one-electron function spread over the whole molecule. Empirically, near any nucleus a
molecular orbital looks much like an atomic orbital of that atom (the local potential there is nearly the
isolated-atom potential), so it was already common in *semi-empirical* work to write a molecular orbital as
a sum of atomic orbitals — a "linear combination of atomic orbitals." In that semi-empirical usage the
coefficients were fixed by fitting to data and the working "secular equations" were carried over by analogy
from simpler problems; their matrix entries were empirical parameters rather than quantities tied to the
molecular Hamiltonian itself.

**Atomic orbitals as building blocks.** Slater-type orbitals (exponentials e^{-ζr} times powers and
spherical harmonics) with screening-constant exponents fit to isolated atoms were the standard concrete
atomic functions. Such functions are normalized but, when centered on *different* atoms, are **not mutually
orthogonal** — two orbitals on neighboring atoms have a nonzero overlap integral ∫χ_p* χ_q dv ≠ 0.

## Baselines

**Numerical (grid) Hartree–Fock for atoms.** Solve F φ = ε φ by finite-difference / numerical integration
of the radial equation, iterating to self-consistency. Core idea: discretize the orbital on a radial grid
and integrate the differential equation directly. This approach relies on spherical symmetry to reduce a
three-dimensional problem to one dimension.

**Hartree's product SCF.** Average-field iteration with a product (non-antisymmetric) wavefunction. Each
electron sees the mean Coulomb field of the others; iterate to self-consistency. The wavefunction used is
a product rather than a determinant, so the energy minimized is not that of an antisymmetric fermionic
state and there is no exchange term.

**Semi-empirical LCAO / valence-bond secular methods.** Write the molecular state in terms of
atom-centered functions and diagonalize a small "secular" matrix whose entries are parameters fit to
experiment (ionization potentials, spectra). These schemes capture chemistry cheaply with a few
atom-based functions and empirical integrals. The matrix entries are empirical parameters rather than
quantities derived from the molecular Hamiltonian, and the atom-centered functions are treated as if they
formed a clean orthonormal set.

## Evaluation settings

The natural testbed is a small closed-shell molecule whose orbitals can be built from a handful of
atom-centered functions — the canonical case being a light first-row hydride such as water, taken at a
fixed geometry (a definite bond length and bond angle), with a minimal set of Slater-type (or, later,
Gaussian-expanded) atomic orbitals: 1s, 2s, 2p on the heavy atom and 1s on each hydrogen. The required
inputs are the integrals over those atom-centered functions — overlaps, one-electron core integrals
(kinetic + nuclear attraction), and two-electron repulsion integrals — together with the classical
nuclear–nuclear repulsion. The yardstick is the total electronic energy (an upper bound to the true value,
by the variational principle), its convergence with successive iterations, and the resulting orbital energies
and orbital shapes; metrics are energy in atomic units (Hartree) and the agreement of successive iterates.

## Code framework

What already exists, in pre-method terms: a way to specify a molecule and basis, routines that return the
fixed integral matrices over the chosen atom-centered functions, dense linear algebra (matrix
multiply, symmetric eigensolver, matrix functions), and the variational/self-consistency idea (iterate an
average field to a fixed point). The unknown to be filled in is *how to turn the one-electron eigenproblem
into something finite over these integrals, and how to organize the iteration*.

```python
import numpy as np
from numpy import einsum
from scipy.linalg import eigh, fractional_matrix_power

class Molecule:
    """Holds a molecule + a fixed set of atom-centered basis functions and the
    integrals over them. The integral routines are standard, pre-existing machinery."""
    def __init__(self, atom_charges, atom_coords, n_electrons, nbf):
        self.atom_charges = atom_charges      # nuclear charges Z_A
        self.atom_coords  = atom_coords       # nuclear positions
        self.n_electrons  = n_electrons       # total electron count (closed shell: even)
        self.nbf          = nbf               # number of basis functions m

    # --- pre-existing integral machinery over the atom-centered basis {chi_p} ---
    def overlap(self) -> np.ndarray:          # S_pq = <chi_p | chi_q>
        ...
    def hcore(self) -> np.ndarray:            # H_pq = kinetic + nuclear-attraction one-electron integrals
        ...
    def eri(self) -> np.ndarray:              # (pq|rs) two-electron repulsion integrals, shape (m,m,m,m)
        ...
    def e_nuclear(self) -> float:             # classical nucleus-nucleus repulsion
        ...

# We have a normalized, atom-centered basis {chi_p} and want one-electron functions
# phi_i for the electrons, plus the ground-state energy. The variational principle and
# the self-consistent-field (average-field-iterated-to-a-fixed-point) idea are available.
# What is NOT yet known is how to make the one-electron problem finite and how to set up
# and iterate it over the integral matrices above.

def build_one_electron_operator(mol: Molecule, occupied_state) -> np.ndarray:
    # TODO: form the matrix of the effective one-electron operator from the integrals
    #       and the current occupied state.
    pass

def solve_one_electron_problem(mol: Molecule, op_matrix) -> np.ndarray:
    # TODO: extract the one-electron functions (as coefficients over {chi_p}) from op_matrix.
    pass

def summarize_occupied_state(mol: Molecule, coeffs) -> np.ndarray:
    # TODO: form whatever summary of the occupied functions the operator needs.
    pass

def energy(mol: Molecule, occupied_state) -> float:
    # TODO: total ground-state energy for the current occupied state (+ nuclear repulsion).
    pass

def scf(mol: Molecule, guess):
    # iterate the average field to self-consistency; structure is known, the contents are TODO
    state = guess
    for _ in range(64):
        op = build_one_electron_operator(mol, state)
        coeffs = solve_one_electron_problem(mol, op)
        state = summarize_occupied_state(mol, coeffs)
        # TODO: check convergence of energy and state
    return energy(mol, state), state
```
