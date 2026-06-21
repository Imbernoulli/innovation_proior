# Context: fast electrostatics for large periodic molecular systems

## Research question

In a molecular-dynamics or Monte-Carlo simulation of a large biomolecule in
explicit solvent — tens of thousands of atoms in a periodically repeated unit
cell — the electrostatic energy and the force on every atom must be recomputed
at *every* time step, millions of times over a run. Coulomb's law is
long-ranged: the potential of a point charge falls off only as 1/r, so a naive
truncation at a finite cutoff radius introduces large, qualitatively wrong
artifacts (spurious ordering of water, drift in protein structure, distorted
ion solvation). The physically correct treatment of a periodic charged system
is the Ewald lattice sum, which converges absolutely and respects the periodic
boundary conditions exactly. But the textbook Ewald sum, applied naively, costs
O(N²) operations per step (a sum over all pairs of charges); even with the
splitting parameter tuned optimally it is only O(N^{3/2}).

The problem to solve: how to evaluate the Ewald electrostatic energy and forces
on all N atoms of a large periodic unit cell efficiently, for general
non-orthogonal unit cells, in a way that slots into existing MD codes that
already maintain a short-range neighbor (Verlet) list.

## Background

**Periodic Coulomb sums are conditionally convergent.** For N point charges
q₁,…,q_N at positions r₁,…,r_N in a neutral unit cell U with lattice vectors
a₁,a₂,a₃, the electrostatic energy summed over all periodic images is

  E = ½ Σ′_n Σ_{i,j} q_i q_j / |r_i − r_j + n|,   n = n₁a₁+n₂a₂+n₃a₃,

the prime omitting i=j when n=0. This outer series is only *conditionally*
convergent: its value depends on the order of summation and on the macroscopic
shape and surroundings of the crystal.

**Ewald's 1921 splitting.** Paul Ewald removed the conditional convergence with
a theta-transform identity. Around each point charge place a neutralizing
Gaussian charge cloud of width set by a parameter β; the original sum becomes
two absolutely convergent pieces plus bookkeeping terms:
- a **direct (real-space) sum** of *screened* charges,
  Σ_n erfc(β|r+n|)/|r+n|, which decays like a Gaussian and so is negligible
  beyond a few Å — with β large enough only the minimum image survives and the
  direct sum is O(N) using a cutoff;
- a **reciprocal (Fourier-space) sum** of the smooth compensating Gaussians,
  (1/πV) Σ_{m≠0} [exp(−π²m²/β²)/m²] exp(2πi m·r), summed over reciprocal
  lattice vectors m = m₁a₁*+m₂a₂*+m₃a₃*, V the cell volume;
- a **self-energy** correction −(β/√π) Σ_i q_i² (each charge interacts with its
  own screening Gaussian and that must be removed);
- a **surface/dipole term** J(D), derived by de Leeuw, Perram and Smith (1980),
  depending on the unit-cell dipole moment D and the boundary conditions /
  external dielectric (the "tinfoil" vs. vacuum choice).

The total energy is invariant to β (changing β only adds a constant to the pair
potential ψ = Φ_dir + Φ_rec for a neutral system); β merely shifts work between
the two sums. The **reciprocal sum**, written as a sum over charge pairs,
½ Σ_{i,j} q_i q_j Φ_rec(r_j − r_i;β), is the part that scales as O(N²):
naively one evaluates Φ_rec for every pair, every step. With β tuned to balance
direct and reciprocal cost the optimum is O(N^{3/2}) (Karasawa and Goddard,
1989).

**The structure factor.** The reciprocal sum can be written using the
electrostatic *structure factor* S(m) = Σ_{j=1}^N q_j exp(2πi m·r_j), as
E_rec = (1/2πV) Σ_{m≠0} [exp(−π²m²/β²)/m²] S(m)S(−m). The cost of S(m) is the
issue: for each of O(N) reciprocal vectors m one sums over N charges.

**Particle–mesh ideas from plasma/cosmology.** Hockney and Eastwood
(*Computer Simulation Using Particles*, 1981) introduced the
particle–particle/particle–mesh (P³M) family: *assign* the particle charges to
a regular grid using a smooth assignment function (nearest-grid-point,
cloud-in-cell, triangular-shaped-cloud, …), solve the discrete Poisson equation
for the long-range part *on the grid* with the fast Fourier transform — an
FFT on a grid of M points costs O(M log M) — then interpolate the resulting
field back to the particles. This achieves O(N log N) for the long-range part
in plasma and gravitational simulations.

**Diagnostic facts about the regime (knowable before any new method):**
- The Gaussian screening makes erfc(βr)/r drop below ~10⁻⁸ within ~9 Å for
  typical β, so the real-space sum is genuinely short-ranged and O(N) with a
  neighbor list.
- The reciprocal kernel exp(−π²m²/β²)/m² is itself a smooth, rapidly decaying
  function of m — the reciprocal sum has only a modest number of significant
  terms — but evaluating S(m) charge-by-charge for each m is the bottleneck.
- Lagrange interpolation of a smooth function on a grid of spacing h has error
  that shrinks geometrically with the interpolation order; high-order *global*
  polynomial interpolation, however, is ill-conditioned (Runge's phenomenon —
  the interpolant oscillates wildly near the ends of the interval).
- Cell-multipole / fast-multipole expansions (Greengard–Rokhlin, 1987, and
  several MD adaptations) also reach near-linear scaling.

## Baselines

**Conventional Ewald summation.** Compute the direct sum with a real-space
cutoff (O(N)), the self and surface terms (O(N)), and the reciprocal sum
directly as ½ Σ_{i,j} q_i q_j Φ_rec(r_j−r_i) or equivalently via the structure
factor Σ_{m≠0} (kernel) |S(m)|². Exact (to the chosen number of m-vectors and
real-space cutoff) and the gold standard for correctness. The reciprocal part is
O(N²) at fixed β and O(N^{3/2}) with β optimized.

**Plain cutoff truncation (Verlet-list Coulomb).** Sum 1/r interactions only
within a cutoff using a neighbor list; O(N). Cheap and trivial to implement.
It is the *speed* target every accurate method is compared against.

**Hockney–Eastwood particle–mesh (P³M).** Assign charges to a grid with a
low-order smooth window, solve Poisson on the grid by FFT, interpolate back;
O(N log N). The structural template for grid-based long-range solvers.

**Cell-multipole / fast-multipole methods.** Hierarchically group distant
charges and approximate their field by truncated multipole expansions;
near-linear scaling (Greengard–Rokhlin and MD adaptations by Brooks et al.,
Ding–Karasawa–Goddard, Board et al., Schmidt–Lee, Lee–Warshel).

**Other long-range accelerations.** Cubic-polynomial expansion of the Ewald pair
potential (Adams–Dubey), table lookup of the pair potential (Smith–Pettitt),
Wigner potentials (Cichocki et al.), and twin-range / multiple-time-step
splitting (Berendsen).

## Evaluation settings

The natural yardstick is a set of large macromolecular crystal unit cells built
from crystallographic heavy-atom coordinates via the space-group symmetry, with
point charges assigned from a standard molecular-mechanics force field
(AMBER). Representative systems: an ionic B-DNA crystal (~8000 atoms,
orthorhombic P2₁2₁2₁), an HIV-1 protease crystal (~30000 atoms, tetragonal
P4₁2₁2), and a p21/ras crystal (~26000 atoms, trigonal P3₂21) — including
non-orthogonal cells. Accuracy is measured against the *exact* Ewald pair
potential evaluated for all pairs (to relative accuracy ~10⁻⁷): the relative
rms energy error, the relative rms force error over all atoms, and the maximum
relative single-atom force error. Cost is the percent overhead of the
fast electrostatics evaluation relative to a conventional truncated-Coulomb
nonbond evaluation using a 9 Å atom-based Verlet list, on a single processor.
Tunable knobs are the reciprocal-grid spacing (set by the grid dimensions
K₁,K₂,K₃ relative to the cell edges) and the interpolation order; the screening
parameter β is fixed so the real-space sum is converged at the chosen cutoff.

## Code framework

The pieces that already exist: a periodic simulation cell with lattice vectors
and the conjugate reciprocal vectors; a force field giving each atom a charge
and a position; standard numerical libraries (linear algebra, complex
transforms); and an MD nonbond loop that already computes short-range
interactions from a Verlet list. The short-range (direct, screened) Coulomb sum,
the self-energy, and the surface/dipole term are standard Ewald bookkeeping and
are computed as usual. The open slot is the fast evaluation of the
reciprocal-space contribution.

```python
import numpy as np

def reciprocal_kernel(m2, beta, V):
    # Ewald reciprocal weight exp(-pi^2 |m|^2 / beta^2) / (pi V |m|^2); 0 at m=0
    out = np.zeros_like(m2)
    nz = m2 > 0
    out[nz] = np.exp(-(np.pi**2) * m2[nz] / beta**2) / (np.pi * V * m2[nz])
    return out

def direct_space_energy_forces(positions, charges, beta, cutoff, neighbor_list):
    # screened real-space Coulomb: sum_i<j q_i q_j erfc(beta r)/r within cutoff
    # standard short-range term, O(N) with the Verlet list
    ...
    return E_dir, F_dir

def self_energy(charges, beta):
    return -(beta / np.sqrt(np.pi)) * np.sum(charges**2)

def surface_term(positions, charges, cell):
    # de Leeuw-Perram-Smith dipole/boundary term J(D); O(N)
    ...
    return E_surf, F_surf

class ReciprocalSpaceEvaluator:
    """The expensive O(N^2) reciprocal Ewald sum lives here.
    Goal: produce E_rec and the force on every atom much faster than O(N^2)."""

    def __init__(self, cell, beta, **params):
        self.cell = cell
        self.beta = beta
        # TODO: store whatever the scheme we design needs

    def setup(self):
        # TODO: any one-time precomputation
        pass

    def energy_and_forces(self, positions, charges):
        # TODO: the fast reciprocal-space scheme we will design
        pass

def electrostatics(positions, charges, cell, beta, cutoff, neighbor_list, recip):
    E_dir, F_dir = direct_space_energy_forces(positions, charges, beta, cutoff, neighbor_list)
    E_rec, F_rec = recip.energy_and_forces(positions, charges)
    E_self = self_energy(charges, beta)
    E_surf, F_surf = surface_term(positions, charges, cell)
    E = E_dir + E_rec + E_self + E_surf
    F = F_dir + F_rec + F_surf
    return E, F
```
