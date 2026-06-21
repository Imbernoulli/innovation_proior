## Research question

A classical molecular-mechanics (MM) force field assigns a molecule a total potential energy as a
sum of cheap, physically interpretable terms — bond stretches, angle bends, torsions about
rotatable bonds, and nonbonded van der Waals and electrostatic interactions. The bond, angle, and
nonbonded terms are reasonably transferable: a C–C bond's stiffness or a partial charge can be
carried from molecule to molecule with little error. The energy of *internal rotation* about a
single bond — the torsion profile — is the part that resists this treatment. Conjugation,
hyperconjugation, the imperfect cancellation of 1–4 nonbonded interactions across a bond, and
subtle electronic effects all shape the barrier to rotation, and none of these is captured cleanly
by the harmonic bond/angle springs or the pairwise nonbonded sum. A high-level quantum-mechanical
(QM) calculation can produce the true rotational energy profile of a molecule as a function of a
chosen dihedral angle. Given that QM profile, how should the parameters of an explicit torsion term
be determined so that, when that term is *added* to everything else the force field already
computes, the model reproduces the QM barrier?

## Background

**The MM energy decomposition.** The standard additive force field writes the total energy as

    E = sum_bonds k_b (l - l0)^2
      + sum_angles k_a (theta - theta0)^2
      + sum_torsions E_tors(phi)
      + sum_{i<j nonbonded} [ eps_ij ((r0_ij/r_ij)^12 - 2 (r0_ij/r_ij)^6) + q_i q_j / (4 pi eps0 r_ij) ].

Bonds and angles are harmonic about reference values; the nonbonded sum is a Lennard-Jones 12-6
term plus a Coulomb term over atom-centered point charges, with interactions between atoms
separated by exactly three bonds (the "1–4" pairs, which straddle the rotatable bond) scaled down
by a fixed factor (0.5 in OPLS) because those atoms also feel the explicit torsion term.

**The torsion term as a Fourier series.** Rotation about a bond is periodic: the energy is a
function of the dihedral angle phi with period 2pi. The universal choice is a truncated Fourier
series. In the AMBER form (Cornell et al. 1995) each dihedral contributes

    E_tors(phi) = sum_n (V_n / 2) [1 + cos(n phi - gamma_n)],

with amplitudes V_n, integer periodicities n, and phase offsets gamma_n. In the OPLS all-atom form
(Jorgensen, Maxwell, Tirado-Rives 1996) the phases are pinned to 0 or 180 degrees and absorbed into
the sign of the cosine, giving

    E_tors(phi) = (1/2) [ V1 (1 + cos phi) + V2 (1 - cos 2phi) + V3 (1 + cos 3phi) + V4 (1 - cos 4phi) ].

The periodicities carry chemical meaning known before any fit: an sp3–sp3 bond has three staggered
minima, so n = 3 dominates; conjugation or a planarity preference (cis/trans) shows up as n = 2;
an asymmetry between gauche and anti appears as n = 1. The "1 +" offset is a baseline convention:
with conventional non-negative amplitudes each term stays >= 0, so the series has a well-defined
zero, but the offset only shifts the energy baseline and does not change the shape. An equivalent re-expression as a
polynomial in cos(phi), the Ryckaert–Bellemans form E = sum_{m=0}^{5} C_m cos(psi)^m with
psi = phi - 180, is used by some engines; the C_m are exact linear combinations of the coefficients
used in the chosen Fourier convention.

**QM rotational profiles as the fit target.** The reference data is a *relaxed* torsion scan: the
target dihedral is constrained to a grid of values from -180 to 180 degrees (steps of ~10–15
degrees), and at each grid point a constrained QM geometry optimization relaxes every other degree
of freedom. This yields E_QM(phi), the energy of the lowest-strain conformer at each angle.
Historically the level of theory was modest (RHF/6-31G* in the original OPLS-AA work); other
established protocols use dispersion-corrected DFT for the geometry and double-hybrid single points
(e.g. wB97X-D/6-311++G(d,p) geometries with B2PLYP-D3BJ/aug-cc-pVTZ energies).

**The non-torsion MM energy must be subtracted first.** The torsion term is not meant to
reproduce the whole barrier — only the part the rest of the force field misses. So the fit target
is not E_QM(phi) itself but the *residual* after the force field's bond, angle, and nonbonded
contributions (with the torsion term switched off) are removed. Setting all torsion amplitudes to
zero and re-evaluating the MM energy on the same scanned conformations gives
E_MM_without_torsion(phi); the quantity the torsion term must reproduce is
E_QM(phi) - E_MM_without_torsion(phi). In practice the no-torsion MM energy is obtained directly
from a simulation engine — for example, by setting every torsional parameter to zero in a GROMACS
topology and recomputing the energies — so that the 1–4 scaling and every other term are treated
exactly as they will be in production.

**Regularization and scan-point weighting.** Two established countermeasures address overfitting:
restrict which periodicities are allowed (use only the chemically expected n), or admit all
periodicities up to some cutoff and regularize — shrink redundant amplitudes toward zero (an L1 /
Lasso penalty drives them exactly to zero), or bound the amplitudes to a physical range. A second,
orthogonal choice is how to weight scan points: a uniform least-squares weight treats every angle
equally, whereas a Boltzmann weight exp(-E_rel / kT) in the conformer's relative energy E_rel above
the minimum, or a flat-then-attenuating weight, emphasizes
the low-energy, thermally populated regions of the profile that actually matter for simulation and
de-emphasizes high, sparsely visited barriers.

## Baselines

**Hand/iterative fitting against QM (OPLS-AA, Jorgensen, Maxwell, Tirado-Rives 1996).** The
original all-atom parameterization fit V1, V2, V3 (and sometimes V4) for each torsion type to RHF
rotational profiles of >50 small molecules, adjusting amplitudes so the summed MM curve matched the
QM curve, with the nonbonded/bond/angle terms held fixed and the 1–4 interactions scaled by 0.5.

**AMBER torsion parameterization (Cornell et al. 1995).** Uses the same residual idea with explicit
phases gamma_n, fitting V_n by matching QM conformational energies. Keeping gamma as a free
parameter makes the model *nonlinear* in its parameters (cos(n phi - gamma) is not linear in
gamma), which requires a nonlinear optimizer.

**General nonlinear optimizers (ForceBalance-style).** Modern pipelines pose torsion fitting as
minimizing a weighted sum of squared QM-vs-MM energy deviations over the scan and descend it with a
general optimizer (e.g. steepest descent / Levenberg–Marquardt), often holding periodicities,
phases, and 1–4 scaling fixed and optimizing only the amplitudes. Automated tooling (TorsionDrive
for the scan; bespoke per-molecule fits) wraps this.

## Evaluation settings

The natural target molecules are small organic fragments containing a single rotatable bond of
interest (alkanes for n = 3 sanity checks, conjugated and substituted systems for n = 1, 2 effects;
biomolecular backbone and side-chain dihedrals; nucleic-acid glycosidic torsions). The reference
data is a relaxed QM torsion scan on a -180..180 degree grid at a stated level of theory; the MM
side evaluates the same scanned conformations with the same nonbonded/1–4 treatment and the
torsion term zeroed. The yardstick for a fit is how closely the reconstructed MM profile
(everything plus the fitted torsion
term) tracks the QM profile across the scan — typically the residual energy error per scan point,
optionally Boltzmann-weighted toward populated conformers. Transferability is assessed by sharing
one parameter set across all dihedrals of the same atom-type quadruplet within and across molecules.

## Code framework

The available machinery is an MM engine that can evaluate the energy decomposition with the
rotational term switched on or off, a routine to measure dihedral angles along a scan, and numerical
linear algebra for solving coefficient systems.

```python
import numpy as np

# existing: MM energy decomposition with a switchable rotational term
def mm_energy(coords, params, include_rotational_term=True):
    e = bonded_bonds(coords, params) + bonded_angles(coords, params) \
        + nonbonded(coords, params)            # LJ + Coulomb, with 1-4 scaling
    if include_rotational_term:
        e += rotational_energy(coords, params)
    return e

def rotational_energy(coords, params):
    # TODO: fill in the periodic rotational term once its coefficients are known
    pass

# existing: geometry / scan utilities
def measure_dihedral(coords, quad):
    """Dihedral angle (degrees) for an atom quadruplet."""
    ...  # standard geometry

def load_matched_scan(qm_file, mm_file, dih_dir):
    # TODO: collect QM energies, no-rotational-term MM energies, and angle columns
    pass

# existing: generic coefficient solver
def solve_coefficients(system_matrix, rhs, bounds=None):
    # TODO: obtain the unknown coefficients from the assembled linear system
    pass

def estimate_rotational_coefficients(qm_energies, mm_without_rotational_energies,
                                     dihedral_angle_files, coefficient_count):
    # TODO: reduce matched scan data to coefficients for the rotational energy slot
    pass
```
