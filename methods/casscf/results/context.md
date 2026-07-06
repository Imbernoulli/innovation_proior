# Context

## Research question

Given a molecule, we want a variational wavefunction that stays qualitatively correct
everywhere on the potential energy surface — at the equilibrium geometry, along a bond that is
being stretched to dissociation, in a diradical, in an electronically excited state, in a
first-row transition-metal complex. The single-determinant Hartree–Fock picture fails precisely
where chemistry is most interesting: when two or more electron configurations become
energetically close (near-degeneracy), one determinant can no longer carry the physics. The
goal is a method that (i) describes these multi-configuration ("static" / non-dynamical
correlation) situations, (ii) keeps the wavefunction a proper spin eigenfunction, (iii) optimizes
both the configuration mixing and the orbitals self-consistently, and (iv) does so without the
user having to hand-pick which excited configurations to include — and at a cost that does not
explode with the size of the configuration list. Such a method would give smooth, spin-pure
potential surfaces for bond making and breaking and a balanced zeroth-order description of
excited and open-shell states.

## Background

**Why one determinant breaks.** A closed-shell restricted Hartree–Fock (RHF) determinant places
both electrons of a bond in the bonding orbital σ. Stretch the bond toward dissociation and the
correct wavefunction must become an equal, spin-pure mixture of σ² and σ*² (the two
configurations become degenerate, corresponding to two neutral fragments); RHF cannot do this and
develops a spurious ionic tail, giving a dissociation energy that is far too high. Unrestricted
Hartree–Fock (UHF) lets the α and β orbitals differ and recovers a qualitatively correct curve
shape, but the determinant is no longer an eigenfunction of Ŝ² — it is spin-contaminated, and
energies are quantitatively poor. To break a single covalent bond of a closed-shell molecule and
keep a spin eigenfunction one needs at least two configurations, ···(σ)² and ···(σ*)², in the
zeroth-order wavefunction. The same near-degeneracy logic applies to diradicals, low-spin
open-shell states, and excited states with strong configurational mixing.

**Multiconfiguration SCF.** The response is to write the wavefunction as a short CI expansion
Ψ = Σ_I c_I Φ_I over selected determinants or configuration state functions (CSFs), determine the
CI coefficients c_I variationally, and — crucially — determine the orbitals not as those that
minimize a single determinant (Hartree–Fock) but as those that minimize the energy of the whole
multiconfiguration expansion. Both sets of parameters, c_I and the molecular-orbital
coefficients, are optimized to make the energy stationary. This is the multiconfiguration
self-consistent field (MCSCF) idea.

**Second-quantized machinery available at the time.** The electronic Hamiltonian is written with
the spin-summed unitary-group generators Ê_pq = Σ_σ a†_{pσ} a_{qσ},
Ĥ = Σ_pq h_pq Ê_pq + ½ Σ_pqrs (pq|rs) ê_pqrs, with ê_pqrs = Ê_pq Ê_rs − δ_qr Ê_ps. The energy
of any CI state is a contraction of the one- and two-electron integrals with the one- and
two-particle reduced density matrices γ_pq = ⟨Ψ|Ê_pq|Ψ⟩ and Γ_pqrs = ⟨Ψ|ê_pqrs|Ψ⟩:
E = Σ_pq h_pq γ_pq + ½ Σ_pqrs (pq|rs) Γ_pqrs. Orbital changes are conveniently written as a
unitary transformation of the molecular orbitals, C → C·U with U unitary so orthonormality is
preserved.

**Solving large CI problems without storing the Hamiltonian.** The direct CI idea (Roos 1972)
obtains the CI eigenvector iteratively straight from the list of two-electron integrals, never
forming or storing the Hamiltonian matrix, which is what makes long CI expansions tractable. The
graphical unitary group approach (Paldus; Shavitt 1977–78) gives a systematic, spin-adapted (CSF)
bookkeeping in which the coupling coefficients depend only on the relative occupations of the
orbitals, so they can be tabulated and reused, and the reduced density matrices fall out of the
same symbolic machinery.

## Baselines

**Hartree–Fock (Roothaan).** Single determinant; orbitals optimized so the Fock matrix is
diagonal (Brillouin: no first-order coupling to single excitations). It is the reference everyone
starts from, but it carries exactly one configuration; at bond-breaking and in near-degenerate
situations it is qualitatively wrong (spurious ionic dissociation for RHF; spin contamination for
UHF). It leaves open the entire problem of static correlation.

**General (hand-selected) MCSCF.** The wavefunction is Σ_I c_I Φ_I over a list of configurations
chosen by the user, with c_I and orbitals optimized together. This can describe near-degeneracy
and stays spin-pure if CSFs are used. Its gaps: the user must decide which configurations to
include, so the result depends on that choice and the procedure is not black-box; convergence of
the coupled orbital/CI optimization is delicate; and the energy's invariance properties under
orbital rotation interact with the chosen configuration list in ways that complicate the
optimization.

**Super-CI orbital optimization via the generalized Brillouin theorem (Grein & Chang 1971;
Levy & Berthier 1968).** For a multiconfiguration reference, the generalized Brillouin theorem
says that at the optimal orbitals the reference does not couple, through Ĥ, to any state obtained
by a single orbital replacement on it. This is turned into an algorithm: build the singly-excited
states on the reference, solve the small "super-CI" secular problem of Ĥ in the space spanned by
the reference plus these singles, and read the orbital corrections off the CI coefficients of the
singles; rotate the orbitals accordingly and iterate to self-consistency. It is an elegant
first-order orbital optimizer. Its limitation as originally formulated: the singly-excited space
is built on the whole reference expansion, so the size of the super-CI secular problem and the
work per orbital step grow with the length of the configuration list, which becomes the
bottleneck once the reference is large.

**Full-valence MCSCF / "full optimized reaction space" (Ruedenberg and co-workers, 1978–1982).**
Rather than selecting individual configurations, take *all* configurations that can be formed
within a chosen set of valence orbitals and optimize the orbitals for that complete expansion.
This is conceptually clean and removes per-configuration selection. Its limitation is algorithmic:
the complete expansion is large, and optimizing the orbitals for it with the available
orbital-optimization machinery is expensive, since that machinery's cost grows with the length of
the configuration list.

## Evaluation settings

The natural yardsticks are potential energy curves and surfaces for bond dissociation (e.g. a
diatomic single bond, where RHF is known to fail), spin-state and excitation energies for
open-shell and excited states, and small first-row molecules where near-degeneracy is present
(the kind of system for which a single determinant is inadequate). The relevant accuracy targets
are qualitative correctness of the dissociation curve and a spin-pure description across the
surface, computed in a standard Gaussian basis with the orbitals and configuration mixing
optimized self-consistently. The comparison point is what a single-determinant SCF, or a
hand-selected MCSCF, gives on the same system and basis.

## Code framework

Pre-existing primitives: an integral/SCF package supplies the one-electron integrals h_pq, the
two-electron integrals (pq|rs), and a converged set of orthonormal molecular orbitals; a direct
full-CI / unitary-group solver can return the lowest CI vector in a chosen orbital space together
with its reduced density matrices. The orbital classes and the slot where the new optimization
goes are left empty.

```python
import numpy as np
from scipy.linalg import expm

# --- existing infrastructure (SCF + integrals + a direct CI / UGA solver) ---
hcore_ao, eri_ao = get_integrals(mol)        # h_{μν}, (μν|λσ) in AOs
mo = run_rhf(mol)                             # converged orthonormal MOs (start guess)

def ao2mo(mo):
    """Transform AO integrals into the current MO basis -> h_pq, (pq|rs)."""
    ...
    return h_mo, eri_mo

def fci_solve(h_eff, eri_act, ncas, nelec_act):
    """Direct full CI in the active space; return CI vector and active-space RDMs."""
    ...
    return ci, cas_dm1, cas_dm2     # gamma_tu, Gamma_tuvw  (active indices only)

# --- orbital partition (the chemistry input): inert vs. correlating orbitals ---
ncore = ...     # doubly occupied in every configuration
ncas  = ...     # the chosen window of correlating orbitals
# remaining orbitals are empty in every configuration

def energy(mo, ci_dm1, ci_dm2):
    """Total energy from integrals contracted with the density matrices."""
    # TODO
    pass

def optimize(mo):
    """The object we will design: make the energy stationary with respect to BOTH
       the configuration mixing and the orbitals, given the partition above."""
    # TODO
    pass

mo_opt = optimize(mo)
```
