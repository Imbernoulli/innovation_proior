# Context

## Research question

Given a complete active space self-consistent field (CASSCF) wave function — a small full
configuration interaction inside a chosen set of active orbitals, with the orbitals optimized — how
do we recover the rest of the correlation energy that the CASSCF model is missing, cheaply and in a
way that stays valid where the simpler single-determinant theory breaks down?

A CASSCF calculation is built to capture *static* (near-degeneracy) correlation: bond breaking,
diradicals, partially filled d/f shells, low-lying excited states. With a tractable active space
(a dozen orbitals or so) it gives a balanced, spin-pure, qualitatively correct reference, but it
recovers only a small fraction of the total correlation energy. The bulk of the correlation — the
short-range *dynamic* correlation arising from the electron–electron cusp, which lives in the huge
space of excitations into the inactive and virtual orbitals — is left out. CASSCF energies are
therefore not quantitatively useful: bond energies and excitation energies can be off by an
electron-volt or more.

We already know how to add dynamic correlation to a *single-determinant* reference cheaply: apply
second-order perturbation theory to a Hartree–Fock determinant (the MP2 recipe), which is
non-iterative and size-extensive. The open problem is to obtain the same cheap, non-iterative
dynamic-correlation correction *on top of a multiconfigurational CASSCF reference* — so that the
zeroth-order description is correct at bond breaking and for open shells (where the single-reference
recipe fails outright), and the perturbation only supplies the missing short-range correlation. The
central obstruction is that perturbation theory needs a soluble zeroth-order Hamiltonian whose
zeroth-order state is an eigenfunction, and a variationally optimized CASSCF CI vector is not an
eigenfunction of any obvious one-electron operator.

## Background

**Hartree–Fock (Fock 1930; Roothaan 1951).** The wave function is a single Slater determinant; the
molecular orbitals minimize the energy in the mean field of all the electrons. The one-electron Fock
operator F̂ = Σ_p ĥ(p) + Σ_{p,i}[Ĵ_i(p) − K̂_i(p)] has the occupied and virtual canonical orbitals as
eigenfunctions with orbital energies ε_p; by Koopmans' theorem ε_p ≈ −(IP) for an occupied orbital
and ≈ −(EA) for a virtual one. HF is the best single determinant but, being one configuration, is
qualitatively wrong wherever two or more configurations are near-degenerate — most sharply at
homolytic bond breaking, where the restricted solution develops an unphysical ionic tail, and for
open-shell and excited states.

**Møller–Plesset second-order perturbation theory (MP2; Møller & Plesset 1934).** Take the HF
determinant Φ₀ as the zeroth-order state and the sum of one-electron Fock operators H₀ = Σ_p F̂(p) as
the zeroth-order Hamiltonian. Then every Slater determinant built from the HF spin-orbitals is an
eigenfunction of H₀ with eigenvalue equal to the sum of its occupied orbital energies — a ready-made
complete spectrum. The perturbation V = Ĥ − H₀ is the *fluctuation potential*: the instantaneous
repulsion Σ_{i<j} 1/r_ij minus the averaged HF mean field. Through first order one only recovers HF
itself (E⁽⁰⁾+E⁽¹⁾ = E_HF), so the first correlation correction appears at second order. Singles
vanish by the stationarity of HF (generalized Brillouin) and triples-and-higher vanish because Ĥ is
a two-body operator (Slater–Condon), so only double excitations contribute:

  E⁽²⁾ = Σ_{i<j,a<b} |⟨ij‖ab⟩|² / (ε_i + ε_j − ε_a − ε_b),

with occupied indices i,j, virtual a,b, and the denominator always negative so E⁽²⁾ ≤ 0. MP2 is
cheap (one closed-form sum after an O(N⁵) integral transform), non-iterative, and size-extensive.
But it is single-reference: where no single determinant dominates the wave function — bond breaking,
near-degeneracies, many open-shell and excited-state situations — the HF zeroth order is poor, the
small denominators ε_a+ε_b−ε_i−ε_j collapse, and the second-order energy blows up or gives
nonsense. It also fails to be a valid zeroth order for spin-pure open shells.

**CASSCF (Roos, Taylor & Siegbahn 1980; FORS, Ruedenberg).** Choose a subset of orbitals as
*active*, a number of active electrons, and form the full CI of all ways to distribute those
electrons among the active orbitals, keeping the other orbitals doubly occupied (*inactive*) or
empty (*virtual*); then optimize both the CI coefficients and the molecular orbitals. The result is
black-box (no hand-picking of configurations), invariant to rotations within each orbital class, and
spin-adapted. In second quantization with the unitary-group generators Ê_pq = Σ_σ â†_{pσ}â_{qσ} and
Ĥ = Σ_pq h_pq Ê_pq + ½ Σ_pqrs (pq|rs) ê_pqrs, the energy is E = Σ_pq h_pq D_pq + ½ Σ (pq|rs) P_pqrs,
depending on the CI vector only through the active-space one- and two-particle reduced density
matrices D and P. CASSCF captures static correlation, but a tractable active space is small, so it
leaves out the dynamic correlation that lives in the external space. (The cost scales
combinatorially with the active size, so the active space cannot simply be enlarged to absorb
dynamic correlation.)

**Multireference CI (Siegbahn).** Add single and double excitations on top of a multiconfigurational
reference and diagonalize. This recovers dynamic correlation on a correct reference and is
variational, but truncated CI is *not* size-extensive (its error grows with system size) and it is
expensive.

**Rayleigh–Schrödinger perturbation theory.** Partition Ĥ = Ĥ₀ + V and expand E and |Ψ⟩ in powers of
the perturbation. With intermediate normalization ⟨Ψ⁽⁰⁾|Ψ⟩ = 1: E⁽⁰⁾ = ⟨Ψ⁽⁰⁾|Ĥ₀|Ψ⁽⁰⁾⟩,
E⁽¹⁾ = ⟨Ψ⁽⁰⁾|V|Ψ⁽⁰⁾⟩, the first-order wave-function correction solves the projected equation
(Ĥ₀ − E⁽⁰⁾)|Ψ⁽¹⁾⟩ = −(V − E⁽¹⁾)|Ψ⁽⁰⁾⟩ on the orthogonal complement of |Ψ⁽⁰⁾⟩, and
E⁽²⁾ = ⟨Ψ⁽⁰⁾|V|Ψ⁽¹⁾⟩. The whole construction *requires* a zeroth-order Hamiltonian of which the
zeroth-order state is an eigenfunction; for a single determinant the Fock operator supplies this, but
no such natural one-electron operator is known for a multiconfigurational CASSCF state.

**The generalized Brillouin / variational principle.** Because the CASSCF state is variationally
optimal both within the active-space CI and with respect to orbital rotations, its coupling through
the full Ĥ to any other configuration that stays inside the active space vanishes, and (for a
canonical reference) the inactive–virtual block of the orbital Fock matrix is zero. These
stationarity facts constrain which excitations can interact with the reference at first order.

## Baselines

**MP2 on an HF reference.** Core idea and equations above. *Where it breaks:* its zeroth-order state
is a single determinant, so wherever the true wave function is multiconfigurational the reference is
qualitatively wrong, the perturbation is no longer small, near-zero excitation-energy denominators
appear, and the series diverges or yields unphysical energies. It cannot be used at bond breaking,
for many open-shell systems, or for excited states.

**CASSCF.** Core idea and equations above. *Where it breaks:* with a computationally tractable
active space it captures static but not dynamic correlation, so it leaves most of the correlation
energy unrecovered and is quantitatively inaccurate; enlarging the active space to capture dynamic
correlation is barred by the combinatorial scaling.

**Multireference CI.** Core idea above. *Where it breaks:* truncated MRCI is not size-extensive — its
correlation error grows with the number of electrons, so relative energies between systems of
different size are unbalanced — and it is computationally heavy.

## Evaluation settings

A method of this kind would be assessed, at the time, on small first- and second-row molecules and
transition-metal systems where multireference effects are decisive: dissociation curves of diatomics
(e.g. N₂, F₂) compared against full CI in small bases; spectroscopic constants (equilibrium bond
length, harmonic frequency, dissociation energy); and vertical excitation energies of organic
chromophores against experiment. Reference data are full-CI or large-MRCI energies in modest
Gaussian basis sets (e.g. ANO / correlation-consistent sets), with a frozen-core option for the deep
inactive orbitals. The natural diagnostics are size-extensivity (additivity of the correlation
energy for non-interacting fragments) and the behaviour of the correction across a bond-breaking
coordinate.

## Code framework

Existing primitives. An integral/SCF/CASSCF backend already provides, for a chosen orbital partition
into inactive (i,j), active (t,u,v), and virtual (a,b) orbitals: the one-electron MO integrals
`h[p,q]`, the two-electron integrals `eri[p,q,r,s]` in chemists' notation `(pq|rs)`, the CASSCF
reference energy `E0`, and the active-space one- and two-particle reduced density matrices
`dm1[t,u]`, `dm2[t,u,v,w]` of the reference. Dense linear algebra (`numpy.linalg.eigh`, an iterative
linear solver such as conjugate gradient) is available.

```python
import numpy as np

class CASReference:
    """Outputs of a converged CASSCF: orbital partition, integrals, RDMs, E0."""
    def __init__(self, ncore, nact, nvirt, h, eri, dm1, dm2, E0):
        self.ncore, self.nact, self.nvirt = ncore, nact, nvirt
        self.h, self.eri = h, eri          # MO integrals: h[p,q], eri[p,q,r,s] = (pq|rs)
        self.dm1, self.dm2 = dm1, dm2       # active-space 1- and 2-RDMs of the reference
        self.E0 = E0                        # CASSCF reference energy

def build_zeroth_order_operator(ref):
    """One-electron operator built from the reference, used to define H0."""
    # TODO: the effective one-electron operator we will design from the reference
    pass

def first_order_space(ref):
    """Enumerate the configurations that can interact with the reference at first order."""
    # TODO: the set of excitations on the reference that we will determine here
    pass

def build_linear_system(ref):
    """Assemble the matrices/vectors of the first-order equations."""
    # TODO: the equations whose solution gives the first-order correction
    pass

def correlation_energy(ref):
    """Second-order correction to the CASSCF energy."""
    H0op   = build_zeroth_order_operator(ref)
    space  = first_order_space(ref)
    system = build_linear_system(ref)
    # TODO: solve the first-order equations and contract to the energy correction
    pass
```

The contribution fills these stubs: the one-electron operator, the first-order space, the linear
system, and the contraction that produces the energy.
