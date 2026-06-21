# Context — correcting the mean-field picture of a many-electron system

## Research question

For an atom or molecule with many electrons, the exact non-relativistic electronic energy is the lowest eigenvalue of

$$\hat H = \sum_{p} \hat h(p) + \sum_{p<q} \frac{1}{r_{pq}},$$

with $\hat h(p) = -\tfrac12\nabla_p^2 - \sum_A Z_A/r_{pA}$ the one-electron (kinetic + nuclear-attraction) operator and $1/r_{pq}$ the instantaneous Coulomb repulsion between electrons $p$ and $q$ (atomic units). The best *single-determinant* description — each electron moving in the averaged field of all the others — is obtainable and chemically useful, but it replaces the instantaneous repulsion by an average. That missing piece is the **correlation energy**: it is small (of order 1% of the total electronic energy) but it is exactly the part that decides bond energies, barrier heights, and reaction thermochemistry.

The question this context sets up: *given* the best mean-field (Hartree–Fock) solution and all of its by-products — orbitals, orbital energies, the determinant — what is a principled route to computing the correlation correction for an arbitrary many-electron system?

## Background

**Hartree–Fock / self-consistent field.** Following Hartree's self-consistent-field idea and Fock's antisymmetric reformulation (Fock 1930), the ground state is approximated by a single Slater determinant $\Phi_0$ of $N$ orthonormal spin-orbitals $\{\psi_i\}$. Minimizing $\langle\Phi_0|\hat H|\Phi_0\rangle$ subject to orthonormality yields the one-electron eigenvalue problem

$$\hat F\,\psi_i = \varepsilon_i\,\psi_i, \qquad \hat F(1) = \hat h(1) + \sum_{j}^{\rm occ}\big[\hat J_j(1) - \hat K_j(1)\big],$$

where the Coulomb and exchange operators are built from the occupied orbitals,
$$\hat J_j(1)\psi(1) = \Big[\int d2\,\psi_j^*(2)\tfrac{1}{r_{12}}\psi_j(2)\Big]\psi(1),\quad \hat K_j(1)\psi(1) = \Big[\int d2\,\psi_j^*(2)\tfrac{1}{r_{12}}\psi(2)\Big]\psi_j(1).$$
The Fock operator $\hat F$ is itself built from its own eigenfunctions, so the equations are solved iteratively to self-consistency. The diagonalizing ("canonical") solution produces $N$ occupied spin-orbitals with energies $\varepsilon_i$ and, for any finite basis of size $M$, a further set of $2M-N$ **virtual** (unoccupied) spin-orbitals $\psi_a$ with energies $\varepsilon_a$. Two facts about this solution are load-bearing here:
- The HF determinant is **stationary**: at the converged solution the energy is unchanged to first order under any orthonormality-preserving mixing of orbitals. Equivalently, the Fock matrix is diagonal in the canonical basis, $\langle\psi_i|\hat F|\psi_a\rangle = 0$ for any occupied $i$, virtual $a$.
- The total HF energy is **not** the sum of orbital energies. Summing $\varepsilon_i$ double-counts each electron–electron interaction (it appears once in each partner's mean field), so $E_{\rm HF} = \sum_i \varepsilon_i - \tfrac12\sum_{ij}\langle ij\|ij\rangle$, where $\langle pq\|rs\rangle \equiv \langle pq|rs\rangle - \langle pq|sr\rangle$ is the antisymmetrized two-electron integral in physicists' notation, $\langle pq|rs\rangle = \int d1\,d2\,\psi_p^*(1)\psi_q^*(2)\tfrac{1}{r_{12}}\psi_r(1)\psi_s(2)$.

**Slater determinants as a basis.** Any determinant built from the HF spin-orbitals — the reference $\Phi_0$, the singly substituted $\Phi_i^a$ (occupied $i\to$ virtual $a$), the doubly substituted $\Phi_{ij}^{ab}$, and so on — is a legitimate $N$-electron antisymmetric function, and the full set of them (for a given orbital basis) spans the $N$-electron space. The **Slater–Condon rules** give matrix elements of the Hamiltonian between such determinants: because $\hat H$ contains only one- and two-body operators, $\langle\Phi_0|\hat H|\Phi\rangle$ vanishes whenever $\Phi$ differs from $\Phi_0$ by **three or more** spin-orbitals.

**Rayleigh–Schrödinger perturbation theory (RSPT).** Split $\hat H = \hat H_0 + \lambda \hat V$ with $\hat H_0$ exactly soluble and $\hat V$ a (hopefully small) perturbation. Expand $\Psi = \sum_n \lambda^n\Psi^{(n)}$ and $E = \sum_n\lambda^n E^{(n)}$, insert into $\hat H\Psi = E\Psi$, and collect powers of $\lambda$. With $\Psi^{(0)}$ a nondegenerate eigenstate of $\hat H_0$ (eigenvalue $E^{(0)}$) and intermediate normalization $\langle\Psi^{(0)}|\Psi^{(n)}\rangle = \delta_{n0}$, the standard results are
$$E^{(1)} = \langle\Psi^{(0)}|\hat V|\Psi^{(0)}\rangle,\qquad E^{(2)} = \sum_{\mu\neq 0}\frac{|\langle\Psi^{(0)}|\hat V|\Psi_\mu^{(0)}\rangle|^2}{E^{(0)} - E_\mu^{(0)}}.$$
The second-order sum runs over the *other* eigenstates of $\hat H_0$ — which requires a soluble $\hat H_0$ whose complete set of eigenstates is easy to enumerate for a many-electron system.

**The state of correlation methods circa the early 1930s.** Perturbation theory had been used for two-electron problems: Hylleraas (1930) treated He and two-electron ions with explicitly $r_{12}$-dependent trial functions and variational second- and third-order energy functionals, accurate but tailored to two electrons. The variational route more broadly (configuration interaction: diagonalize $\hat H$ in a basis of substituted determinants) gives a rigorous upper bound to the ground-state energy for a given orbital basis.

## Baselines

**Hartree–Fock (the reference to be corrected).** Core idea above. It gives a variational upper bound and hands over a complete orthonormal orbital set (occupied + virtual), the orbital energies $\varepsilon$, the determinant, and the stationarity property — reusable structure for any correction method.

**Configuration interaction (variational correlation).** Write $\Psi = c_0\Phi_0 + \sum c_i^a\Phi_i^a + \sum c_{ij}^{ab}\Phi_{ij}^{ab} + \cdots$ and diagonalize $\hat H$ in this determinant basis. Full CI is exact within the orbital basis; the difference $E_{\rm FCI} - E_{\rm HF}$ is the basis-set correlation energy and serves as the gold-standard yardstick. Truncating the expansion (CISD, etc.) makes the calculation tractable for larger systems.

**Hylleraas-type explicitly correlated perturbation theory.** Minimizing the Hylleraas functionals yields second- and third-order energy corrections for two-electron systems with $r_{12}$ in the trial function; very accurate for He.

## Evaluation settings

The natural yardstick is the **basis-set correlation energy**, $E_{\rm corr} = E_{\rm FCI} - E_{\rm HF}$, for small closed-shell systems where full CI is feasible (e.g. He, H$_2$, HeH$^+$, LiH, Ne, HF, H$_2$O) in standard small Gaussian basis sets (minimal STO-$n$G, split-valence 6-31G and the like). Energies are reported in Hartree (or millihartree). One assesses (i) what fraction of $E_{\rm corr}$ a method recovers, (ii) whether it is size-consistent (energy of well-separated non-interacting fragments equals the sum of fragment energies), and (iii) its computational cost as a power of system size $M$ (number of basis functions).

## Code framework

The pre-existing machinery is an SCF/Hartree–Fock program that, given a molecule and a basis, returns the converged orbital coefficients, orbital energies, and the means to obtain two-electron integrals. Built on top of that, the slot below is where the correlation correction goes.

```python
import numpy as np

# --- already available: a converged Hartree-Fock (SCF) solution ---
# C    : (nbf, nmo) MO coefficient matrix from the SCF
# eps  : (nmo,)     canonical orbital energies, ascending
# nocc : number of occupied spin-orbitals (= number of electrons)
# I_ao : (nbf,)*4   two-electron repulsion integrals (mu nu | la si) in the AO basis
# E_hf : converged Hartree-Fock total energy

def ao_to_mo(I_ao, C):
    """Transform the two-electron integrals from the AO basis into the MO basis.
    A primitive; the standard four-index transformation already exists."""
    return np.einsum('pi,qj,pqrs,rk,sl->ijkl', C, C, I_ao, C, C, optimize=True)

def correlation_correction(eps, mo_integrals, nocc):
    """Given the HF orbital energies and the MO two-electron integrals,
    return the correlation energy correction to E_hf.

    The mean field is the best single determinant; what it omits is recovered here.
    """
    # TODO: the correlation-energy expression we will derive
    pass

# E_total = E_hf + correlation_correction(eps, ao_to_mo(I_ao, C), nocc)
```
