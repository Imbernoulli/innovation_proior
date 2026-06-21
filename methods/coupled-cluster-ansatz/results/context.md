# Context: the electron-correlation problem and the tools on the table

## Research question

Given a molecule, the electronic Schrödinger equation `H Ψ = E Ψ` (in the clamped-nuclei Born–Oppenheimer approximation) is, in principle, solved exactly by the **Hartree–Fock (HF)** mean-field approximation plus a correction. HF replaces the instantaneous electron–electron repulsion by an average field; the single Slater determinant `|Φ0⟩` it produces already captures roughly 99% of the total electronic energy. But the missing ~1% — the **correlation energy** `ΔE = E_exact − E_HF` (Löwdin 1955) — is exactly where chemistry lives: bond dissociation energies, reaction barriers, the energetics that distinguish one molecule from another. The mean field misses the fact that electrons *avoid each other instantaneously*; recovering that motion is the correlation problem.

The precise goal: starting from the HF determinant `|Φ0⟩` built from a finite one-electron (spin-orbital) basis, compute the correlation energy and the correlated wavefunction **accurately and at a cost that is usable for real molecules** — and, crucially, in a way that **scales correctly with the size of the system**. A method that gives a beautiful correlation energy for one water molecule but the wrong energy for two non-interacting water molecules (it should be exactly twice) is useless for the central task of chemistry, which is *comparing* energies: reaction energies, dissociation curves, relative stabilities. So the solution must satisfy:

- **Size-consistency**: for two subsystems A and B pulled infinitely far apart, `E(A···B) = E(A) + E(B)`.
- **Size-extensivity**: for `N` non-interacting copies of a system, the energy scales as `N·E` (equivalently, the correlation energy per particle is independent of system size).

These two properties — trivially satisfied by the *exact* full solution — are the hard constraint that any practical, *truncated* method must also satisfy, and the one the prevailing tools fail.

## Background

**The many-electron Hamiltonian and the independent-particle starting point.** In a basis of `M` orthonormal spin-orbitals `{φp}`, with `n` occupied (holes `i,j,k,…`) and `N = M − n` virtual (particles `a,b,c,…`), the electronic Hamiltonian contains one- and two-body terms only:
```
H = Σ_pq ⟨p|h|q⟩ a†_p a_q + ¼ Σ_pqrs ⟨pq||rs⟩ a†_p a†_q a_s a_r ,
```
written in second quantization with creation/annihilation operators obeying the fermion anticommutators `{a_p, a†_q} = δ_pq`, and `⟨pq||rs⟩ = ⟨pq|rs⟩ − ⟨pq|sr⟩` the antisymmetrized two-electron integral. The reference `|Φ0⟩ = a†_i a†_j … |vac⟩` is the HF determinant, with `H|Φ0⟩` giving back `E_HF = Σ_i ⟨i|h|i⟩ + ½ Σ_ij ⟨ij||ij⟩`. Correlation is described by admixing excited determinants `|Φ_i^a⟩, |Φ_ij^ab⟩, …` (one, two, … occupied orbitals replaced by virtuals).

**Second-quantization machinery (the standard toolkit).** The objects available for manipulating these determinants and operators are: the **occupation-number representation** of determinants; **normal ordering** of operator strings relative to a chosen vacuum; **contractions** between pairs of operators; and **Wick's theorem** (Wick 1950), which writes any product of creation/annihilation operators as a sum of normal-ordered strings with all possible contractions, so that a vacuum expectation value reduces to its *fully contracted* terms. The natural vacuum for an electronic problem with a filled HF sea is the **Fermi vacuum** `|Φ0⟩` itself, with the **particle–hole** reinterpretation: `a†_a, a_i` create excitations (a particle above the sea, a hole in it), `a_a, a†_i` destroy them. Normal ordering relative to the Fermi vacuum makes `⟨Φ0|{…}|Φ0⟩ = 0` automatically, and lets one define the **normal-ordered Hamiltonian** `H_N = H − ⟨Φ0|H|Φ0⟩ = F_N + W_N`, splitting off the (Fock) one-body and (fluctuation) two-body parts. For a canonical HF reference the Fock operator is diagonal, `f_pq = ε_p δ_pq`.

**The diagrammatic language (from many-body physics).** Many-body perturbation theory in physics had, by the late 1950s, developed a fully diagrammatic apparatus: Goldstone (1957) diagrams (time-ordered), the Hugenholtz antisymmetrized-vertex variant, and the associated **linked-diagram theorem** (Brueckner 1955 for the cancellation at low order; Goldstone 1957 to all orders). The theorem's content: the exact correlation energy and wavefunction are sums of **linked** diagrams only — the **unlinked** ones (disconnected pieces multiplying the reference) cancel against renormalization terms. This is precisely the diagrammatic statement of size-extensivity: a sum that contains unlinked diagrams scales wrongly with particle number; a purely linked sum scales correctly. Hubbard (1957) and Hugenholtz expressed the exact wave operator that maps `|Φ0⟩` to the exact state in terms of these linked contributions.

**The motivating empirical facts (knowable before any new method).**
1. *Truncated CI is not size-extensive.* This is a structural, demonstrable fact, not a measurement: for two non-interacting subsystems each treated at the singles-and-doubles level, the combined wavefunction needs *simultaneous* double excitations on both — i.e. a quadruple excitation overall — which CISD cannot represent. The CISD energy of the pair is strictly above twice the energy of one. Numerically the error grows with system size and never goes away.
2. *The dominant "high" excitations are products of low ones.* When two electron pairs sit in well-separated regions of a molecule, the leading four-electron effect is the *simultaneous, independent* correlation of the two pairs. In amplitude terms this is a **disconnected** quadruple — a product of two double-excitation amplitudes (`∼ n²N²` independent numbers) — and it is far larger than the genuine *connected* four-electron cluster (`∼ n⁴N⁴` numbers). The physically important higher excitations are *factorizable*.
3. *Many-body perturbation theory keeps only linked diagrams and is extensive to every order*, but it is a finite-order, order-by-order expansion of fixed accuracy, not a self-consistent resummation.

**The borrowed idea from nuclear physics.** For nuclear matter — many identical fermions interacting through a strong short-range force — Coester (1958) and Coester & Kümmel (1960) had proposed writing the correlated wavefunction as the action of an *exponentiated* correlation operator on a model state, an "Ursell-type" / cluster expansion (the same combinatorial device Ursell and Mayer introduced for the partition function of an interacting gas in statistical mechanics). The idea was understood in the nuclear community by the late 1950s but regarded as intractable to turn into closed, solvable equations for a realistic interacting many-fermion system; no one had reduced it to a finite set of working equations for the electronic structure of atoms and molecules.

## Baselines

**Configuration interaction (CI).** Write the correlated state as a *linear* expansion over the reference and its excitations,
```
|Ψ_CI⟩ = (1 + C) |Φ0⟩,   C = C1 + C2 + C3 + … ,
C1|Φ0⟩ = Σ c_i^a |Φ_i^a⟩,   C2|Φ0⟩ = Σ c_ij^ab |Φ_ij^ab⟩,  …
```
and determine the coefficients `{c}` variationally (diagonalize `H` in the chosen determinant space). **Full CI** (all excitations up to `n`-fold) is the exact answer in the basis, variational and size-extensive — but the number of determinants is `∼ (nN)^n`, so it is impossible beyond tiny systems. The practical method is **truncated CI**, almost always **CISD** (singles + doubles), `|Ψ_CISD⟩ = (1 + C1 + C2)|Φ0⟩`. Its eigenvalue equations are
```
⟨Φ_i^a| (H − E) C1 + H C2 |Φ0⟩ = 0,
⟨Φ_ij^ab| H C1 + (H − E) C2 |Φ0⟩ = 0,   E_CISD = ⟨Φ0|H(C1 + C2)|Φ0⟩ + E_ref .
```
CISD is variational, cheap-ish (`∼ n²N⁴`), and orbital-invariant within the occupied and within the virtual block. **The gap it leaves:** truncated CI is *not size-extensive*. The eigenvalue carries `−E·C_n` renormalization terms whose contribution does not cancel once the configuration space is cut to a subset; these are exactly the *unlinked* contributions that the linked-diagram theorem says must cancel in the exact theory. The consequence is concrete: CISD on `A···B` far apart does not give `E(A) + E(B)`, the error grows with size, and there can be no table of truncated-CI energies one could add up to get reaction energies. Extending the CI space to recover the missing physics means including (at least) quadruple excitations, `∼ n⁴N⁴` coefficients — for `n = 10, N = 100` that is `∼ 10⁶` times the work of CISD — even though, by fact (2) above, the *important* part of those quadruples is just a product of the doubles one already has.

**Rayleigh–Schrödinger / many-body perturbation theory (MBPT).** Partition `H = H0 + V` with `H0` the sum of Fock operators, expand the energy and wavefunction in powers of `V`. Order by order this is just the perturbative way to extract the CI eigenvalue; the second-order energy is the familiar
```
E^(2) = ¼ Σ_ijab |⟨ij||ab⟩|² / (ε_i + ε_j − ε_a − ε_b).
```
Once *all* configurations that can contribute in a given order are kept, RSPT becomes **MBPT**, which is a purely linked-diagram theory and therefore size-extensive at every order (Brueckner 1955; Goldstone 1957; Kelly 1969). **The gap it leaves:** MBPT is a *finite-order* expansion. It gives a fixed-accuracy snapshot (MBPT(2), MBPT(3), MBPT(4)) and does not resum a chosen class of effects to infinite order; for strongly correlated situations the low-order truncation is simply not accurate enough, and going to high order by hand is hopeless.

**The exponential / cluster idea from nuclear physics (Coester 1958; Coester & Kümmel 1960).** The proposal to represent the correlated state by an exponentiated correlation operator acting on a model determinant, by analogy with the Ursell–Mayer cluster expansion. **The gap it leaves:** it had not been reduced to closed, finite working equations for the molecular electronic-structure problem, nor connected operationally to the linked-diagram bookkeeping in a form a quantum chemist could compute. It was a promising representation without a usable algorithm.

## Evaluation settings

The natural yardsticks that existed at the time, against which a correlation method would be judged:

- **Reference exact answer:** full CI in the same finite spin-orbital basis is the unambiguous "exact" target; the figure of merit is the fraction of the full-CI correlation energy recovered (`ΔE_method / ΔE_FCI`).
- **Systems:** small atoms and molecules (e.g. the BH, HF, H₂O, N₂ series) in modest Gaussian or Slater bases; the **uniform electron gas**, where the random-phase approximation provides a comparison; and stretched-bond / dissociation geometries (e.g. R, 1.5R, 2.0R), which stress size-consistency hardest.
- **Metrics:** correlation energy (millihartree) versus full CI; behaviour of the dissociation curve; and — the decisive structural test — whether the method gives `E(A) + E(B)` for separated fragments and the correct `N·E` scaling for `N` replicas.
- **Cost accounting:** the computational scaling in `n` (occupied) and `N` (virtual) — `∼ n²N⁴` for a doubles-level iterative method — and the number of independent amplitudes to store.

## Code framework

The pieces below already exist before any correlation method is filled in: a routine that hands back the HF orbitals, orbital energies, and the two-electron integrals transformed into the molecular spin-orbital basis (antisymmetrized), the Fock matrix, and the orbital-energy denominators. The contribution will occupy the empty correlation block.

```python
import numpy as np

# --- pre-existing: integrals over HF spin-orbitals ---
# f[p, q]          : Fock matrix in the MO spin-orbital basis (diagonal for canonical HF)
# g[p, q, r, s]    : antisymmetrized two-electron integrals <pq||rs>
# o = slice(0, nocc); v = slice(nocc, nso)   : occupied / virtual ranges
# eps = np.diagonal(f)                        : orbital energies

def mo_integrals(...):
    """Return f, g, o, v, eps from a converged HF calculation. (Given.)"""
    ...
    return f, g, o, v, eps

# Orbital-energy denominator for a double substitution i,j -> a,b
# (amplitude arrays are laid out [a, b, i, j]: two virtual, two occupied)
def denominator_doubles(eps, o, v):
    n = np.newaxis
    return (-eps[v, n, n, n] - eps[n, v, n, n]
            + eps[n, n, o, n] + eps[n, n, n, o])

# --- the correlation model: to be designed ---
# Choose how the correlated wavefunction is parameterized in terms of the
# excitation amplitudes living on the reference, and what equations those
# amplitudes must satisfy. This is the open slot.

def amplitude_residual(t, f, g, o, v):
    """Given current amplitudes t, return the quantity that must vanish at
    the correlated solution. TODO: the object we define here."""
    pass  # TODO

def correlation_energy(t, f, g, o, v):
    """Energy correction implied by the converged amplitudes.
    TODO: the energy expression for the chosen parameterization."""
    pass  # TODO

def solve(f, g, o, v, eps, maxit=100, tol=1e-10):
    """Iterate the amplitude equations to self-consistency."""
    D = denominator_doubles(eps, o, v)
    t = np.zeros(...)             # TODO: shape of the amplitude array
    e_old = 0.0
    for it in range(maxit):
        r = amplitude_residual(t, f, g, o, v)   # TODO
        t = t + r / D                            # Jacobi update
        e = correlation_energy(t, f, g, o, v)    # TODO
        if abs(e - e_old) < tol:
            break
        e_old = e
    return t, e
```
