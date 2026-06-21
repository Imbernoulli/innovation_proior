# Context

## Research question

The matter wave has been quantized once already. Schrödinger's wave function lives in an abstract
3N-dimensional configuration space — one set of coordinates per particle — and that is exactly the
feature that makes a many-body gas awkward to think about: you cannot picture it, the indistinguishability
of identical particles has to be enforced by hand through symmetrization or antisymmetrization of the
wave function, and the description does not parallel the way the radiation field is now handled. The
program of *second quantization* fixes this for some cases: promote the field amplitude on ordinary
three-dimensional space to a non-commuting quantity (a *q*-number), and the existence of discrete
particles together with their statistics emerges from the operator algebra of that single field. Two
cases are settled this way — the radiation field, and the Bose-Einstein matter gas.

The open problem is the third case: **construct a quantized matter-wave field whose quanta obey the
Pauli exclusion principle** — a field theory in which the occupation number of every single-particle
mode is restricted to the two values 0 and 1, and in which the many-particle states are correctly
*antisymmetric*. Such a construction would put Fermi gases (electrons in metals, the degenerate
electron gas) on the same field-theoretic footing as photons and Bose gases, remove the abstract
configuration space entirely, and make the Pauli principle a *consequence* of the field's
multiplication rules rather than an extra postulate.

## Background

**Matrix mechanics and the first quantized field.** Since Born, Heisenberg and Jordan (Z. Physik 35,
557, 1926) the dynamical variables of a quantum system are non-commuting matrices (*q*-numbers). The
suggestion already there is that the radiation field, not only the material atoms, should be treated by
the same matrix methods.

**Quantizing the radiation field (Dirac 1927).** Dirac (Proc. Roy. Soc. London A 114, 243 and 710, 1927)
quantizes the electromagnetic field: it is an assembly of harmonic oscillators, and the field amplitude
becomes an operator built from creation and annihilation operators. A mode in occupation number `n`
is raised to `n+1` by a creation operator and lowered by an annihilation operator; the number operator
counts photons, and its eigenvalues are `0,1,2,…`. Light quanta, and every effect explained by assuming
light quanta, follow from the quantization of the wave.

**The Bose-Einstein matter gas (Jordan 1927; Jordan and Klein 1927).** The de Broglie matter wave is
quantized the same way. For each single-particle mode `r` one introduces amplitudes `b_r`, `b_r^†` and
the occupation-number operator `N_r = b_r^† b_r`, with a conjugate phase `Θ_r`. The defining relations
realized by a representation are
```
b_r^† b_r = N_r ,     b_r b_r^† = 1 + N_r ,     e^{(2πi/h) Θ_r} = (phase conjugate to N_r) ,
```
so that `[b_r, b_r^†] = 1`. The eigenvalues of `N_r` are then `0,1,2,3,…` — Einstein (Bose) statistics.
The matrix elements of the field amplitudes carry factors built from `(1 + m_r)` and `m_r`; with these
factors the second-quantized theory reproduces, on ordinary 3D space, the Einstein density-fluctuation
formula, the mean square fluctuation being proportional to `n_r (1 + n_r)`. The whole machinery —
amplitudes, number operators, conjugate phases, the energy written as `Σ_{rs} H_{rs} b_r^† b_s` — is in
place and understood for bosons.

**Pauli's fluctuation result for fermions.** Pauli's calculation of the analogous fluctuation for a
Fermi gas gives a mean square proportional to `n_r (1 − n_r)`. The original content of the Pauli
principle is that no two electrons can occupy the same single-particle state — equivalently, the
many-electron wave function is antisymmetric under particle exchange, the Slater-determinant /
Heisenberg-Dirac form.

**The single-mode two-state algebra is available.** Independently of the many-body question, a single
two-state mode has a known matrix realization: with
```
b = (0 1 ; 0 0) ,   b^† = (0 0 ; 1 0) ,   N = (0 0 ; 0 1) ,
```
one has `b^† b = N`, `b b^† = 1 − N`, `b^2 = 0`, `(b^†)^2 = 0`, and `N` has eigenvalues `{0,1}`. These
are, up to relabeling, the Pauli spin-1/2 matrices: `b = σ^-`, `b^† = σ^+`, and `1 − 2N = −σ^z`. So a
single fermionic mode and a single spin-1/2 are the same two-level system. The difficulty is entirely in
how to combine `K` such modes.

**The exchange sign and the fixed ordering.** The antisymmetric `N`-particle amplitude is the
Heisenberg-Dirac determinant
```
Ψ = (1/√N!) Σ_perm ε_n  Φ(β^(1), q^(n_1)) Φ(β^(2), q^(n_2)) ⋯ ,
```
`ε_n = ±1` for even/odd permutations. This object only has a *single-valued* meaning once one fixes, once
and for all, an ordering of the single-particle eigenvalues, `β'_1 < β'_2 < ⋯ < β'_K` (the symbol `<` is
just a chosen order, not a magnitude). Reordering the arguments of an antisymmetric function multiplies it
by `(−1)` raised to the number of transpositions. This bookkeeping of signs is the pre-method fact that
any operator description of adding or removing a particle will have to respect.

## Baselines

**Bose second quantization (Jordan; Jordan–Klein 1927) — the method this reacts to.** Core idea: amplitudes
`b_r, b_r^†` with `[b_r, b_s^†] = δ_{rs}`, number operator `N_r = b_r^† b_r`, energy `Σ_{rs} H_{rs} b_r^† b_s`,
field expanded as `ψ = Σ_r b_r u_r` in single-particle modes `u_r`. The transformation law between two
single-particle bases is `b_α(β') = Σ_{q'} Φ_{αp}(β', q') b_p(q')`, and the commuting amplitudes
`b_α(β') b_β(β'') − b_β(β'') b_α(β') = 0` reproduce a *symmetric* many-body space. The eigenvalues of
`N_r` run `0,1,2,…`.

**Imposing single occupancy as an external constraint.** One could keep `[b,b^†]=1` and simply *declare*
that physical states have `n_r ≤ 1`, treating this as a selection rule on top of the bosonic algebra.

**First-quantized antisymmetric wave functions (Heisenberg–Dirac determinants).** Core idea: write the
`N`-electron state as the antisymmetric determinant above in the 3N-dimensional configuration space; the
Pauli principle is encoded as the vanishing of the determinant when two single-particle states coincide.
The antisymmetry is enforced by construction (the determinant form), and the indistinguishability
bookkeeping (the `1/√N!`, the permutation sum, the fixed ordering) is carried along explicitly in every
calculation.

## Evaluation settings

This is a theoretical construction; the natural yardsticks are internal consistency checks and recovery
of known results, not benchmark datasets.
- **Statistics of the occupation number.** The constructed number operator `N_r` must have eigenvalues
  exactly `{0,1}` and the number operators of different modes must be simultaneously diagonalizable
  (mutually commuting), so that a joint occupation-number basis exists.
- **Exchange behavior.** Removing/adding particles in two different modes in the opposite order must
  produce the antisymmetric sign; the many-body space generated must be the antisymmetric one.
- **Recovery of the fluctuation formula.** The density-fluctuation mean square computed from the new
  amplitudes must come out proportional to `n_r (1 − n_r)`, the Fermi form, matching Pauli's result, and
  the corrected field expansion must leave the fluctuation result unchanged.
- **Equivalence to the configuration-space theory.** The second-quantized energy `Σ_{rs} H_{rs} a_r^† a_s`
  in the occupation-number space must be matrix-equivalent to the ordinary antisymmetric many-electron
  operator `V = V_1 + ⋯ + V_N` acting on the determinant wave functions, including inner products.
- **A concrete one-dimensional testbed.** A vibrating string / one-dimensional continuum
  `∂²ψ/∂x² = ∂²ψ/∂t²` with fixed ends `ψ(0,t)=ψ(l,t)=0`, expanded in standing modes
  `ψ = Σ_r (amplitude) sin(rπx/l)`, is the simplest field on which to display the construction and its
  fluctuation, and a one-dimensional lattice of two-level systems is the simplest discrete analogue.

## Code framework

Pre-existing primitives: dense linear algebra over small complex matrices (Kronecker/tensor products to
build many-mode operators, eigen-decomposition, the discrete Fourier transform on a periodic lattice).
For one two-level mode the ladder and number matrices `b, b^†, N` already exist; tensoring them places a
single-mode operator on a chosen mode of a `K`-mode register.

```python
import numpy as np

I  = np.eye(2)
b  = np.array([[0, 1], [0, 0]], dtype=complex)   # single-mode lowering / two-level
bd = b.conj().T                                  # single-mode raising
Nop = bd @ b                                      # single-mode number, eigenvalues {0,1}

def on_mode(local, j, K):
    """Place a single-mode 2x2 operator on mode j of a K-mode register (tensor product)."""
    m = np.array([[1]], dtype=complex)
    for k in range(K):
        m = np.kron(m, local if k == j else I)
    return m

def field_annihilation(j, K):
    """Many-mode amplitude a_j built from the single-mode pieces on a K-mode register.

    The single-mode operators on different modes commute under `on_mode`. The map
    that turns them into amplitudes with the correct many-particle statistics is the
    object to be designed here.
    """
    # TODO: the operators we will define here
    pass

def field_creation(j, K):
    # TODO: the operators we will define here
    pass

def second_quantized_energy(H, K):
    """Energy Sigma_{rs} H[r,s] a_r^dagger a_s on the K-mode register, once the
    amplitudes above are defined."""
    dim = 2 ** K
    V = np.zeros((dim, dim), dtype=complex)
    for r in range(K):
        for s in range(K):
            V += H[r, s] * (field_creation(r, K) @ field_annihilation(s, K))
    return V
```
