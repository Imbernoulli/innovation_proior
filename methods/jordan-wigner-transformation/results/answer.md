# The Jordan-Wigner transformation

## Problem solved

Build a quantized matter-wave field whose quanta obey the Pauli exclusion principle: an operator algebra
of mode amplitudes `a_r, a_r^†` such that (i) every occupation number `N_r = a_r^† a_r` has eigenvalues
only `{0,1}`, and (ii) the many-particle state space is antisymmetric under exchange. The bosonic field
algebra `[b_r, b_s^†] = δ_{rs}` fails on both counts — it gives occupation `0,1,2,…` and a symmetric
(commuting) exchange. Equivalently, on a one-dimensional lattice the construction is an exact map between
spin-1/2 operators (which commute on different sites) and fermion operators (which anticommute), turning
quantum spin chains into fermion problems.

## Key idea

A single mode capped at one particle is a two-level system: with `b = σ^- = (0\,1;0\,0)`,
`b^† = σ^+`, `N = b^†b`, one has `b^† b = N`, `b b^† = 1 − N`, `b^2 = 0`, and `1 − 2N = −σ^z`. The cap is
automatic per mode, but tensoring `K` such modes makes operators on different modes **commute** — the
wrong statistics. The missing exchange sign lives in the antisymmetric many-electron determinant, which is
single-valued only after a global ordering `1 < 2 < ⋯ < K` of the modes is fixed; removing a particle from
mode `j` then costs `(−1)` raised to the number of occupied modes to its **left**. Writing that sign as an
operator — the **Jordan-Wigner string** — and attaching it to the bare two-level operator converts the
commuting pieces into anticommuting fermion amplitudes.

## The transformation (final form)

String / parity operator (an involution, eigenvalues `±1`):
```
v_j = ∏_{l < j} (1 − 2 N_l) = ∏_{l < j} (−σ^z_l) = exp( i π Σ_{l<j} N_l ),     v_j^2 = 1.
```

Fermion amplitudes from the two-level (spin) operators:
```
a_j = v_j · σ^-_j = [∏_{l<j} (−σ^z_l)] σ^-_j ,        a_j^† = σ^+_j · v_j = [∏_{l<j} (−σ^z_l)] σ^+_j ,
```
and the inverse dictionary, spins from fermions:
```
S^z_j = N_j − 1/2 ,      S^+_j = a_j^† e^{ i π Σ_{l<j} N_l} ,      S^-_j = a_j e^{ −i π Σ_{l<j} N_l} ,
```
with `N_j = a_j^† a_j` and `σ^z_j = 2 N_j − 1` (so `1 − 2N_j = −σ^z_j` is the parity factor). In the
standard-Pauli convention, where `Z = diag(1, −1) = 1 − 2N` is the parity (`X, Y, Z` the usual Pauli
matrices), this is `a_j^† = Z_0 ⋯ Z_{j−1} (X_j − i Y_j)/2`, `a_j = Z_0 ⋯ Z_{j−1} (X_j + i Y_j)/2`.

Explicit matrix elements in the `2^K`-state occupation basis `x = (x_1,…,x_K)`, `x_k ∈ {0,1}`:
```
a_j(x; y)   = (−1)^{x_1 + ⋯ + x_{j−1}} · (∏_{k≠j} δ_{x_k y_k}) · δ_{x_j, 0} δ_{y_j, 1} ,
a_j^†(x; y) = (−1)^{x_1 + ⋯ + x_{j−1}} · (∏_{k≠j} δ_{x_k y_k}) · δ_{x_j, 1} δ_{y_j, 0} .
```

## Result: the canonical anticommutation relations are forced

Because `b` anticommutes with `1 − 2N` on its own mode (`b(1−2N) = −(1−2N)b`), the dressed amplitudes satisfy
```
{a_i, a_j} = 0 ,        {a_i^†, a_j^†} = 0 ,        {a_i^†, a_j} = δ_{ij} .
```
**Proof of the converse (the physics is the algebra):** from `{a_r^†,a_r}=1` and `a_r^2 = 0`,
```
N_r (1 − N_r) = a_r^† a_r · a_r a_r^† = a_r^† (a_r a_r) a_r^† = 0   ⟹   N_r^2 = N_r ⟹ N_r ∈ {0,1},
```
the Pauli cap, and `[N_r, N_s] = 0` follows from the anticommutators — so single occupancy and
antisymmetry are *consequences* of the multiplication rules, not extra postulates. The density fluctuation
of the corrected field `ψ = Σ_r a_r u_r` comes out `\overline{Δ²} ∝ n_r(1 − n_r)`, the Fermi form.

## Equivalence to the many-body theory and uniqueness

The second-quantized one-body energy `Ω = Σ_{κλ} H_{κλ} a_κ^† a_λ` on the occupation-number space is
unitarily equivalent to the ordinary antisymmetric `V = V_1 + ⋯ + V_N` on configuration-space
determinants; the matrix-element signs match because the string is built from the determinant's own
reordering parity. **Uniqueness (Clifford / Majorana coda):** the Hermitian combinations
```
α_κ = a_κ + a_κ^† ,        α_{K+κ} = (1/i)(a_κ − a_κ^†)     (κ = 1,…,K)
```
satisfy the Clifford/Dirac relation
```
α_κ α_λ + α_λ α_κ = 2 δ_{κλ}     ⟹    α_κ^2 = 1,   α_κ α_λ = − α_λ α_κ  (κ ≠ λ).
```
The `2K` matrices `α` together with `−1` generate a group of order `2^{2K+1}`, center `{1,−1}`, with
`2^{2K} + 1` conjugacy classes; it has `2^{2K}` one-dimensional reps plus exactly one of dimension `d`
with `2^{2K}·1 + d^2 = 2^{2K+1}`, i.e. `d = 2^K`. That unique `2^K`-dimensional irreducible representation
is the string construction, so the anticommutation relations determine `a, a^†` up to a unitary
(canonical) transformation.

## Worked example: 1D XX chain → free fermions

On a nearest-neighbor bond the strings of `S^+_j` and `S^-_{j+1}` telescope, leaving
`S^+_j S^-_{j+1} = a_j^†(1−2N_j)a_{j+1} = a_j^† a_{j+1}`. Hence
```
H = −J Σ_j (S^x_j S^x_{j+1} + S^y_j S^y_{j+1}) = −(J/2) Σ_j (a_j^† a_{j+1} + a_{j+1}^† a_j),
```
a free fermion gas. On a periodic chain, `a_j = (1/√N) Σ_k ã_k e^{ikj}` diagonalizes it:
```
H = Σ_k ω_k ã_k^† ã_k ,        ω_k = − J cos k .
```
The ground state fills all `ω_k < 0`; the magnetization is `⟨Σ_j S^z_j⟩ = ⟨Σ_j (N_j − 1/2)⟩`. An added
`−J_z Σ_j S^z_j S^z_{j+1} = −J_z Σ_j (N_j − 1/2)(N_{j+1} − 1/2)` becomes a quartic density-density
interaction, so freeness is special to the bilinear (`XX` / transverse-Ising) lines; the map itself is
exact for any couplings.

## Implementation

```python
import numpy as np

# single two-level mode  ==  one spin-1/2  ==  one Fermi mode
I  = np.eye(2)
sm = np.array([[0, 1], [0, 0]], dtype=complex)   # b = sigma^- (lowering)
sp = sm.conj().T                                 # b^dagger = sigma^+
n1 = sp @ sm                                     # single-mode N, eigenvalues {0,1}
mz = I - 2 * n1                                  # 1 - 2N = -sigma^z (the string's local factor)

def on_mode(local, j, K):
    m = np.array([[1]], dtype=complex)
    for k in range(K):
        m = np.kron(m, local if k == j else I)
    return m

def jw_string(j, K):                             # prod_{l<j} (1 - 2 N_l) = prod_{l<j} (-sigma^z_l)
    s = np.eye(2 ** K, dtype=complex)
    for l in range(j):
        s = s @ on_mode(mz, l, K)
    return s

def a(j, K):    return jw_string(j, K) @ on_mode(sm, j, K)    # a_j      = string . b_j
def adag(j, K): return on_mode(sp, j, K) @ jw_string(j, K)    # a_j^dag  = b_j^dag . string

# canonical anticommutation relations
K = 4
for i in range(K):
    for j in range(K):
        tgt = np.eye(2 ** K) if i == j else np.zeros((2 ** K, 2 ** K))
        assert np.allclose(adag(i, K) @ a(j, K) + a(j, K) @ adag(i, K), tgt)   # {a_i^dag,a_j}=d_ij
        assert np.allclose(a(i, K) @ a(j, K) + a(j, K) @ a(i, K), 0)           # {a_i,a_j}=0

# XX spin chain == free fermions (exact)
J = 1.0
H_spin = sum(-(J / 2) * (on_mode(sp, j, K) @ on_mode(sm, j + 1, K)
                         + on_mode(sm, j, K) @ on_mode(sp, j + 1, K)) for j in range(K - 1))
H_free = sum(-(J / 2) * (adag(j, K) @ a(j + 1, K) + adag(j + 1, K) @ a(j, K)) for j in range(K - 1))
assert np.allclose(H_spin, H_free)

# periodic dispersion
N = 8
omega = [-J * np.cos(2 * np.pi * m / N) for m in range(N)]   # omega_k = -J cos k
```

The string operator `∏_{l<j}(−σ^z_l)` is the single nonlocal ingredient; it is what converts
commuting local spin-1/2 (or capped-boson) operators into anticommuting fermion amplitudes, making the
Pauli exclusion principle and Fermi statistics emerge from the algebra of a quantized wave field.
