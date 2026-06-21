# The LCAO Self-Consistent Field: Hartree–Fock as the matrix eigenproblem FC = SCε

## Problem

Compute the closed-shell electronic ground state (energy and orbitals) of a molecule with fixed nuclei.
The exact single-determinant (Hartree–Fock) condition is a one-electron eigenvalue equation
**F φ_i = ε_i φ_i**, but for a molecule φ_i is an arbitrary three-dimensional function and F is a nonlinear
integro-differential operator (it contains the Laplacian and the nonlocal exchange kernel) with no
separation of variables — solvable on a grid for atoms (by spherical symmetry), impractical for molecules.

## Key idea

Expand each molecular orbital in a **fixed, finite set of atom-centered basis functions** {χ_p} (LCAO):

  φ_i = Σ_p C_{pi} χ_p.

The unknowns become a finite list of coefficients C_{pi}; the variational principle keeps the energy an
upper bound and turns "solve a PDE" into "minimize over numbers." The atom-centered functions on different
nuclei are **non-orthogonal**, so their overlap matrix S ≠ I is carried explicitly. Minimizing the
closed-shell energy under the orbital-orthonormality constraint C_i† S C_j = δ_{ij} (Lagrange multipliers;
the Hermitian multiplier matrix diagonalized using the determinant's invariance under unitary mixing of
occupied orbitals) collapses every function integral into a fixed matrix and yields a **generalized matrix
eigenvalue problem**. Because the Fock matrix is built from the occupied orbitals, it is solved
**self-consistently** (SCF) by iteration, each step a matrix diagonalization rather than a numerical PDE solve.

## Final equations

Closed shell, N = 2n electrons, n doubly-occupied spatial orbitals; m basis functions (m ≥ n).

**Generalized eigenproblem (Roothaan–Hall):**

  F C = S C ε,    equivalently  F c_i = ε_i S c_i,   secular equation  Det(F − ε S) = 0.

m real roots ε_i; fill the n lowest (aufbau) as occupied, the rest are virtual. S = I recovers the ordinary
eigenproblem.

**Matrices (integrals over the basis {χ_p}):**

  S_{pq}   = ∫ χ_p* χ_q dv                                  (overlap; diagonal = 1)
  H^core_{pq} = ∫ χ_p* ( −½∇² − Σ_A Z_A/r_A ) χ_q dv         (kinetic + nuclear attraction)
  (pq|rs)  = ∫∫ χ_p(1)χ_q(1) (1/r₁₂) χ_r(2)χ_s(2) dv₁dv₂     (two-electron repulsion, Mulliken notation)

**Density matrix (closed shell):**

  D_{κλ} = 2 Σ_{i ∈ occ} C_{κi} C_{λi}.

**Fock matrix:**

  F_{μν} = H^core_{μν} + Σ_{κλ} D_{κλ} [ (μν|κλ) − ½ (μκ|νλ) ].

The first term is the classical Coulomb field of the whole electron density; the second (minus sign,
factor ½) is exchange, the term forced by antisymmetry of the determinant (absent in a plain Hartree mean field).

**Total energy:**

  E = ½ Σ_{μν} D_{μν} ( H^core_{μν} + F_{μν} ) + E_nuc,

with E_nuc = Σ_{A<B} Z_A Z_B / R_{AB} the nuclear repulsion.

**Solving the generalized problem (Löwdin symmetric orthogonalization).** Diagonalize S = U s Uᵀ and form
X = S^{-1/2} = U s^{-1/2} Uᵀ (the orthonormalizer least-distorting from the original AOs). Transform
F′ = Xᵀ F X, solve the ordinary symmetric eigenproblem F′ C′ = C′ ε, back-transform C = X C′.

## SCF algorithm

1. Compute once: S, H^core, the two-electron integrals (pq|rs), E_nuc, and X = S^{-1/2}.
2. Guess a density D (e.g. D = 0, so the first F = H^core).
3. Build F = F(D); compute E = ½ Σ D⊙(H^core + F) + E_nuc.
4. F′ = Xᵀ F X; diagonalize → ε, C′; C = X C′.
5. New density D = 2 Σ_{i∈occ} C_{·i} C_{·i}ᵀ from the n lowest orbitals.
6. Repeat 3–5 until ΔE and rms(ΔD) fall below thresholds.

## Reference code (closed-shell RHF SCF)

```python
import numpy as np
from scipy.linalg import fractional_matrix_power

def density_matrix(C, nocc):
    # D_{kl} = 2 * sum_{i in occ} C_{ki} C_{li}  (double occupancy)
    Cocc = C[:, :nocc]
    return 2.0 * Cocc @ Cocc.T

def fock_matrix(Hcore, eri, D):
    # F_{uv} = Hcore_{uv} + sum_{kl} D_{kl} [ (uv|kl) - 1/2 (uk|vl) ]
    J = np.einsum("uvkl,kl->uv", eri, D)   # Coulomb
    K = np.einsum("ukvl,kl->uv", eri, D)   # exchange
    return Hcore + J - 0.5 * K

def total_energy(Hcore, F, D, e_nuclear):
    # E = 1/2 sum_{uv} D_{uv} (Hcore_{uv} + F_{uv}) + E_nuc
    return 0.5 * np.sum(D * (Hcore + F)) + e_nuclear

def scf(S, Hcore, eri, e_nuclear, n_electrons,
        max_iter=64, e_tol=1e-10, d_tol=1e-8, verbose=True):
    nbf  = S.shape[0]
    nocc = n_electrons // 2
    X    = fractional_matrix_power(S, -0.5)        # S^{-1/2}, symmetric (Loewdin) orthogonalizer

    D, E_old = np.zeros((nbf, nbf)), 0.0           # zero-density guess -> first Fock is bare Hcore
    for it in range(max_iter):
        F = fock_matrix(Hcore, eri, D)
        E = total_energy(Hcore, F, D, e_nuclear)

        Fp        = X.T @ F @ X                    # F' = S^{-1/2} F S^{-1/2}
        eps, Cp   = np.linalg.eigh(Fp)            # ordinary symmetric eigenproblem F' C' = C' eps
        C         = X @ Cp                         # back-transform C = S^{-1/2} C'
        D_new     = density_matrix(C, nocc)        # rebuild from n lowest orbitals (aufbau)

        dE, rmsD = abs(E - E_old), np.sqrt(np.mean((D_new - D) ** 2))
        if verbose:
            print(f"iter {it:2d}  E = {E:.10f}  dE = {dE:.2e}  rmsD = {rmsD:.2e}")
        if dE < e_tol and rmsD < d_tol:
            return E, C, eps, D_new
        D, E_old = D_new, E
    raise RuntimeError("SCF did not converge")
```

For water (O–H = 1.1 Å, ∠HOH = 104°) in a minimal STO-3G basis, this converges to a total SCF energy of
about **−74.9421 Hartree**, with the iteration history showing monotone convergence of E and a decaying
density change — the closed-shell self-consistent field for a molecule, obtained entirely by linear algebra.

## Optional acceleration: symmetry blocking

If the Fock operator carries the molecular point-group symmetry, MOs transform as irreducible
representations and F, S are block-diagonal in a symmetry-adapted basis; the secular equation factors into
one small block per irrep, Det(F^Γ − ε S^Γ) = 0, cutting the diagonalization cost and labeling orbitals by
symmetry. Same answer, found faster.
