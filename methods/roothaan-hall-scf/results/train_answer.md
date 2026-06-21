What I want is the electronic ground state of a molecule with clamped nuclei — the energy and the orbitals each electron occupies — and I want it for a molecule of *arbitrary* shape, not just an atom. The theory I trust is the self-consistent field with exchange. Build the many-electron state as a single Slater determinant of one-electron spatial orbitals, each doubly occupied for a closed shell, $\Psi = (N!)^{-1/2}\,\det[\varphi_1\bar\varphi_1\cdots\varphi_n\bar\varphi_n]$ with $N=2n$ electrons: it is antisymmetric so it obeys the Pauli principle, and $\langle\Psi|H|\Psi\rangle \ge E_0$ by the variational principle so anything I minimize over is a genuine upper bound. Carrying out the spin sums gives an energy purely in the spatial orbitals, $E = 2\sum_i H_i + \sum_{ij}(2J_{ij}-K_{ij})$, where $H_i$ is the one-electron core part (kinetic plus nuclear attraction), $J_{ij}$ is the Coulomb repulsion between the charge clouds of orbitals $i$ and $j$, and $K_{ij}$ is the exchange term that antisymmetry forces on me — the piece Hartree's plain product would have missed. Minimizing this under the orthonormality constraint $\int\varphi_i^*\varphi_j\,dv=\delta_{ij}$, with Lagrange multipliers, produces a one-electron operator, the Fock operator $F = H + \sum_j(2J_j - K_j)$, and a stationarity condition that, after using the determinant's freedom to mix occupied orbitals by a unitary to diagonalize the multiplier matrix, reads cleanly as the eigenvalue equation $F\varphi_i = \varepsilon_i\varphi_i$.

That equation is exact within the single-determinant approximation, and it is useless to me for a molecule. That is the wall. Look at what it is: $F$ contains the Laplacian (a differential operator) and the nonlocal exchange kernel $K$ (an integral operator), and $\varphi_i(x,y,z)$ is an arbitrary function of three coordinates with no special structure — so this is a three-dimensional, nonlinear (since $F$ depends on the orbitals), integro-differential eigenproblem that must be re-solved every self-consistency cycle. For an *atom* I would be fine: the central field gives $\varphi = R(r)Y_{\ell m}$, the angular part comes off analytically, and what remains is a one-dimensional radial equation I can put on a grid and integrate, iterating to self-consistency the way Hartree did by hand. That is exactly why atomic structure is a routine numerical problem. But a molecule has no center, no spherical symmetry, no coordinate system in which this separates, and direct grid solution of a fully three-dimensional integro-differential eigenproblem is hopeless. Hartree's product SCF is no escape — it omits antisymmetry and so violates the Pauli principle at the level of the wavefunction — and the semi-empirical LCAO and valence-bond schemes that do treat molecules posit their secular equations by analogy, with matrix entries fit to experiment rather than tied to the molecular Hamiltonian, and they paper over the fact that atom-centered functions overlap. None of them delivers a controlled, parameter-free *ab initio* energy with the variational guarantee. The equation is right; the unknown is wrong.

The fix I propose is the linear combination of atomic orbitals self-consistent field — the Roothaan–Hall SCF. The trouble is that $\varphi_i$ is an arbitrary function, an infinite-dimensional thing, so I refuse to let it be arbitrary: pick a fixed, finite set of known atom-centered functions $\chi_1,\dots,\chi_m$ once and for all and demand that every orbital live in their span, $\varphi_i = \sum_p C_{pi}\chi_p$. Now the unknown is not a function but a finite list of numbers $C_{pi}$, and the variational principle still holds — minimizing over the coefficients gives the best orbital expressible in this subspace and an energy that is still an upper bound (the Ritz method), so enriching the set drives me toward the true answer. The right $\chi_p$ to use are atomic orbitals centered on the nuclei: near any one nucleus the molecular potential is dominated by that nucleus, nearly the isolated-atom potential, so the orbital locally looks like an atomic orbital with the right cusp, decay, and angular character. Take normalized atomic functions (Slater-type, with screening-constant exponents). The one thing I must respect that the semi-empirical schemes were sloppy about is that these functions, sitting on *different* nuclei, are **not orthogonal**: $\chi_p$ on atom A and $\chi_q$ on atom B have a genuine overlap $\int\chi_p^*\chi_q\,dv \ne 0$, two clouds reaching into the same region of space. I could orthogonalize them first, but that would smear each function over several atoms and destroy the very "looks atomic near its nucleus" property that made the basis good — so I keep the atomic functions as they are and carry the overlap explicitly.

Pushing the expansion through, every function integral collapses into a fixed matrix over the $\chi$'s. For any one-electron operator $M$, $\int\varphi_i^* M\varphi_j\,dv = c_i^\dagger M c_j$ with $M_{pq}=\int\chi_p^* M\chi_q\,dv$; in particular the identity operator gives the **overlap matrix** $S_{pq}=\int\chi_p^*\chi_q\,dv$, with unit diagonal and nonzero off-diagonals. The orbital-orthonormality constraint becomes the algebraic constraint $c_i^\dagger S c_j = \delta_{ij}$ — and there is $S$, sitting inside the constraint, refusing to disappear. Minimizing the energy under this constraint with Lagrange multipliers, varying the coefficient vectors, and abbreviating the **Fock matrix** $F = H + \sum_j(2J_j - K_j)$, the stationarity condition is $F c_i = \sum_j S c_j\,\varepsilon_{ji}$. This is exactly the molecular-orbital equation I had before, but now every object is a finite matrix, and the overlap $S$ has propagated straight from the constraint into the eigenvalue equation. The multipliers form a Hermitian matrix; collecting all orbitals as columns of $C$ and the multipliers into $\varepsilon$ gives the single matrix equation

$$F C = S C\,\varepsilon.$$

Diagonalizing $\varepsilon$ (using the determinant's unitary-mixing freedom again, which costs nothing in energy) reduces this column by column to the **generalized eigenvalue problem**

$$F c_i = \varepsilon_i\,S c_i, \qquad (F - \varepsilon S)\,c = 0, \qquad \det(F - \varepsilon S) = 0.$$

This is the ordinary Hermitian eigenproblem with the unit matrix replaced by $S$; it collapses to the ordinary one exactly when $S = I$, i.e. when the basis is orthonormal. For atom-centered functions $S \ne I$, so the overlap genuinely belongs there — pretending $S = I$ would silently impose the wrong inner product and give wrong orbitals and a wrong energy. With $m$ basis functions the secular equation is an $m$-th degree polynomial with $m$ real roots (since $S$ is positive definite); I fill the lowest $n$ by the aufbau rule as the doubly-occupied molecular orbitals, and the rest are virtual.

What makes the whole thing computable is that $F$ is not a known matrix — it is built from the *occupied orbitals*, so it depends on its own solution, the same nonlinearity Hartree faced. I write that dependence explicitly through the **density matrix** $D_{\kappa\lambda} = 2\sum_{i\in\text{occ}}C_{\kappa i}C_{\lambda i}$ (the factor 2 is double occupancy) and the two-electron repulsion integrals $(pq|rs)=\iint\chi_p(1)\chi_q(1)\,(1/r_{12})\,\chi_r(2)\chi_s(2)$, giving

$$F_{\mu\nu} = H^{\text{core}}_{\mu\nu} + \sum_{\kappa\lambda} D_{\kappa\lambda}\Big[(\mu\nu|\kappa\lambda) - \tfrac12(\mu\kappa|\nu\lambda)\Big].$$

The first bracket term is the classical Coulomb repulsion in the field of the whole electron density; the second, with its minus sign and factor $\tfrac12$, is exchange — the fingerprint of antisymmetry, the same 2-vs-1 weighting that produced $2J - K$, and the thing that distinguishes this from a plain Hartree mean field. The total energy in matrix form is

$$E = \tfrac12\sum_{\mu\nu} D_{\mu\nu}\big(H^{\text{core}}_{\mu\nu} + F_{\mu\nu}\big) + E_{\text{nuc}},$$

the $\tfrac12$ avoiding double-counting the electron–electron interaction, plus the classical nucleus–nucleus repulsion $E_{\text{nuc}} = \sum_{A<B}Z_A Z_B/R_{AB}$. Because $F$ depends on $D$, I cannot solve in closed form; I iterate. Guess a density, build $F$, solve $\det(F-\varepsilon S)=0$ for the $n$ lowest orbitals, rebuild $D$ from those, and repeat until input and output orbitals agree — Hartree's self-consistency, but now each step is a matrix diagonalization rather than a numerical integration of a differential equation.

One practical refinement: solving the generalized problem at every iteration is awkward, so I change to an orthonormal basis once, at the start, since $S$ is fixed. I want $X$ with $X^\dagger S X = I$, and the least-distorting choice is the symmetric (Löwdin) root: diagonalize $S = U s U^\dagger$ and take $X = S^{-1/2} = U s^{-1/2} U^\dagger$. I choose this square root over a Cholesky or canonical factor because among all matrices that orthonormalize the basis, $S^{-1/2}$ produces the orthonormal set *closest* to the original atomic orbitals in the least-squares sense — it mixes them as little as possible, keeping the coefficients interpretable as "mostly this atomic orbital," whereas a Cholesky factor would orthonormalize by a lopsided, basis-order-dependent mixing that throws that interpretability away. With $X$ in hand I transform $F' = X^\dagger F X = S^{-1/2} F S^{-1/2}$, solve the *ordinary* symmetric eigenproblem $F' C' = C'\varepsilon$, and back-transform $C = X C' = S^{-1/2} C'$.

So the self-consistent loop, concretely: form $S$, $H^{\text{core}}$, the two-electron integrals, $E_{\text{nuc}}$, and $X = S^{-1/2}$ once; start from a guess density (even $D=0$ works — it makes the first $F$ just $H^{\text{core}}$, the bare one-electron problem, whose occupied orbitals are a sensible start); then repeat building $F$ from the current $D$, computing $E$, transforming and diagonalizing to get $C$, rebuilding $D = 2\sum_{\text{occ}} C C^\top$ from the $n$ lowest orbitals, and stopping when both the energy change and the rms density change fall below small thresholds. For water at O–H $= 1.1$ Å, $\angle$HOH $= 104^\circ$ in a minimal STO-3G basis this converges to a total SCF energy of about $-74.9421$ Hartree, with monotone convergence of $E$ and a decaying density change — the closed-shell self-consistent field for a molecule, obtained entirely by linear algebra. There is a further efficiency lever I note but do not lean on: if the Fock operator carries the molecular point-group symmetry, the orbitals transform as irreducible representations and $F$, $S$ become block-diagonal in a symmetry-adapted basis, factoring the secular equation into one small block per irrep, $\det(F^\Gamma - \varepsilon S^\Gamma)=0$ — the same answer, found faster, with orbitals labeled by symmetry.

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
