I want the electronic ground state of a molecule — the energy and the orbitals — and I want it for a molecule of *arbitrary* shape, not just an atom.

The object I trust is the self-consistent field with exchange. Start from a single determinant of one-electron spatial orbitals, each doubly occupied for a closed shell: Ψ = (N!)^{-1/2} det[φ₁ φ̄₁ φ₂ φ̄₂ … φₙ φ̄ₙ], with N = 2n electrons. The determinant is antisymmetric, so it respects the Pauli principle, and ⟨Ψ|H|Ψ⟩ ≥ E₀ by the variational principle, so whatever I minimize over is a genuine upper bound. Doing the spin sums on this closed-shell determinant gives an energy that is purely in terms of the spatial orbitals,

E = 2 Σᵢ Hᵢ + Σᵢⱼ (2Jᵢⱼ − Kᵢⱼ),

where Hᵢ = ∫ φᵢ* H φᵢ dv is the one-electron core part (kinetic energy plus attraction to the nuclei), Jᵢⱼ = ∫∫ φᵢ*(1)φᵢ(1) (1/r₁₂) φⱼ*(2)φⱼ(2) is the Coulomb repulsion between the charge cloud of orbital i and that of orbital j, and Kᵢⱼ = ∫∫ φᵢ*(1)φⱼ(1) (1/r₁₂) φⱼ*(2)φᵢ(2) is the exchange term that the antisymmetry of the determinant forces on me — the term Hartree's plain product would have missed. The factor 2 on Hᵢ is double occupancy; the 2Jᵢⱼ − Kᵢⱼ is what survives after the spin integrations (parallel-spin pairs contribute exchange, opposite-spin pairs don't).

Minimize this energy over the orbitals, subject to the constraint that they stay orthonormal, ∫ φᵢ* φⱼ dv = δᵢⱼ. Lagrange multipliers for the constraint, vary the φᵢ, set the variation to zero, and out drops a one-electron operator — the Fock operator,

F = H + Σⱼ (2Jⱼ − Kⱼ),

with Jⱼ the operator that multiplies by the averaged Coulomb potential of orbital j and Kⱼ the *nonlocal* exchange operator (acting on a function it returns an integral over all space weighted by φⱼ). The stationarity condition is F φᵢ = Σⱼ φⱼ εⱼᵢ, where the εⱼᵢ are the multipliers; they form a Hermitian matrix (they have to, for the energy at the stationary point to be real). And because the determinant is invariant under a unitary mixing of the occupied orbitals among themselves — mixing the columns of a determinant by a unitary matrix only multiplies it by a phase — I'm free to choose that mixing to diagonalize the multiplier matrix. Do that and the equation becomes clean:

F φᵢ = εᵢ φᵢ.

So far so good. This is exact within the single-determinant approximation, and it's a beautiful eigenvalue equation. And it's useless to me for a molecule. That's the wall.

Why useless? Look at what F φᵢ = εᵢ φᵢ actually *is*. F contains H, which has the Laplacian — a differential operator — and it contains K, which is an integral kernel. So this is a three-dimensional, nonlinear (F depends on the φ's), integro-differential eigenvalue equation, and φᵢ(x,y,z) is an arbitrary function of three coordinates with no special structure. For an *atom* I'd be fine: the central field means φ = R(r)Yₗₘ(θ,φ), the angular part comes off analytically, and I'm left with a one-dimensional radial equation R(r) that I can put on a grid and integrate, iterating to self-consistency the way Hartree did by hand. That's exactly why atomic structure is a solved, routine numerical problem. But a molecule has no center, no spherical symmetry, no coordinate system in which this separates. I'd be asking to integrate a nonlinear integro-differential eigenproblem for a fully three-dimensional unknown function, re-solving it every self-consistency cycle. For anything past the very simplest case that is hopeless. The exactness of F φ = εφ is no consolation if I cannot solve it.

So the equation is right and the unknown is wrong. The trouble is that φᵢ is an *arbitrary function* — an infinite-dimensional thing. What if I refuse to let it be arbitrary? Let me stop searching the whole infinite-dimensional space of functions and search only inside a finite-dimensional subspace that I pick in advance. Pick a fixed, finite set of known functions χ₁, …, χₘ once and for all, and demand that every orbital live in their span:

φᵢ = Σ_p c_{pi} χ_p.

Now the unknown is not a function at all — it is the list of numbers c_{pi}. A finite list. The variational principle is still on my side: minimizing the energy over the c's gives the best orbital *that can be written in this subspace*, and the energy is still an upper bound (it's just an upper bound restricted to a smaller family — the Ritz method). I've traded "the true Hartree–Fock orbital" for "the best orbital expressible in m fixed functions," which is exactly the kind of controlled approximation I want: make the set richer and I approach the true answer.

What should the χ_p be? They should be functions for which a few of them already capture most of the answer, so the subspace is small but good. Stare at a molecular orbital near any one nucleus: the potential there is dominated by that nucleus, nearly the isolated-atom potential, so the orbital locally looks like an *atomic* orbital of that atom — it has the cusp, the right decay, the right angular character. So build the molecular orbitals out of atomic orbitals centered on the nuclei — the linear combination of atomic orbitals. This is also exactly how the semi-empirical people already write molecular orbitals; the difference is that I am going to *derive* the equations the coefficients satisfy from the real many-electron Hamiltonian and the variational principle, not posit them by analogy. Concretely take normalized atomic functions (Slater-type, say, with screening-constant exponents), so ∫ χ_p* χ_p dv = 1.

One thing I have to be careful about and the semi-empirical schemes were sloppy about: these atomic functions, sitting on *different* nuclei, are **not orthogonal**. χ_p on atom A and χ_q on atom B have a genuine overlap ∫ χ_p* χ_q dv ≠ 0 — that's just two clouds reaching into the same region of space. I could try to orthogonalize them first, but that would smear each function over several atoms and destroy the very "looks atomic near its nucleus" property that made the basis good. Better to keep the atomic functions as they are and carry the overlap explicitly. Let me see what that costs.

Push the expansion through. I'll need integrals of every operator over the basis, so for any one-electron operator M define its matrix M_{pq} = ∫ χ_p* M χ_q dv, collected into a matrix M; if M is Hermitian, so is M. In particular the core operator gives H_{pq}, and — the one that's going to matter — the identity operator gives the **overlap matrix**

S_{pq} = ∫ χ_p* χ_q dv,

whose diagonal is 1 (normalized functions) and whose off-diagonal entries are nonzero (non-orthogonality). Writing the orbital as a column vector of coefficients c_i = (c_{1i}, …, c_{mi})ᵀ, an integral like ∫ φᵢ* M φⱼ dv becomes, by substituting the expansion, c_i† M c_j — the function integrals turn into vector–matrix–vector products over fixed matrices. So the overlap of two molecular orbitals is

∫ φᵢ* φⱼ dv = c_i† S c_j,

and the orthonormality constraint I had on the orbitals, ∫ φᵢ* φⱼ = δᵢⱼ, becomes the algebraic constraint **c_i† S c_j = δᵢⱼ**. There's the overlap, sitting inside the constraint. Hold onto that — I have a feeling S is going to refuse to disappear.

Likewise the one-electron part becomes Hᵢ = c_i† H c_i, and the two-electron Coulomb and exchange pieces, built from the operators Jⱼ, Kⱼ over the basis, become c_i† Jⱼ c_i and c_i† Kⱼ c_i — i.e. the whole energy is now a function of the coefficient vectors and a fixed set of integrals over the χ's. Good. Everything is finite. Now I minimize.

Vary the coefficients. Let each c_i change by an infinitesimal δc_i (each coefficient C_{pi} nudged by δC_{pi}). The variation of the energy, mirroring the orbital calculation but now with vectors, is

δE = 2 Σᵢ (δc_i†){ H + Σⱼ (2Jⱼ − Kⱼ) } c_i + (complex conjugate),

which I'll abbreviate by defining the **Fock matrix** F = H + Σⱼ (2Jⱼ − Kⱼ), so

δE = 2 Σᵢ (δc_i†) F c_i + 2 Σᵢ (δc_i)ᵀ F̄ c̄_i.

But the δc_i aren't free — they have to respect the orthonormality constraint c_i† S c_j = δᵢⱼ. Vary that:

(δc_i†) S c_j + c_i† S (δc_j) = 0.

Attach Lagrange multipliers. Multiply this constraint variation by multipliers −2εⱼᵢ and sum over i,j, then add to δE. The combined stationarity condition δE′ = 0, holding for every independent choice of δc_i, forces the bracket multiplying each δc_i† to vanish:

F c_i = Σⱼ S c_j εⱼᵢ.

Stop and look at that. It is *exactly* the molecular-orbital equation F φᵢ = Σⱼ φⱼ εⱼᵢ I had before — but every object is now a finite matrix or vector, and there is an S wedged in on the right-hand side that wasn't there in the orthonormal-orbital version. That S is the overlap, the price of keeping the atomic functions non-orthogonal, and it has propagated straight from the constraint into the eigenvalue equation. The multipliers εⱼᵢ again form a Hermitian matrix ε (same reason — real energy), and the two conjugate equations become one. Collect all the orbitals as columns of a matrix C and all the multipliers into ε, and the whole set of stationarity conditions is a single matrix equation:

F C = S C ε.

There it is. The intractable three-dimensional nonlinear integro-differential eigenproblem has become an algebraic matrix equation in fixed-size matrices F, S, C, ε. Everything in it is an integral over the χ's (computable once) or an unknown number to solve for.

Now diagonalize ε. As before I'm free to mix the occupied orbitals by a unitary transformation without changing the determinant (hence without changing the energy), and that freedom lets me choose ε diagonal with real diagonal entries εᵢ. Then F C = S C ε reduces, column by column, to

F c_i = εᵢ S c_i,

or, dropping the index, the single relation

F c = ε S c, i.e. (F − ε S) c = 0.

This is the generalized eigenvalue problem for the pair of matrices (F, S). It's the ordinary Hermitian eigenvalue problem F c = ε c with the unit matrix replaced by S — and it collapses to the ordinary one exactly when S = I, i.e. when the basis happens to be orthonormal. For atom-centered functions S ≠ I, so the overlap genuinely belongs there; pretending S = I would be silently imposing the wrong inner product and would give wrong orbitals and a wrong energy. A nontrivial c exists only when the coefficient matrix is singular, so the orbital energies are the roots of the **secular equation**

Det(F − ε S) = 0.

With m basis functions this is an m-th degree polynomial in ε, and (because S is positive definite — overlaps of linearly independent functions) it has m real roots. The lowest n of them I fill (their eigenvectors are the n occupied molecular orbitals, doubly occupied — the aufbau filling), and the remaining m − n are the unoccupied "virtual" orbitals. The whole machinery of Hermitian eigenproblems — real eigenvalues, orthogonality of eigenvectors (now in the S-metric, c_i† S c_j = δᵢⱼ), degeneracy handling — carries over to (F − εS)c = 0 with only the obvious modifications.

One more thing nags. I wrote F as if it were a known matrix, but it isn't: F = H + Σⱼ (2Jⱼ − Kⱼ), and the Coulomb and exchange matrices Jⱼ, Kⱼ are built from the *occupied orbitals* — that's the averaged electron–electron field, and the orbitals are exactly what I'm solving for. So F depends on C. This is the same nonlinearity Hartree faced: the operator depends on its own solution. So I can't solve it in closed form; I have to iterate. Guess a set of orbitals (a set of c's), build F from them, solve Det(F − εS) = 0 for the n lowest c's, build a new F from *those*, solve again, and repeat until the orbitals I put in and the orbitals I get out agree. That fixed-point loop is precisely Hartree's self-consistency idea — only now each step is a matrix diagonalization, not a numerical integration of a differential equation. This is the self-consistent-field procedure; with the orbitals expanded in atomic orbitals, the linear-combination-of-atomic-orbitals self-consistent field.

Let me make the iteration concrete, because the way F depends on C wants to be written explicitly in terms of the integrals, not left as abstract operators J_j, K_j. The averaged field only cares about the *occupied* orbitals through a single object: the sum over occupied orbitals of their coefficient outer products. Define the density matrix (closed shell, double occupancy)

D_{κλ} = 2 Σ_{i ∈ occ} C_{κi} C_{λi}.

The factor 2 is the two electrons per spatial orbital. Then the averaged Coulomb field of all the electrons, and the exchange, are linear in D and assembled from the two-electron repulsion integrals (pq|rs) = ∫∫ χ_p(1)χ_q(1) (1/r₁₂) χ_r(2)χ_s(2) over the basis. Working out Σⱼ(2Jⱼ − Kⱼ) in terms of D and the (pq|rs): the Coulomb part pairs the density with the (μν|κλ) integral, the exchange part pairs it with the (μκ|νλ) integral and carries the determinant-mandated factor ½ relative to Coulomb (the same 2-vs-1 that gave 2J − K). So the Fock matrix is

F_{μν} = H^core_{μν} + Σ_{κλ} D_{κλ} [ (μν|κλ) − ½ (μκ|νλ) ].

The first bracket term is the classical Coulomb repulsion in the field of the whole electron density; the second, with its minus sign and factor ½, is exchange — the fingerprint of antisymmetry, the thing that distinguishes this from a plain Hartree mean field. And the total energy, in matrix form, is

E = ½ Σ_{μν} D_{μν} ( H^core_{μν} + F_{μν} ) + E_nuc,

the ½ avoiding double-counting the electron–electron interaction (it sits in both H^core's absence and F's presence), plus the classical nucleus–nucleus repulsion E_nuc which doesn't involve the electrons.

Now, solving the *generalized* problem F c = ε S c at every iteration is slightly awkward; I'd rather solve an ordinary eigenproblem. The clean way is to change to an orthonormal basis once, at the start, since S is fixed. I want a matrix X with Xᵀ S X = I; then in the X-basis the overlap is the identity. The natural, least-distorting choice is the symmetric (Löwdin) one: diagonalize S = U s Uᵀ (s the diagonal of overlap eigenvalues, all positive), and take

X = S^{-1/2} = U s^{-1/2} Uᵀ.

Why this particular square root and not, say, a Cholesky factor? Among all matrices that orthonormalize the basis, S^{-1/2} produces the orthonormal set that is *closest* to the original atomic orbitals in the least-squares sense — it mixes them as little as possible, so the coefficients stay interpretable as "mostly this atomic orbital." A Cholesky or canonical orthogonalization would orthonormalize too, but by a lopsided, basis-order-dependent mixing that throws away that interpretability; the symmetric root is the democratic, minimal-distortion choice. With X in hand, transform the Fock matrix into the orthonormal basis, F′ = Xᵀ F X = S^{-1/2} F S^{-1/2}, solve the *ordinary* symmetric eigenproblem F′ C′ = C′ ε, and back-transform the orbitals C = X C′ = S^{-1/2} C′. (Equivalently I can hand F and S to a generalized symmetric eigensolver directly — it does the same thing internally.)

So the self-consistent loop, concretely: form S, H^core, the two-electron integrals (pq|rs), and S^{-1/2} once. Start from a guess density (even D = 0 works — it makes the first F just H^core, i.e. the bare one-electron problem, whose occupied orbitals are a sensible starting point). Then repeat: build F from the current D; transform F′ = S^{-1/2} F S^{-1/2}; diagonalize to get C′ and ε; back-transform C = S^{-1/2} C′; take the n lowest orbitals and build a new D = 2 Σ_occ CCᵀ; compute the energy. Stop when the energy and the density stop changing between iterations (both below small thresholds). The converged D and E are the closed-shell self-consistent-field solution for the molecule.

There's a real efficiency lever I should note, even if I don't lean on it. If the molecule has point-group symmetry and I arrange the Fock operator to carry that symmetry, the molecular orbitals must transform as irreducible representations of the group, and orbitals of different symmetry have zero Fock and zero overlap matrix elements between them. So F and S become block-diagonal in a symmetry-adapted basis, one block per irreducible representation, and the single big secular equation Det(F − εS) = 0 factors into several small ones, Det(F^Γ − εS^Γ) = 0, one per symmetry species. That cuts the diagonalization cost dramatically and labels each orbital by its symmetry. It's an acceleration and a bookkeeping aid, not a change to the method — the answer is the same, found faster.

Let me trace the whole chain back to convince myself it's tight. The exact closed-shell Hartree–Fock condition is a one-electron eigenproblem F φ = ε φ, but for a molecule φ is an arbitrary three-dimensional function and the equation is a nonlinear integro-differential eigenproblem with no separation of variables — unsolvable in practice. Confine φ to the span of a fixed, finite set of atom-centered functions, φᵢ = Σ_p c_{pi} χ_p; the variational principle keeps the energy an upper bound and turns the unknown into a finite list of numbers. Because atom-centered functions on different nuclei overlap, carry the overlap matrix S explicitly rather than orthogonalizing the chemistry away. Substituting the expansion and minimizing the energy under the orbital-orthonormality constraint c_i† S c_j = δᵢⱼ (Lagrange multipliers, Hermitian multiplier matrix diagonalized by the determinant's unitary-mixing freedom) collapses every function integral into a fixed matrix and lands on the algebraic generalized eigenvalue problem F C = S C ε, i.e. F c = ε S c with secular equation Det(F − εS) = 0. Since F is built from the occupied orbitals (through the density matrix D and the two-electron integrals, F = H^core + Σ D[(μν|κλ) − ½(μκ|νλ)]), it depends on its own solution, so iterate it to self-consistency exactly as Hartree did — but now each step is a matrix diagonalization. That is the linear-combination-of-atomic-orbitals self-consistent field: the integro-differential Hartree–Fock problem rewritten as a finite matrix eigenproblem a computer can actually solve.

```python
import numpy as np
from scipy.linalg import fractional_matrix_power

def density_matrix(C, nocc):
    # D_{kl} = 2 * sum_{i in occupied} C_{ki} C_{li}   (double occupancy, closed shell)
    Cocc = C[:, :nocc]
    return 2.0 * Cocc @ Cocc.T

def fock_matrix(Hcore, eri, D):
    # F_{uv} = Hcore_{uv} + sum_{kl} D_{kl} [ (uv|kl) - 1/2 (uk|vl) ]
    # Coulomb: (uv|kl) contracted with D ; exchange: (uk|vl) contracted with D, with -1/2
    J = np.einsum("uvkl,kl->uv", eri, D)     # classical Coulomb field of the whole density
    K = np.einsum("ukvl,kl->uv", eri, D)     # exchange: the antisymmetry term
    return Hcore + J - 0.5 * K

def solve_generalized(F, X):
    # X = S^{-1/2}: move to the orthonormal (Loewdin) basis so it's an ordinary eigenproblem
    Fp = X.T @ F @ X                         # F' = S^{-1/2} F S^{-1/2}
    eps, Cp = np.linalg.eigh(Fp)             # ordinary symmetric eigenproblem F' C' = C' eps
    C = X @ Cp                               # back-transform: C = S^{-1/2} C'
    return eps, C

def scf(S, Hcore, eri, e_nuclear, n_electrons, max_iter=64, e_tol=1e-10, d_tol=1e-8):
    nbf  = S.shape[0]
    nocc = n_electrons // 2                  # doubly-occupied spatial orbitals
    X    = fractional_matrix_power(S, -0.5)   # S^{-1/2}, the minimal-distortion orthonormalizer

    D = np.zeros((nbf, nbf))                  # guess: zero density -> first Fock is bare Hcore
    E_old = 0.0
    for it in range(max_iter):
        F = fock_matrix(Hcore, eri, D)        # average field built from current occupied orbitals
        E = 0.5 * np.sum(D * (Hcore + F)) + e_nuclear   # total energy (1/2 avoids double counting)
        eps, C = solve_generalized(F, X)      # solve F c = eps S c  via the orthonormal basis
        D_new = density_matrix(C, nocc)       # rebuild density from the n lowest orbitals (aufbau)

        dE   = abs(E - E_old)
        rmsD = np.sqrt(np.mean((D_new - D) ** 2))
        print(f"iter {it:2d}  E = {E:.10f}  dE = {dE:.2e}  rmsD = {rmsD:.2e}")
        if dE < e_tol and rmsD < d_tol:       # self-consistent: input and output orbitals agree
            return E, C, eps, D_new
        D, E_old = D_new, E
    raise RuntimeError("SCF did not converge")
```
