# Context: Fast randomized numerical linear algebra for sparse, overconstrained matrices

## Research question

Many core numerical-linear-algebra tasks on a tall matrix A ∈ R^{n×d} with n ≫ d (rank r ≤ d)
have the same shape: there is an exact O(nd²) or O(n³) algorithm, and a much faster randomized
*approximate* algorithm that first compresses A to a small surrogate and then works on the surrogate.
The four tasks of interest:

- **Overconstrained least-squares regression.** Given A and b ∈ R^n, output x' with
  ‖Ax'−b‖₂ ≤ (1+ε)·min_x ‖Ax−b‖₂. The exact minimizer is x* = A⁻b (= (AᵀA)⁻¹Aᵀb at full rank),
  computable in O(nd²).
- **Low-rank approximation.** Given A ∈ R^{n×n} and k, output a rank-k matrix A' with
  ‖A'−A‖_F ≤ (1+ε)‖A−A_k‖_F, where A_k is the best rank-k approximation. Exact via SVD in O(n³).
- **Leverage-score approximation.** For A = UΣVᵀ, output, for every row i simultaneously, a
  constant-factor approximation to the i-th leverage score u_i = ‖U_{i,*}‖₂² (the diagonal of the
  projector onto C(A); basis-independent; measures correlation of the column space with e_i).
- **ℓ_p regression**, 1 ≤ p < ∞: output x' with ‖Ax'−b‖_p ≤ (1+ε)·min_x ‖Ax−b‖_p.

The precise goal that motivates everything below: the running times of the existing randomized
algorithms all have a *leading term that reads the entire matrix* — Θ(nd) or Θ(nd log n) — even when
A is very sparse. Write nnz(A) for the number of nonzero entries of A. For many real matrices
(document-term, graph adjacency, recommender, web) nnz(A) ≪ nd, often nnz(A) = O(n). The question
is whether the leading term can be brought down to O(nnz(A)) — pay only for the nonzeros, plus a
lower-order additive overhead that is polynomial in d, k, 1/ε and *not* polynomial in n. For
regression the dream is O(nnz(A)) + poly(d/ε); for low-rank, O(nnz(A)) + n·poly(k/ε).

## Background

**Sketch-and-solve.** The unifying tool is a random linear map S : R^n → R^t with t ≪ n. One
replaces A by the sketch SA (and b by Sb), solves the small problem on the sketch, and lifts the
solution back. For this to give a (1+ε)-approximation for regression and low-rank, S must
*approximately preserve norms inside the relevant subspace*: a t×n matrix S is an ε-subspace
embedding for A if, simultaneously for all x ∈ R^d,
  ‖SAx‖₂ = (1 ± ε)‖Ax‖₂.
Equivalently, with U an orthonormal basis for the column space C(A), all singular values of SU lie
in [1−ε, 1+ε]. "Oblivious" means S is drawn from a fixed distribution that works for any given A with
high probability, without looking at A. Given any ε-subspace embedding as a black box, all four tasks
above reduce to small problems on the sketch — so the running time is dominated by the cost of
*forming the sketch* SA.

**The two costs of a sketch.** A sketching distribution is judged on (i) the target dimension t
needed for the embedding guarantee, and (ii) the time to compute the matrix product S·A. Smaller t
makes the downstream small problem cheaper; faster S·A makes the leading term cheaper. The standard
constructions trade these off, but every one of them reads all of A.

**Leverage scores as the geometry of a column space.** For A = UΣVᵀ, u_i = ‖U_{i,*}‖₂² are the
leverage scores; Σ_i u_i = r. They are exactly the importances that make row-sampling work for
regression and they are a fixed property of the *subspace* C(A) (independent of the chosen basis).
A standard diagnostic fact about them: a unit vector y in the column space satisfies, coordinatewise,
y_i² ≤ u_i (since y = Ux for a unit x and Cauchy-Schwarz gives y_i² = (U_{i,*}·x)² ≤ ‖U_{i,*}‖² ).
So the leverage scores cap how large any single coordinate of any unit column-space vector can be,
and because Σ_i u_i = r, only a few coordinates can be large at once.

**Concentration tools available at the time.** (a) The Johnson–Lindenstrauss lemma: a dense random
sign/Gaussian S with t = O(ε⁻² log(1/δ)) rows preserves the norm of a *fixed* vector to (1±ε) with
probability 1−δ. The standard way to upgrade "fixed vector" to "whole subspace" is a *net argument*:
place an ε-net on the unit sphere of C(A) (it has e^{O(r)} points), preserve each net point by a
union bound, then extend to all vectors by linearity — this requires the per-point failure
probability to be e^{−Ω(r)}. (b) Khintchine's inequality, for the moments of a signed sum Σ_i σ_i a_i
with σ_i ∈ {±1}. (c) The Hanson–Wright inequality (1971): for z ∈ R^n of i.i.d. ±1 entries and
symmetric B, E|zᵀBz − tr(B)|^ℓ ≤ (CQ)^ℓ with Q = max{√ℓ ‖B‖_F, ℓ‖B‖₂} — a tail bound for quadratic
forms. (d) Bernstein/Azuma and matrix-Bernstein inequalities for sums of bounded (matrix-valued)
random variables. (e) The random ±1-hash trick of Alon–Matias–Szegedy: hashing items to counters
with random signs makes a counter an unbiased estimator of a sum of squares, because cross terms
have mean zero.

**Inspiration from streaming.** In data-stream norm estimation, an effective strategy is to maintain
*separate data structures for the heavy and the light components* of a high-dimensional vector: a few
coordinates carry most of the mass and must be tracked accurately, while the many small coordinates
can be handled by cheap randomized aggregation (Nelson–Woodruff 2010; Kane–Nelson–Porat–Woodruff
2011). The notion of "which coordinates are heavy" is the load-bearing modeling choice in those
results.

## Baselines

**Fast Johnson–Lindenstrauss / dense sign sketches (Sarlós, FOCS 2006).** Take S to be a
Fast-JL transform; with t = O(d/ε²) it is a subspace embedding, and S·A can be applied in
O(nd log t) time when d < n^{1/2−γ}. This established the whole sketch-and-solve framework for
regression, low-rank approximation, and approximate matrix multiplication, and gave the first
relative-error guarantees. *Limitation:* computing S·A costs Θ(nd log n) — it processes the full
dense array. For an A with nnz(A) = O(n) nonzeros, this pays a factor n more than the number of
nonzeros; a dense Gaussian S is even worse, at Θ(ndt).

**Subsampled randomized Hadamard transform / leverage-score sampling
(Drineas–Mahoney–Muthukrishnan; Drineas–Mahoney–Muthukrishnan–Sarlós, Numer. Math. 2011).**
Precondition A with a randomized Hadamard transform H_nD_n (which approximately equalizes the
leverage scores), then uniformly sample O(d log d/ε²) rows and solve the induced problem; or sample
rows directly with probabilities ∝ u_i. Running time O(nd log d) for least squares; this line
introduced leverage scores as the correct sampling importances. *Limitation:* the Hadamard
preconditioning multiplies the full n×d matrix (Θ(nd log n)); and computing leverage scores exactly
needs an SVD/QR (O(nd²)). Approximate leverage scores were known in O(nd log n + d³ log d log n)
(Drineas–Magdon-Ismail–Mahoney–Woodruff, ICML 2012) — again with an nd leading term.

**Sparse Johnson–Lindenstrauss transforms (Dasgupta–Kumar–Sarlós, STOC 2010;
Kane–Nelson, SODA 2012).** Random hashing matrices with few nonzeros per column that preserve the
norm of a *fixed* vector; the analysis crucially needs the input vector to have small ∞-norm
(no single coordinate carrying too much mass), and KN12 give a tight version via Hanson–Wright.
*Limitation:* these are JL statements for a fixed vector or an arbitrary finite set of vectors —
they say nothing, by themselves, about preserving every vector of an entire d-dimensional subspace,
where the standard net argument needs e^{−Ω(d)} per-point failure that these sparse maps do not
provide for worst-case vectors.

**Streaming CountSketch data structure (Charikar–Chen–Farach-Colton, ICALP 2002 / TCS 2004).**
For frequency estimation: t hash tables of b counters each, with h_j mapping items to [b] and a
random sign s_j; an item q updates C[j, h_j(q)] += s_j(q), and its frequency is estimated as
median_j s_j(q)·C[j, h_j(q)], which is unbiased because of the random signs. *Limitation:* this is a
per-coordinate frequency estimator analyzed in the streaming model; it was never analyzed as a
norm-preserving map on a linear subspace, and the proof technique (per-item median over independent
tables) does not give a simultaneous all-vectors guarantee.

**Iterative / Krylov methods (CG, Lanczos; e.g. Trefethen–Bau; Zouzias–Freris 2012).** Repeatedly
form matrix-vector products Ax (each Θ(nnz(A))). *Limitation:* the number of iterations N depends on
the condition number and spectral properties of A and can be large (surveys report N = Θ(k) for
Krylov methods to get k leading singular vectors), with no a-priori condition-independent bound.

The common gap across all baselines: the leading term reads all nd entries (or depends on the
conditioning of A), even though the information content of a sparse A is only nnz(A) numbers.

## Evaluation settings

The natural yardstick is the leading-order running time as a function of nnz(A), n, d (or rank r),
k, and 1/ε, with constants independent of the numerical properties of A — and the approximation
guarantees: (1+ε)-relative error for regression and low-rank Frobenius error, constant-factor (or
(1±ε)) relative error for all leverage scores simultaneously, (1+ε)-relative error for ℓ_p
regression. The standard test corpus for sparse-matrix numerical methods is the University of Florida
Sparse Matrix Collection (matrices from many application areas, a wide range of nnz/n, sizes up to
thousands of rows); the natural quality metric for low-rank approximation is the ratio of the
achieved Frobenius error to the best rank-k Frobenius error, as a function of the sketch size
relative to k.

## Code framework

The primitives that already exist: dense/sparse linear algebra, a small dense least-squares solver,
and the sketch-and-solve template. The piece that does not exist yet is the sketching map itself —
the distribution over linear maps S and the procedure that applies it to A. The final method fills
exactly the stub below.

```python
import numpy as np
from scipy import sparse
from scipy import linalg


def build_sketch(t, n, rng):
    """Return a t-by-n linear map S (the random sketching matrix), to be applied as S @ A.

    The whole method lives here: which distribution over linear maps R^n -> R^t makes
    S A preserve the geometry of the column space of A, while being cheap to apply.
    """
    # TODO: construct the sketching map.
    pass


def sketch_apply(S, A):
    """Form the sketch S A."""
    return S @ A


def embedding_dimension(r, eps):
    """Sketch size t needed for an eps-subspace-embedding guarantee on a rank-r column space."""
    # TODO: the dimension implied by the analysis of build_sketch.
    pass


def regression_sketch_and_solve(A, b, t, rng):
    """(1+eps)-approximate argmin_x ||A x - b||_2 via sketch-and-solve."""
    Ab = _adjoin(A, b)                 # work in the column space of [A | b]
    S = build_sketch(t, Ab.shape[0], rng)
    SAb = sketch_apply(S, Ab)
    SA, Sb = SAb[:, :-1], SAb[:, -1]
    x = linalg.lstsq(_dense(SA), _dense(Sb))[0]   # small dense least squares on the sketch
    return x


def lowrank_sketch(A, k, t, rng):
    """(1+eps)-approximate best rank-k approximation of A via sketching."""
    # TODO: a sketch-based reduction to a small problem whose solution lifts to rank k.
    pass


def leverage_scores(A, t, rng):
    """Constant-factor approximation to all leverage scores u_i = ||U_{i,*}||^2 simultaneously."""
    # TODO: a sketch-based estimate of the row norms of an orthonormal basis of C(A).
    pass


# --- helpers that already exist (sparsity-aware plumbing) ---
def _adjoin(A, b):
    b = np.asarray(b).reshape(-1, 1)
    if sparse.issparse(A):
        return sparse.hstack([A, sparse.csc_matrix(b)]).tocsc()
    return np.hstack([A, b])


def _dense(M):
    return M.toarray() if sparse.issparse(M) else np.asarray(M)
```
