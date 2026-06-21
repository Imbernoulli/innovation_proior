# Low Rank Approximation and Regression in Input-Sparsity Time

## Problem

Take a tall matrix A ∈ R^{n×d} with n ≫ d and rank r. The randomized
sketch-and-solve paradigm reduces least squares, low-rank approximation, leverage-score
approximation, and ℓ_p regression to small problems on a sketch SA, provided S is a *subspace
embedding* for the column space C(A): ‖SAx‖₂ = (1±ε)‖Ax‖₂ for all x. The cost is dominated by
forming SA. Every prior sketch (Fast-JL, subsampled randomized Hadamard, dense sign/Gaussian) reads
all nd entries — Θ(nd log n) or worse — even when A has only nnz(A) ≪ nd nonzeros. The goal: a
subspace embedding applied in O(nnz(A)) time.

## Key idea

Use the **sparse embedding matrix** S = ΦD (the CountSketch matrix): a hash h:[n]→[t] and signs
D_{ii}∈{±1}, with S having exactly one nonzero per column — value D_{ii} in row h(i). Then SA is
computed by hashing each nonzero of A into a bucket with its sign and summing: **O(nnz(A)) time**.

Such an S cannot preserve the norms of an arbitrary set of e^{O(r)} vectors (two heavy coordinates
colliding in a bucket cause full distortion), so the usual net argument fails. The resolution: the
e^{O(r)} unit vectors of C(A) are not arbitrary — for any unit y = Ux, y_i² = (U_{i,*}·x)² ≤ ‖U_{i,*}‖²
= u_i, the i-th leverage score. So the coordinates that can ever be large lie in a *fixed* set H of
size ≤ r/α (the large leverage scores, since Σ u_i = r). Perfectly hash H (exact isometry on the
heavy part), control the light part (small ∞-norm) by Hanson–Wright, bound the cross term by
Cauchy–Schwarz/Khintchine, and close with a net over the r-dimensional subspace.

## Construction

For sketch dimension t, draw h:[n]→[t] uniform, Φ∈{0,1}^{t×n} with Φ_{h(i),i}=1 (else 0), and
diagonal D with D_{ii}∈{±1} uniform i.i.d. The **sparse embedding matrix** is S = ΦD.
E[‖ΦDy‖²] = ‖y‖² (random signs kill the cross terms).

## Main theorem (subspace embedding in input-sparsity time)

**Theorem.** There is t = O((r/ε)⁴ log²(r/ε)) such that, with probability ≥ 9/10, S = ΦD is an
ε-subspace embedding for A: ‖ΦDy‖₂ = (1±ε)‖y‖₂ for all y ∈ C(A). S can be applied in O(nnz(A))
time. (A refined analysis reduces the dimension to t = r²ε⁻² · polylog(r/ε).)

**Proof.** Order rows so u_i are non-increasing; fix threshold T, let s = min{i : u_i ≤ T}; split a
unit y ∈ C(A) into heavy y^H = y_{1:(s−1)} (u_i > T) and light y^L = y_{s:n} (u_i ≤ T). Recall
y_i² ≤ u_i for all i.

*Heavy (perfect hashing).* Let E_B = {h injective on {1,…,s−1}}. Pr[¬E_B] ≤ s²/t (birthday).
On E_B, distinct heavy coordinates hit distinct buckets, so ‖ΦD y^H‖² = Σ_{i<s} y_i² = ‖y^H‖²
exactly.

*Bucket loads (event E_h).* For bucket j let X_i = u_i·𝟙[h(i)=j, i≥s] ∈ [0,T]; E[ΣX_i] ≤ r/t,
V = ‖u_{s:n}‖²/t. Bernstein (X_i∈[0,T], V ≤ LT²/6 ⇒ Pr[ΣX_i ≥ ΣE[X_i]+LT] ≤ e^{−L}) with
L=log(t/δ_h) gives, when t ≥ 6‖u_{s:n}‖²/(L T²), failure δ_h/t per bucket; union bound ⇒ with prob
≥ 1−δ_h every bucket has light mass ≤ W ≡ T log(t/δ_h) + r/t.

*Light (Hanson–Wright).* ‖ΦDy^L‖² = zᵀBz, z=diag(D)∈{±1}^n, B_{ii'}=y_i y_{i'}𝟙[h(i)=h(i')] (i,i'≥s),
tr(B)=‖y^L‖². On E_h: ‖B‖_F² = Σ_{i≥s} y_i² Σ_{i':h(i')=h(i)} y_{i'}² ≤ W, and ‖B‖₂ = max_j
(bucket mass) ≤ W (the per-bucket vectors are the eigenvectors). With ℓ ≤ 1/W,
Q = max{√ℓ‖B‖_F, ℓ‖B‖₂} ≤ √(ℓW). Hanson–Wright E|zᵀBz−tr B|^ℓ ≤ (CQ)^ℓ + Markov with ℓ=log(1/δ_L):
|‖ΦDy^L‖² − ‖y^L‖²| ≤ K_L √(W log(1/δ_L)) with failure δ_L.

*Cross term (Khintchine).* On E_B each heavy bucket holds ≤1 heavy coordinate; the cross term is
Σ_{i≥s} y_i D_{ii} z_i with z_i the colliding heavy value. Khintchine:
E[(·)^{2p}]^{1/p} ≤ C_p Σ_{i≥s} y_i² z_i² = C_p Σ_{i'<s} y_{i'}² Σ_{i:h(i)=i'} y_i² ≤ C_p W, so
|cross| ≤ K_C √(W log(1/δ_C)) with failure δ_C.

*Fixed y.* On E_h, E_B: |‖ΦDy‖² − ‖y‖²| ≤ K_L√(W log(1/δ_L)) + 2K_C√(W log(1/δ_C)). With
δ_L=δ_C=δ_y/2 and W ≤ K_y ε²/log(1/δ_y) (K_y ≤ 1/(9(K_L+K_C)²)), this is ≤ ε. Taking ℓ=Θ(r) makes
δ_y = e^{−Ω(r)}.

*Net over the subspace.* Let E = {w ∈ (γ/√r)Z^r : ‖w‖≤1}, γ = 1−1/√2; |E| ≤ e^{cr}, c=1/γ+2
(Arora–Hazan–Kale). For J = UᵀSᵀSU − I_r, applying the fixed-y bound to Ux, Uy, U(x+y) for x,y∈E
gives |xᵀJy| ≤ ε/2; union bound (failure δ_y K_{sub}^r ≤ 1/10) over E, then
"|uᵀJv|≤ε ∀u,v∈E ⇒ |wᵀJw| ≤ ε/(1−γ)² ∀ unit w" gives ‖Sw‖²=(1±ε)‖w‖² for all w ∈ C(A).

*Dimension.* Need t ≥ s², and s ≤ r/T with T = Θ(ε²/(r log)) ⇒ s = Θ((r/ε)²) ⇒ t = O((r/ε)⁴log²).
(Refined: group leverage scores by powers of two G_j={u_i∈(2^{−j},2^{−j+1}]}; perfect-hash only the
unit scores, bound the spectral norm ‖Û_j‖₂² of each collided submatrix via matrix Bernstein, giving
t = r²ε⁻² polylog.) ∎

## Corollaries (input-sparsity-time algorithms)

- **Least squares.** Embed C([A|b]); solve argmin_x ‖ΦDAx − ΦDb‖. Time O(nnz(A)) + Õ(d³ε⁻²).
  A preconditioned-iteration variant (residual contracts by 3ε₀ < 1 per step for a small
  enough constant ε₀, since AR(x^{(m+1)}−x*) = U(Σ−Σ³)Vᵀ(x^{(m)}−x*) and the diagonal of Σ−Σ³ is
  ≤ σ_i((1+ε₀)²−1) ≤ 3ε₀σ_i) gives
  O(nnz(A) log(n/ε) + r³ log²r + r² log(1/ε)) — log dependence on ε.
- **Leverage scores.** Π₁ = (Fast-JL)∘(sparse embedding), QR for R, read row norms of A(RΠ₂):
  all u_i to (1±ε) in O(nnz(A) log n + r³ log²r + r² log n).
- **Low rank.** Right-sketch AR⊤ (rank-k space within (1+ε)Δ_k by generalized regression), then a
  left affine embedding S=(SRHT)∘(sparse) with the rank-k restriction solved by [Ũᵀ SA]_k:
  O(nnz(A)) + Õ(nk²ε⁻⁴ + k³ε⁻⁵), giving orthonormal L,W and diagonal D with
  ‖A − LDW⊤‖_F ≤ (1+ε)Δ_k.
- **ℓ_p regression**, 1 ≤ p < ∞. Block the rows, embed each block to ℓ₂ with the high-probability
  generalized embedding (a JL inside each bucket), build a well-conditioned ℓ_p basis, sample rows ∝
  ℓ_p norms: O(nnz(A) log n) + poly(r/ε).

Two refinements support these. **Partition refinement:** geometric leverage groups + matrix-Bernstein
bound on collided submatrices ⇒ t = r²ε⁻² polylog. **Generalized embedding:** hash into ~d²/log n
buckets and run an O(log n/ε²) JL per bucket ⇒ success 1−1/poly(n) (needed for the block union bound
in ℓ_p / low rank), time O(nnz(A) log n / ε).

## Code (faithful to the canonical sparse-embedding implementation)

```python
import numpy as np
from scipy import sparse
from scipy import linalg


def cwt_matrix(t, n, rng):
    """Sparse embedding S = Phi D in R^{t x n}: exactly one +-1 per column.

    Column i has its single nonzero in row h(i) (the hash) with value the random sign
    D_ii. As a CSC matrix the column pointers are 0,1,...,n, so each column holds one
    entry. Applying S to A then hashes every nonzero of A into a bucket with a sign and
    adds -- O(nnz(A)) time.
    """
    rows = rng.integers(0, t, size=n)          # h : [n] -> [t]
    cols = np.arange(n + 1)                     # one nonzero per column
    signs = rng.choice([1, -1], size=n)        # diag(D)
    return sparse.csc_matrix((signs, rows, cols), shape=(t, n))


def sparse_embedding(A, t, rng):
    """Form S A in time proportional to nnz(A)."""
    S = cwt_matrix(t, A.shape[0], rng)
    return S @ A


def least_squares(A, b, t, rng):
    """(1+eps)-approximate argmin_x ||A x - b||_2 via sketch-and-solve on [A | b]."""
    b = np.asarray(b).reshape(-1, 1)
    if sparse.issparse(A):
        Ab = sparse.hstack([A, sparse.csc_matrix(b)]).tocsc()
    else:
        Ab = np.hstack([np.asarray(A), b])
    SAb = sparse_embedding(Ab, t, rng)                       # O(nnz(A)) time
    SAb = SAb.toarray() if sparse.issparse(SAb) else SAb
    SA, Sb = SAb[:, :-1], SAb[:, -1]
    x, *_ = linalg.lstsq(SA, Sb)                             # small dense least squares
    return x


def least_squares_iterative(A, b, t, rng, n_iter=40):
    """log(1/eps) variant: sketch -> well-conditioned R -> iterate the fixed point
    x <- x + R^T A^T (b - A R x), which contracts the residual by ~3*eps0 < 1 / step."""
    A = (A.toarray() if sparse.issparse(A) else np.asarray(A)).astype(float)
    b = np.asarray(b).ravel()
    SA = sparse_embedding(A, t, rng)
    SA = SA.toarray() if sparse.issparse(SA) else SA
    _, Rmat = np.linalg.qr(SA)
    Rinv = np.linalg.inv(Rmat)                               # A @ Rinv is well-conditioned
    AR = A @ Rinv
    y = np.zeros(AR.shape[1])
    for _ in range(n_iter):
        y = y + AR.T @ (b - AR @ y)
    return Rinv @ y


def leverage_scores(A, t, rng):
    """Constant-factor approx to all u_i = ||U_{i,*}||^2: embed, basis-change R so that
    (A R) is ~orthonormal, then ||(A R)_{i,*}||^2 = (1 +- eps) u_i; read row norms with a
    tiny JL Pi2 (width O(log n))."""
    A = (A.toarray() if sparse.issparse(A) else np.asarray(A)).astype(float)
    n = A.shape[0]
    SA = sparse_embedding(A, t, rng)
    SA = SA.toarray() if sparse.issparse(SA) else SA
    _, Rmat = np.linalg.qr(SA)
    R = np.linalg.inv(Rmat)
    p = max(1, int(np.ceil(8 * np.log(max(n, 2)))))
    Pi2 = rng.standard_normal((R.shape[1], p)) / np.sqrt(p)
    Y = A @ (R @ Pi2)
    return np.sum(Y ** 2, axis=1)
```

The embedding dimension t is chosen from the analysis above: t = O((r/ε)⁴ log²(r/ε)) for the basic
construction, or t = r²ε⁻² polylog(r/ε) for the partition-refined one. (These are worst-case
guarantees; in practice far smaller t already yields error ratios near 1.)
