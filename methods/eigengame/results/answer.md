# EigenGame: PCA as a Nash Equilibrium

## Problem

Recover the **top-k principal components** of `X ∈ R^{n×d}` — the eigenvectors of
`M = XᵀX`, in order, each a true component (not merely a basis of the top-k subspace) — when
`n` and `d` are so large that a full SVD (`O(min{nd², n²d})` time, `O(nd)` space) and even
forming `M` (`d²` entries) are infeasible. The deployment target is a farm of accelerators,
so the procedure must decentralize: one component per device, minimal communication, and no
centralized re-orthonormalization (`QR`/Stiefel/deflation) that synchronizes all `k` vectors
every step.

## Key idea

Treat each estimated component `v̂_i` as a **player** in a `k`-player game. Player `i`
maximizes its own utility — variance captured along `v̂_i` (a reward) minus its alignment with
its **parents** `v̂_{j<i}` (a penalty) — over the unit sphere:

  `u_i(v̂_i | v̂_{j<i}) = v̂_iᵀM v̂_i − Σ_{j<i} (v̂_iᵀM v̂_j)² / (v̂_jᵀM v̂_j)`
                       `= ‖Xv̂_i‖² − Σ_{j<i} ⟨Xv̂_i, Xv̂_j⟩² / ⟨Xv̂_j, Xv̂_j⟩`,  with `‖v̂_i‖ = 1`.

The reward is the Rayleigh quotient; the penalty is the squared **generalized** inner product
`⟨·,·⟩_M` with each parent, **normalized by the parent's own Rayleigh quotient**. The utilities
are **asymmetric** — a child penalizes its parents but not vice versa — which is exactly what
makes this a game rather than a single optimization, and what respects PCA's natural ordering
(`v̂_1` chases maximal variance freely; deeper players also stay clear of those above them).

Why each choice:
- **Penalize off-diagonals of `R = V̂ᵀMV̂`, not just maximize the trace.** `Tr(V̂ᵀMV̂) = Tr(M)`
  is independent of `V̂` at `k=d`, so trace-max recovers only the subspace. The eigenvalue
  equation `VᵀMV = Λ` says the components are the `V̂` that makes `R` *diagonal*.
- **Generalized (M-weighted) penalty.** A plain Euclidean overlap `Σ⟨v̂_i,v̂_j⟩²` is drowned out
  by large eigenvalues; including `M` balances penalty against reward.
- **Normalize by the parent's Rayleigh quotient.** Puts the two terms on a common scale,
  improves near-optimum convergence (≈ minimizing err² rather than err⁴), and — crucially —
  makes the gradient collapse to a clean form.
- **Hierarchy (DAG), not symmetry.** The hierarchical game has a *unique* strict Nash and a
  clean inductive proof; the fully symmetric variant has no symmetric Nash and its uniqueness
  is unprovable (NP-hard). It also enables a sequential, globally convergent solver.

## Main results

**PCA = unique strict Nash.** Assuming the top-k eigenvalues are positive and distinct, the
top-k eigenvectors form the **unique strict Nash equilibrium** (up to sign). *Proof sketch:*
diagonalize `M = UΛUᵀ`; for `v̂_i = Σ_p w_p v_p` with `‖w‖=1`, set `z_p = w_p² ∈ Δ^{d-1}`. With
exact parents, `u_i = Σ_{p≥i} Λ_pp z_p` — a linear program over the simplex whose unique
maximizer is the vertex `e_i`, i.e. `v̂_i = ±v_i`. Induction over `i` finishes it.

**Gradient = Oja + generalized Gram–Schmidt.**

  `∇_{v̂_i} u_i = 2M[ v̂_i − Σ_{j<i} (v̂_iᵀM v̂_j)/(v̂_jᵀM v̂_j) v̂_j ]`
              `= 2Xᵀ[ Xv̂_i − Σ_{j<i} ⟨Xv̂_i,Xv̂_j⟩/⟨Xv̂_j,Xv̂_j⟩ Xv̂_j ].`

The bracket is one generalized Gram–Schmidt step; the outer `2M` is the Oja/power-iteration
matrix product. On `M = I`, the fixed point `v̂_i ← ½∇u_i` in sequence is classical
Gram–Schmidt. The Generalized Hebbian Algorithm equals this gradient with the **reward**
projected onto the sphere but the **penalty** left unprojected — which is precisely why GHA is
not the gradient of any function (non-symmetric Jacobian).

**Single-basin landscape + global convergence.** Along any deviation `v̂_i = cos(θ_i)v_i +
sin(θ_i)Δ_i` with exact parents, `u_i = Λ_ii − sin²(θ_i)(Λ_ii − Σ_{l>i} z_l Λ_ll)` — a
sinusoid of period `π` (since `±v_i` are the same component): non-concave but every local max
is global. With approximate parents the utility is `A sin²(θ_i) − B sin(2θ_i)/2 + C`; the
maximizer's angular error obeys an `arctan` soft-step in the parents' error, giving a
threshold: parents learned to within `c_i·g_i/((i−1)Λ_{11})` (`c_i ≤ 1/16`, `g_i = Λ_ii −
Λ_{i+1,i+1}`) yield child error `≤ 8c_i`. Combined with a nonconvex Riemannian rate
(`1/ρ²` iterations to `‖∇^R‖ ≤ ρ`), the sequential algorithm converges to within `θ_tol`
**independent of initialization**, with total iterations
`T_k = ⌈ O( k·[ (16Λ_{11})^k (k−1)! / (Π_{j=1}^{k} g_j) · 1/θ_tol ]² ) ⌉`.

**Decentralization.** Each update needs only the parents' broadcast vectors — no `QR`, no SVD,
no `d×d` matrix; the sole coupling is the masked `k×k` ratio matrix. Run one player per device,
update in parallel (parents become quasi-stationary near their optima), broadcast each step.

## Algorithm (Riemannian gradient ascent, parallel)

For each player `i`, repeat: `rewards ← Xv̂_i`; `penalties ← Σ_{j<i}(⟨Xv̂_i,Xv̂_j⟩/⟨Xv̂_j,Xv̂_j⟩)Xv̂_j`;
`∇ ← 2Xᵀ[rewards − penalties]`; `∇^R ← ∇ − ⟨∇,v̂_i⟩v̂_i`; `v̂_i ← v̂_i + α∇^R`;
`v̂_i ← v̂_i/‖v̂_i‖`; `broadcast(v̂_i)`. The variant **without** the Riemannian projection (step
with ambient `∇`, then renormalize) is often more stable near the optimum, since the
renormalization implicitly decays the step there.

## Code

```python
import numpy as np

def reference_components(X, k):
    """Exact eigenvectors of M = X^T X, descending — validation only."""
    w, U = np.linalg.eigh(X.T @ X)
    order = np.argsort(w)[::-1]
    return U[:, order[:k]], w[order[:k]]

def normalize_columns(V):
    return V / np.linalg.norm(V, axis=0, keepdims=True)

def utility(X, V):
    """u_i = ||X v_i||^2 - sum_{j<i} <Xv_i,Xv_j>^2 / <Xv_j,Xv_j>  (per column)."""
    XV = X @ V
    R = XV.T @ XV                                # R_ij = <Xv_i, Xv_j>
    rewards = np.diag(R)
    k = V.shape[1]
    pen = np.array([sum(R[i, j] ** 2 / R[j, j] for j in range(i)) for i in range(k)])
    return rewards - pen

def eigengame_step(X, V, lr, riemannian=True):
    """Each column v_i ascends u_i(v_i | v_{j<i}).
    grad_i = 2 X^T [ X v_i - sum_{j<i} (<Xv_i,Xv_j>/<Xv_j,Xv_j>) X v_j ]
           = 2 X^T X [ v_i - (generalized Gram-Schmidt against parents) ].
    """
    k = V.shape[1]
    XV = X @ V                                   # (n, k): the "rewards" signal
    R = XV.T @ XV                                # (k, k): <Xv_i, Xv_j>
    R_norm = R / np.diag(R)                      # divide each col by parent Rayleigh
    mask = np.tril(2 * np.eye(k) - np.ones((k, k)))   # diag +1 (reward), strict-lower -1 (parents)
    G_s = V @ (R_norm * mask).T                  # column i: v_i - sum_{j<i} (...) v_j
    grad = 2.0 * (X.T @ (X @ G_s))               # Oja matrix product (never forms M)
    if riemannian:                               # project onto sphere tangent space
        grad = grad - V * np.sum(grad * V, axis=0, keepdims=True)
    V = V + lr * grad                            # ascent
    return normalize_columns(V)                  # retract to the sphere

def eigengame(X, k, lr=1e-4, iters=5000, riemannian=True, V0=None):
    d = X.shape[1]
    V = normalize_columns(np.random.randn(d, k)) if V0 is None else V0.copy()
    for _ in range(iters):                       # in parallel: one device per column
        V = eigengame_step(X, V, lr, riemannian)
    return V

def eigengame_streaming(stream, d, k, lr=1e-4, riemannian=False, V0=None):
    """Same step on a stream of minibatches; broadcast(v_i) is the only comm."""
    V = normalize_columns(np.random.randn(d, k)) if V0 is None else V0.copy()
    for Xt in stream:                            # Xt : (m, d)
        V = eigengame_step(Xt, V, lr, riemannian)
    return V

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 30))
    k = 6
    V = eigengame(X, k, lr=3e-4, iters=8000, riemannian=False)
    V_true, w_true = reference_components(X, k)
    print("alignment per component:", np.round(np.abs(np.sum(V * V_true, axis=0)), 4))
    print("recovered Rayleigh:", np.round(np.diag((X @ V).T @ (X @ V)), 2))
    print("true eigenvalues:  ", np.round(w_true, 2))
```

Recovering the **smallest** eigenvectors: run the same game on `M' = Λ_{11}I − M` (its top
eigenvectors are `M`'s bottom ones). Because every term is a generalized inner product
`⟨Xv̂_i, Xv̂_j⟩`, replacing `Xv̂_i` with a learned `f_i(X)` extends the same utility/gradient
machinery beyond the linear case.
