# Sinkhorn distances: entropy-regularized optimal transport

## Problem

Computing an optimal-transport (Earth Mover's) distance between two histograms `r, c ∈ Σ_d` under a
ground cost `M`,

```
d_M(r,c) = min_{P ∈ U(r,c)} ⟨P, M⟩,   U(r,c) = {P ≥ 0 : P1 = r, P^T1 = c},
```

is a linear program. For a general `M` it costs at least `O(d^3 log d)`, its optimum is a brittle
vertex of the polytope (`≤ 2d−1` nonzeros), and the distance is non-differentiable. This blocks
transport distances from large-scale use.

## Key idea

Don't seek the extreme (vertex) plan; seek the cheapest *smooth* one. Penalize couplings of low
entropy — equivalently bound their mutual information with the independence table `rc^T`:

```
P^λ = argmin_{P ∈ U(r,c)}  ⟨P, M⟩ − (1/λ) h(P),    h(P) = −Σ p_ij log p_ij,
d_M^λ(r,c) = ⟨P^λ, M⟩.
```

For positive marginals, `−h(P)` makes the optimizer **unique** and interior. Writing the Lagrangian
for the two marginal constraints and setting `∂/∂p_ij = 0` gives `(1/λ)(log p_ij + 1) + m_ij + α_i +
β_j = 0`, hence

```
p_ij^λ = u_i K_ij v_j,    K = e^{−λM},
```

where the stationary constant `e^{-1}` has been absorbed into the two scaling vectors. Thus
`P^λ = diag(u) K diag(v)`. This is a **matrix-scaling** problem: by Sinkhorn–Knopp (1967), for
`K > 0` there is a unique scaled matrix with row sums `r` and column sums `c` (the scaling vectors
themselves are unique up to reciprocal rescaling). The marginal conditions are

```
u ⊙ (K v) = r  ⟹  u = r / (K v),       v ⊙ (K^T u) = c  ⟹  v = c / (K^T u),
```

and alternating these two updates (each a KL/Bregman projection onto an affine marginal set)
converges linearly (a Hilbert-projective-metric contraction). One iteration is `O(d²)`, vectorizes
over a family of targets `C = [c₁,…,c_N]` (`O(d²N)`), and runs on a GPU.

`λ = 1/ε` interpolates: `λ → 0` smooths toward the independence plan `rc^T` (`d_{M,0} = r^T M c`);
`λ → ∞` sharpens toward the exact EMD but slows convergence (contraction ratio → 1) and underflows
`e^{−λ m_ij}` to zero. Moderate `λ` is both faster and more robust; for very small `ε`, run the
updates in the log domain.

The entropy-constrained form `d_{M,α} = min_{KL(P||rc^T) ≤ α} ⟨P,M⟩` is symmetric and satisfies the
triangle inequality for all `α ≥ 0` (gluing lemma carried through the data-processing inequality);
multiplying by `1_{r≠c}` restores the coincidence axiom when small `α` keeps `d_{M,α}(r,r)` positive.
For a fixed pair and active entropy constraint, a Lagrange multiplier links `d_{M,α}` to some
`d_M^λ`, but the fixed-`λ` quantity is used as a fast smooth surrogate rather than as a proved metric.

## Algorithm

```
Input: M, λ (or ε = 1/λ), histograms r, c.
K = exp(−λ M)
u, v ← positive vectors
repeat:
    v ← c / (Kᵀ u)
    u ← r / (K v)
until column marginal v ⊙ (Kᵀ u) matches c within tolerance
P ← diag(u) K diag(v)
return d_M^λ = ⟨P, M⟩ = Σ u_i (K ⊙ M)_ij v_j
```

## Code

The direct loop mirrors the standard POT-style Sinkhorn–Knopp implementation. `reg` is `ε = 1/λ`;
for one target it returns the plan, and for a matrix of target histograms the cost routine returns
one loss per column without materializing all plans.

```python
import warnings
import numpy as np

def transport_plan(a, b, M, reg, num_iter=1000, stop_thr=1e-9, warn=True):
    """Entropy-regularized OT plan for one target histogram; reg = eps = 1/lambda."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim != 1:
        raise ValueError("transport_plan expects one target histogram; use transport_cost for many targets.")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    if not np.all(active):
        P = np.zeros_like(M, dtype=float)
        P[active, :] = transport_plan(a[active], b, M[active, :], reg, num_iter, stop_thr, warn)
        return P

    K = np.exp(-M / reg)                       # Gibbs kernel K = exp(-lambda M)

    u = np.ones(M.shape[0]) / M.shape[0]
    v = np.ones(M.shape[1]) / M.shape[1]
    Kp = (1.0 / a)[:, None] * K                # fold diag(1/a) into K for the u-update

    for it in range(num_iter):
        uprev, vprev = u.copy(), v.copy()
        KtU = K.T @ u
        v = b / KtU                            # column projection: enforce marginal b
        u = 1.0 / (Kp @ v)                     # row projection: enforce marginal a
        if KtU.min() == 0 or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            if warn:
                warnings.warn(f"numerical errors at iteration {it}")
            u, v = uprev, vprev
            break
        if it % 10 == 0:
            marginal = v * (K.T @ u)           # (diag(u) K diag(v))^T 1
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    return u[:, None] * K * v[None, :]         # P = diag(u) K diag(v)

def transport_cost(a, b, M, reg, num_iter=1000, stop_thr=1e-9, warn=True):
    """Return <P, M>. If b has target columns, return one cost per column."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim == 1:
        return float(np.sum(transport_plan(a, b, M, reg, num_iter, stop_thr, warn) * M))
    if b.ndim != 2:
        raise ValueError("target histogram must be a vector or a matrix with target columns")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    a = a[active]
    M = M[active, :]

    K = np.exp(-M / reg)
    u = np.ones((M.shape[0], b.shape[1])) / M.shape[0]
    v = np.ones((M.shape[1], b.shape[1])) / M.shape[1]
    Kp = (1.0 / a)[:, None] * K

    for it in range(num_iter):
        uprev, vprev = u.copy(), v.copy()
        KtU = K.T @ u
        v = b / KtU
        u = 1.0 / (Kp @ v)
        if np.any(KtU == 0) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            if warn:
                warnings.warn(f"numerical errors at iteration {it}")
            u, v = uprev, vprev
            break
        if it % 10 == 0:
            marginal = np.einsum("ik,ij,jk->jk", u, K, v)
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    return np.einsum("ik,ij,jk,ij->k", u, K, v, M)


def stable_transport_plan(a, b, M, reg, num_iter=1000, stop_thr=1e-9):
    """Log-domain stabilization for one target when reg is small."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)
    if b.ndim != 1:
        raise ValueError("stable_transport_plan expects one target histogram")

    active = a > 0
    if not np.any(active):
        raise ValueError("source histogram has no positive mass")
    if not np.all(active):
        P = np.zeros_like(M, dtype=float)
        P[active, :] = stable_transport_plan(a[active], b, M[active, :], reg, num_iter, stop_thr)
        return P

    Mr = -M / reg
    loga = np.log(a)
    logb = np.full_like(b, -np.inf, dtype=float)
    logb[b > 0] = np.log(b[b > 0])
    u = np.zeros(M.shape[0])                    # log row scaling
    v = np.zeros(M.shape[1])                    # log column scaling

    def logsumexp(A, axis):
        m = A.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    for it in range(num_iter):
        v = logb - logsumexp(Mr + u[:, None], axis=0)
        u = loga - logsumexp(Mr + v[None, :], axis=1)
        if it % 10 == 0:
            logP = Mr + u[:, None] + v[None, :]
            if np.linalg.norm(np.exp(logP).sum(axis=0) - b) < stop_thr:
                break
    return np.exp(Mr + u[:, None] + v[None, :])         # P = diag(e^u) K diag(e^v)
```
