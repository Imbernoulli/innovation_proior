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

`−h(P)` is strictly convex, so `P^λ` is **unique** and lies in the interior. Writing the Lagrangian
for the two marginal constraints and setting `∂/∂p_ij = 0` gives `(1/λ)(log p_ij + 1) + m_ij + α_i +
β_j = 0`, hence

```
p_ij^λ = u_i K_ij v_j,    K = e^{−λM},    u_i = e^{−½−λα_i},  v_j = e^{−½−λβ_j},
```

i.e. `P^λ = diag(u) K diag(v)`. This is a **matrix-scaling** problem: by Sinkhorn–Knopp (1967), for
`K > 0` there is a unique pair of positive scalings making the row sums `r` and column sums `c`. The
marginal conditions are

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
triangle inequality for all `α ≥ 0` (gluing lemma carried through the data-processing inequality), so
it is a genuine distance; `d_M^λ` is its Lagrangian relaxation, used directly for speed.

## Algorithm

```
Input: M, λ (or ε = 1/λ), histograms r, c.
K = exp(−λ M)
v ← 1
repeat:
    u ← r / (K v)
    v ← c / (Kᵀ u)
until column marginal v ⊙ (Kᵀ u) matches c within tolerance
P ← diag(u) K diag(v)
return d_M^λ = ⟨P, M⟩ = Σ u_i (K ⊙ M)_ij v_j
```

## Code

Faithful to the standard `numpy` Sinkhorn–Knopp loop (as in POT's `ot.sinkhorn`). `reg` is `ε = 1/λ`.

```python
import numpy as np

def sinkhorn(a, b, M, reg, num_iter=1000, stop_thr=1e-9):
    """Entropic-regularized OT plan. a,b: histograms; M: ground cost; reg = eps = 1/lambda."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    M = np.asarray(M, dtype=float)

    K = np.exp(-M / reg)                       # Gibbs kernel K = exp(-lambda M)

    u = np.ones(M.shape[0]) / M.shape[0]
    v = np.ones(M.shape[1]) / M.shape[1]
    Kp = (1.0 / a)[:, None] * K                # fold diag(1/a) into K for the u-update

    for it in range(num_iter):
        KtU = K.T @ u
        v = b / KtU                            # column projection: enforce marginal b
        u = 1.0 / (Kp @ v)                     # row projection: enforce marginal a
        if it % 10 == 0:
            marginal = v * (K.T @ u)           # (diag(u) K diag(v))^T 1
            if np.linalg.norm(marginal - b) < stop_thr:
                break

    return u[:, None] * K * v[None, :]         # P = diag(u) K diag(v)

def sinkhorn_distance(a, b, M, reg, **kw):
    P = sinkhorn(a, b, M, reg, **kw)
    return float(np.sum(P * M))                # <P, M>


def sinkhorn_log(a, b, M, reg, num_iter=1000, stop_thr=1e-9):
    """Log-domain stabilization for small reg (small eps / large lambda).
    Carries log-scalings u = log-potential/reg so log P_ij = -M_ij/reg + u_i + v_j."""
    a = np.asarray(a, float); b = np.asarray(b, float); M = np.asarray(M, float)
    Mr = -M / reg
    loga, logb = np.log(a), np.log(b)
    u = np.zeros(M.shape[0])                    # u = log(scaling) for rows
    v = np.zeros(M.shape[1])                    # v = log(scaling) for cols

    def logsumexp(A, axis):
        m = A.max(axis=axis, keepdims=True)
        return (m + np.log(np.exp(A - m).sum(axis=axis, keepdims=True))).squeeze(axis)

    for it in range(num_iter):
        v = logb - logsumexp(Mr + u[:, None], axis=0)   # column marginal -> b
        u = loga - logsumexp(Mr + v[None, :], axis=1)   # row marginal -> a
        if it % 10 == 0:
            logT = Mr + u[:, None] + v[None, :]
            if np.linalg.norm(np.exp(logT).sum(axis=0) - b) < stop_thr:
                break
    return np.exp(Mr + u[:, None] + v[None, :])         # P = diag(e^u) K diag(e^v)
```
