# Shampoo: Preconditioned Stochastic Tensor Optimization

## Problem

Full-matrix preconditioning — premultiplying the gradient by `H_t^{-1}` where, in full-matrix AdaGrad, `H_t = (Σ_{s≤t} g_s g_s^T)^{1/2}` — captures all pairwise correlations between parameters and enjoys an optimal `O(√T)` regret bound. But for a weight matrix `W ∈ R^{m×n}` the flattened gradient `g = vec(G)` has dimension `mn`, so `H_t` is `mn × mn`: `m²n²` memory and `O(m³n³)` to compute its inverse root. Infeasible. Diagonal methods (AdaGrad, Adam) are cheap but discard all cross-coordinate structure. Shampoo keeps a *full* preconditioner per tensor axis, factorizing across axes via a Kronecker product, at `Σ_i n_i²` memory and `O(Σ_i n_i³)` compute.

## Key idea

A parameter is a tensor, not a flat vector. Model only the geometry *within each axis*. For a matrix, keep a left (row) matrix and a right (column) matrix:
```
L_t = ε I_m + Σ_{s≤t} G_s G_s^T   ∈ R^{m×m},     R_t = ε I_n + Σ_{s≤t} G_s^T G_s   ∈ R^{n×n}.
```
Precondition the gradient on both sides and step:
```
W_{t+1} = W_t − η · L_t^{-1/4} G_t R_t^{-1/4}.
```
Via `(L ⊗ R^T) vec(G) = vec(L G R)` and symmetry of `R_t`, this is exactly the flattened mirror-descent step `w_{t+1} = w_t − η H_t^{-1} g_t` with the implicit Kronecker preconditioner `H_t = L_t^{1/4} ⊗ R_t^{1/4}`, never formed explicitly.

**Why the `−1/4`.** A rank/SVD argument plus a commuting geometric-mean inequality gives the lower bound
```
ε I_{mn} + (1/r) Σ_t g_t g_t^T  ⪯  L_T^{1/2} ⊗ R_T^{1/2}        (r = rank bound on the G_t),
```
so `L^{1/2} ⊗ R^{1/2}` is the analogue of the gradient *covariance* `Σ g g^T`, not of its square root. Full-matrix AdaGrad preconditions with the square root of the covariance, so take one more root: `H = (L^{1/2} ⊗ R^{1/2})^{1/2} = L^{1/4} ⊗ R^{1/4}`. The `1/4 = ½ · ½`: one `½` from splitting the covariance across two axes, one `½` from the AdaGrad square root. It also yields the canonical `O(1/√t)` step decay (`L^{1/4}, R^{1/4} ~ t^{1/4}`).

**Regret (matrix case).** With `D = max_t ‖W_t − W*‖_F`,
```
Σ_t f_t(W_t) − Σ_t f_t(W*)  ≤  √(2r) · D · tr(L_T^{1/4}) tr(R_T^{1/4}),
```
and under `‖G_t‖₂ ≤ 1` each trace is `O(T^{1/4})`, so the bound is `O(√T)` — optimal for online/stochastic convex optimization.

**Tensor case (order `k`).** Keep one matrix per axis from the mode-`i` contraction `G^{(i)} = mat_i(G) mat_i(G)^T`:
```
H^i_t = ε I_{n_i} + Σ_{s≤t} G_s^{(i)},     precondition mode i by (H^i_t)^{-1/(2k)} via the tensor-matrix product ×_i.
```
The exponent `−1/(2k) = (1/k)·(½)`: split the covariance across `k` axes, then the AdaGrad root. For `k = 2` it recovers `−1/4`. Regret `≤ √(2r) D ∏_i tr((H^i_T)^{1/(2k)}) = O(√T)` with `r = (∏_i r_i)^{1/k}`.

## Algorithm (matrix case)

```
Initialize W_1 = 0,  L_0 = ε I_m,  R_0 = ε I_n
for t = 1..T:
    G_t = ∇f_t(W_t)
    L_t = L_{t-1} + G_t G_t^T
    R_t = R_{t-1} + G_t^T G_t
    W_{t+1} = W_t − η L_t^{-1/4} G_t R_t^{-1/4}
```
General order-`k` tensor: for each axis `i`, `H^i_t = H^i_{t-1} + G_t^{(i)}`, `G̃_t ← G̃_t ×_i (H^i_t)^{-1/(2k)}`, then `W_{t+1} = W_t − η G̃_t`.

Inverse `p`-th roots `(H^i)^{-1/(2k)}` are computed from the eigen/SVD of the symmetric PSD matrix (`H^α = Σ_j λ_j^α u_j u_j^T`); the `ε I` ridge keeps all eigenvalues positive. Practical knobs: refresh the roots every 20–100 steps (amortize the SVD), use a gradient running average (momentum, `α ≈ 0.9`), treat each parameter tensor independently (block-diagonal across tensors, so the optimizer needs only tensor shapes — no model-structure knowledge), and fall back to a diagonal preconditioner on any axis too large to store/factorize (replace `G^{(i)}` with `diag(G^{(i)})`; the regret bound then uses the entrywise `D_∞`).

## Code

```python
import torch
from torch.optim.optimizer import Optimizer


def _matrix_power(matrix, power):
    # Inverse p-th root of a symmetric PSD matrix via SVD:
    #   H = U diag(s) V^T  ->  H^power = U diag(s^power) V^T.
    u, s, v = torch.svd(matrix)
    return u @ s.pow(power).diag() @ v.t()


class Shampoo(Optimizer):
    """Per-axis Kronecker-factored preconditioning for tensor parameters.

    For a parameter of order k, keep one statistics matrix H^i per axis i
    (the mode-i gradient contraction summed over steps) and precondition the
    i-th mode of the gradient by (H^i)^{-1/(2k)}. For a matrix (k=2) this is
    the two-sided update W <- W - lr * L^{-1/4} G R^{-1/4}.
    """

    def __init__(self, params, lr=1e-1, momentum=0.0, weight_decay=0.0,
                 epsilon=1e-4, update_freq=1):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        epsilon=epsilon, update_freq=update_freq)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()          # tensor order k
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # H^i = eps * I  and a cache for (H^i)^{-1/(2k)}.
                        state[f"precond_{dim_id}"] = group["epsilon"] * torch.eye(dim, out=grad.new(dim, dim))
                        state[f"inv_precond_{dim_id}"] = grad.new(dim, dim).zero_()

                if momentum > 0:
                    grad.mul_(1 - momentum).add_(state["momentum_buffer"], alpha=momentum)
                if weight_decay > 0:
                    grad.add_(p.data, alpha=weight_decay)

                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # Matricize on axis dim_id: this axis becomes the rows.
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    # H^i <- H^i + mat_i(g) mat_i(g)^T.
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / (2 * order)))

                    if dim_id == order - 1:
                        grad = grad_t @ inv_precond          # right-multiply last mode
                        grad = grad.view(original_size)
                    else:
                        grad = inv_precond @ grad            # left-multiply other modes
                        grad = grad.view(transposed_size)

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-group["lr"])

        return loss
```
