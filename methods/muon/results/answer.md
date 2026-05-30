# Muon

## The problem

Adam/AdamW treat every weight entry as an independent scalar: they normalize the gradient
coordinate by coordinate (with EMA off, Adam is exactly `sign(g)`). But a hidden layer's weight is
a **matrix** that acts as a linear operator on the hidden space. The empirical update matrices
produced by SGD-momentum or Adam for these 2D parameters have very high condition number — they are
nearly low-rank, so a handful of singular directions absorb almost the entire step while the many
"rare" small-singular-value directions, which still matter for learning, are starved. A per-entry
optimizer cannot see this structure. We want the right notion of a **unit step for a matrix
parameter**.

## The key idea

View an optimizer step as steepest descent under a chosen norm. For a weight matrix the natural
norm is the **spectral (operator ℓ₂→ℓ₂) norm**, because the matrix is an operator on a Euclidean
hidden space. Solving steepest descent under that norm, the optimal update *direction* turns out to
be the **polar factor** of the (momentum) gradient: if `M = U Σ Vᵀ` is the SVD, the update is
proportional to `U Vᵀ` — keep the singular **vectors**, throw away the singular **values**. This is
the matrix sign function `msign(M)`, equivalently the **nearest semi-orthogonal matrix** to `M` in
Frobenius norm, equivalently `(M Mᵀ)^{-1/2} M`. It equalizes the step across all singular
directions, so no direction dominates and the rare directions get a full-size update.

Computing `U Vᵀ` via an SVD is slow and awkward on a GPU. Instead, run a **Newton–Schulz quintic
iteration** that drives every singular value toward 1 (leaving the singular vectors fixed),
without any SVD or matrix inverse: normalize so the singular values start ≤ 1, then iterate a tuned
odd quintic. Five iterations in bfloat16 suffice.

Two more ingredients make it scale to large LLMs:
1. **Decoupled weight decay** — without it, weight and layer-output RMS grow past bfloat16's
   high-precision range and the over-trained regime degrades.
2. **Per-shape update scaling `√max(A,B)`** — a `[A,B]` orthogonalized update has RMS `√(1/max(A,B))`,
   so the step size depends on the matrix shape; multiplying by `√max(A,B)` cancels it, and a `0.2`
   prefactor sets the RMS to ≈ AdamW's typical 0.2, so a single learning rate / weight decay tuned
   for AdamW transfers directly. AdamW handles the embedding, the LM head, and all 1D parameters.

## The algorithm

For each 2D hidden weight matrix `W` of shape `[A,B]`, with momentum `M`, momentum coefficient
`μ=0.95`, learning rate `η`, weight decay `λ`, `N=5` Newton–Schulz steps:

```
M_t = μ M_{t-1} + G_t                         # SGD-momentum (Nesterov in practice)
O_t = NewtonSchulz5(M_t)                       # ≈ U Vᵀ = msign(M_t)
W_t = W_{t-1} − η ( 0.2 · √max(A,B) · O_t + λ W_{t-1} )
```

Newton–Schulz, with `X₀ = M / ‖M‖_F` (singular values then ≤ 1) and tuned coefficients
`(a,b,c) = (3.4445, −4.7750, 2.0315)`:

```
X_{k+1} = a X_k + b (X_k X_kᵀ) X_k + c (X_k X_kᵀ)² X_k
```

Because `X_k = U Σ_k Vᵀ`, each step applies the scalar map `σ ↦ aσ + bσ³ + cσ⁵` to every singular
value while leaving `U, V` fixed; the map's stable fixed point is near 1, so `X_k → U Vᵀ`.

## Working code

Grounded in the canonical single-device implementation (Newton–Schulz, Nesterov momentum, the
`√max(A,B)` / fan scaling, decoupled weight decay), with an auxiliary AdamW for the non-matrix
parameters.

```python
import math
import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate the polar factor U Vᵀ (matrix sign) of G via a quintic Newton–Schulz iteration.

    The coefficients (a, b, c) are tuned so the scalar map g(s) = a s + b s^3 + c s^5 lifts small
    singular values fast (g'(0) = a is large); it converges to s ≈ 1 (roughly Uniform(0.5, 1.5)
    rather than exactly 1, which does not hurt). Runs stably in bfloat16; 5 steps suffice.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Work with the short side as rows so X Xᵀ is the smaller Gram matrix.
    if X.size(0) > X.size(1):
        X = X.T
    # Frobenius-normalize so every singular value starts ≤ 1, inside the convergence basin.
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A            # quintic assembled from the Gram matrix
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    # SGD-momentum (EMA of the gradient); Nesterov look-ahead in practice.
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:                 # conv weight: flatten the last dims into a 2D matrix
        update = update.view(len(update), -1)
    # Orthogonalize: replace the update by its polar factor U Vᵀ.
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    # Cancel the shape-dependent RMS √(1/max(A,B)); 0.2 sets RMS ≈ AdamW's.
    update *= 0.2 * (max(update.size(-2), update.size(-1)) ** 0.5)
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for 2D hidden weight matrices; an internal AdamW for everything else
    (embeddings, the LM head, norm gains, biases). Pass per-group `use_muon` flags."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0.0)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"],
                                         beta=group["momentum"])
                    p.mul_(1 - lr * wd)                       # decoupled weight decay
                    p.add_(update.reshape(p.shape), alpha=-lr)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - lr * wd)                       # decoupled weight decay
                    p.add_(update, alpha=-lr)
        return loss


# Usage: split params by role. Hidden 2D weights → Muon; embeddings / head / scalars → AdamW.
#   hidden = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
#   embed  = [p for n, p in model.named_parameters() if "embed" in n]
#   scalar = [p for p in model.parameters() if p.ndim < 2]
#   head   = [model.lm_head.weight]
#   groups = [dict(params=hidden, use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.1),
#             dict(params=[*embed, *scalar, *head], use_muon=False, lr=3e-4, weight_decay=0.1)]
#   opt = MuonWithAuxAdam(groups)
```

In a distributed (ZeRO-1) setting the optimizer state is partitioned across the data-parallel
group; because orthogonalization is non-elementwise, the per-shard momentum is gathered into the
full matrix, Newton–Schulz is run on the full matrix in bfloat16, and only the local partition of
the result is kept before the update — adding one bf16 gather over a vanilla ZeRO-1 AdamW, with a
single momentum buffer (half AdamW's optimizer memory).
