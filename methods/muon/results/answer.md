# Muon

## Method

Muon is a matrix-aware optimizer for hidden 2D neural-network weights. For a momentum update
`M = U Sigma V^T`, it replaces the singular values by roughly unit values and uses the polar-style
direction `U V^T`. The direction is the steepest-descent direction under the spectral operator norm,
and it is also the closest semi-orthogonal matrix to `M` in Frobenius norm in the full-rank case.

The exact polar factor is expensive to compute by SVD or inverse roots, so Muon uses a tuned
quintic Newton-Schulz iteration in bfloat16:

```
X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k
(a, b, c) = (3.4445, -4.7750, 2.0315)
```

The matrix is first Frobenius-normalized so its singular values are at most one. The tuned quintic
is an approximate orthogonalizer: it lifts small singular values quickly and returns something like
`U S' V^T` with `S'` near one, rather than an exact `U V^T`.

For an update of shape `[A,B]` and rank `r`, `RMS(U_{:,:r} V_{:r,:}^T) = sqrt(r/(A B))`; in the
full-rank case this is `1/sqrt(max(A,B))`. The large-scale report therefore scales Muon updates by
`0.2 * sqrt(max(A,B))`, matching AdamW's empirical update RMS range while removing the
full-rank shape dependence. Decoupled weight decay is applied directly to the weights:

```
W <- (1 - lr * wd) W - lr * update
```

Muon is used for hidden 2D weights. Embeddings, LM heads, gains, and biases remain on AdamW.

## Reference Code

This is the report-style single-device core, using the Moonlight scaling convention and the current
KellerJordan/Muon Newton-Schulz behavior for batched matrices.

```python
import math
import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate the polar factor / zeroth power of G with bfloat16 matmuls."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """Keller-style EMA momentum plus Newton-Schulz orthogonalization."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    return zeropower_via_newtonschulz5(update, steps=ns_steps)


def adamw_update(grad, exp_avg, exp_avg_sq, step, betas, eps):
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    m_hat = exp_avg / (1 - betas[0] ** step)
    v_hat = exp_avg_sq / (1 - betas[1] ** step)
    return m_hat / (v_hat.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for hidden matrices; AdamW for embeddings, heads, gains, and biases."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 1e-3)
                group["momentum"] = group.get("momentum", 0.95)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["weight_decay"] = group.get("weight_decay", 0.1)
                group["nesterov"] = group.get("nesterov", True)
            else:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-8)
                group["weight_decay"] = group.get("weight_decay", 0.1)
        super().__init__(param_groups, dict())

    @staticmethod
    def report_adjusted_lr(lr, shape):
        A, B = shape[:2]
        return lr * 0.2 * math.sqrt(max(A, B))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if group["use_muon"]:
                    g = p.grad.view(p.grad.size(0), -1) if p.grad.ndim > 2 else p.grad
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    update = muon_update(
                        g,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                    )
                    p.mul_(1 - lr * wd)
                    p.add_(update.reshape(p.shape), alpha=-self.report_adjusted_lr(lr, p.shape))
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    update = adamw_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"],
                        state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - lr * wd)
                    p.add_(update, alpha=-lr)
        return loss
```

KellerJordan/Muon uses the same Newton-Schulz core but keeps a different learning-rate convention:
after orthogonalization it multiplies by `sqrt(max(1, rows / cols))`, while the large-scale report
folds `0.2 * sqrt(max(A,B))` into the per-parameter learning rate.

Distributed Muon keeps the same update but cannot orthogonalize shards independently. A ZeRO-1
implementation updates local momentum, gathers the momentum shards into the full matrix, runs
Newton-Schulz on the full matrix in bfloat16, keeps the local slice of the update, and all-gathers the
updated parameters.
