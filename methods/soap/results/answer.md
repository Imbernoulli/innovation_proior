# SOAP (ShampoO with Adam in the Preconditioner's eigenbasis)

## Problem

AdamW preconditions diagonally — one scalar per coordinate — and is blind to the row/column
correlations of a weight matrix, which costs it iterations on large-model pretraining. Shampoo uses a
Kronecker-factored non-diagonal preconditioner L^{-1/2} G R^{-1/2} (L = ΣGGᵀ, R = ΣGᵀG) and converges
faster, but L^{-1/2}, R^{-1/2} need an eigendecomposition that is only affordable every f steps; since
Shampoo's adaptivity *is* that eigen-refresh, its preconditioner goes stale between refreshes and
degrades as f grows. SOAP keeps Shampoo's non-diagonal preconditioning at near-Adam cost and adds only
one hyperparameter.

## Key idea

Shampoo with power 1/2 (plus the Trace(L) scalar correction and dataset-average factors) is *exactly*
Adafactor run in the eigenbasis of Shampoo's preconditioner. Proof: rotate G by the eigenvectors of L
and R, G' = Q_Lᵀ G Q_R; Shampoo scales coordinate (i,j) by (λ_i μ_j / Σλ)^{-1/2}, and the row/column
marginals of the squared rotated gradient satisfy A_i = u_iᵀ E[GGᵀ] u_i = λ_i, C_j = μ_j, so the two
rescalings coincide. The eigendecomposition only computes the *basis*; preconditioning in that basis is
a cheap diagonal rescaling — the second moment of the rotated gradient.

That separation is the lever. Refresh the basis rarely (every f steps), but run a full Adam in that
basis and update its second moment every step in the (slowly drifting) basis: AdamW
in Shampoo's eigenbasis, with only one new hyperparameter over AdamW (the preconditioning frequency f).
Between refreshes the in-basis second moment keeps adapting, which is exactly what Shampoo lacks.

## Algorithm (per m×n layer)

State: Shampoo factors L (m×m), R (n×n); eigenbases Q_L, Q_R; Adam moments M, V (m×n, V in the rotated
basis). Per step t with gradient G:

```
G'  = Q_Lᵀ G Q_R                       # rotate gradient into the eigenbasis (diagonal preconditioning)
M   = β₁ M + (1−β₁) G                  # first moment in the ORIGINAL space
M'  = Q_Lᵀ M Q_R                       # rotate momentum in (survives basis drift)
V   = β₂ V + (1−β₂) (G' ⊙ G')          # second moment IN the basis, every step
N'  = M' / (√V + ε)                    # Adam step in the rotated frame
N   = Q_L N' Q_Rᵀ                      # rotate back to the original space
W   = W − η · c_t · N                  # c_t = √(1−β₂ᵗ)/(1−β₁ᵗ)
W   = W − ηλ W                         # decoupled weight decay, applied outside the Adam moments
L   = β₂ L + (1−β₂) G Gᵀ ; R = β₂ R + (1−β₂) Gᵀ G    # update Shampoo factors
if t % f == 0:  Q_L = QR(L Q_L);  Q_R = QR(R Q_R)    # refresh basis (power iter + QR; eigh on first)
```

Boundary cases: 1D parameters use plain AdamW (no preconditioner); for huge (vocabulary-sized)
dimensions fix that side's rotation to identity; setting *both* rotations to identity for a 2D layer
recovers Adam. The first step only initializes L, R and the basis and skips the update (so a gradient is
never projected through a basis built from itself). On a basis refresh, momentum is reprojected into the
new basis, and the second-moment table is kept aligned with the sorted basis axes.

The implementation can store the first moment in the current basis between refreshes. When the basis
changes, it projects that stored moment back to the original coordinates and then into the new basis,
which preserves the same original-space momentum invariant as the algorithm above.

## Hyperparameters

- lr (Adam's, in the rotated frame), betas (β₁, β₂) ≈ (0.95, 0.95), eps 1e-8, weight_decay (decoupled).
- precondition_frequency f (default 10) — the only addition over AdamW.
- max preconditioner dimension (≈10000): above it, fix that side's rotation to identity.

## Working code

The code keeps the implementation structure: rotate in, update Adam statistics in the basis, rotate
back, refresh the basis with power-iteration + QR, use AdamW-style bias correction and decoupled weight
decay, and fall back to identity projection for 1D or over-large dimensions.

```python
import torch
from torch.optim.optimizer import Optimizer


def _eigh_basis(P):
    P32 = P.float()
    eye = torch.eye(P32.shape[0], device=P32.device, dtype=P32.dtype)
    _, Q = torch.linalg.eigh(P32 + 1e-30 * eye)
    return torch.flip(Q, dims=[1]).to(P.dtype)     # descending eigenvalue order


def _project_2d(X, QL, QR):
    if QL is not None:
        X = QL.t() @ X
    if QR is not None:
        X = X @ QR
    return X


def _project_back_2d(X, QL, QR):
    if QL is not None:
        X = QL @ X
    if QR is not None:
        X = X @ QR.t()
    return X


def _refresh_basis(P, Q_prev, V, dim):
    if P is None or Q_prev is None:
        return None, V
    P32, Q32 = P.float(), Q_prev.float()
    est_eig = torch.diag(Q32.t() @ P32 @ Q32)
    idx = torch.argsort(est_eig, descending=True)
    V = V.index_select(dim, idx)
    Q32 = Q32[:, idx]
    Q32, _ = torch.linalg.qr(P32 @ Q32)
    return Q32.to(Q_prev.dtype), V


class SOAP(Optimizer):
    """AdamW run in Shampoo's eigenbasis; identity projection gives plain AdamW."""

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), eps=1e-8,
                 weight_decay=0.01, precondition_frequency=10,
                 max_precond_dim=10000, shampoo_beta=-1.0, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        precondition_frequency=precondition_frequency,
                        max_precond_dim=max_precond_dim, shampoo_beta=shampoo_beta,
                        correct_bias=correct_bias)
        super().__init__(params, defaults)

    def _init_state(self, p, G, group):
        beta2 = group["betas"][1]
        shampoo_beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else beta2
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)       # stored in the current basis
        state["exp_avg_sq"] = torch.zeros_like(p)
        state["use_precond"] = G.dim() == 2
        state["shampoo_beta"] = shampoo_beta
        if G.dim() != 2:
            return

        m, n = G.shape
        max_dim = group["max_precond_dim"]
        state["L"] = None if m > max_dim else torch.zeros(m, m, device=G.device, dtype=G.dtype)
        state["R"] = None if n > max_dim else torch.zeros(n, n, device=G.device, dtype=G.dtype)
        if state["L"] is not None:
            state["L"].mul_(shampoo_beta).add_(G @ G.t(), alpha=1 - shampoo_beta)
            state["QL"] = _eigh_basis(state["L"])
        else:
            state["QL"] = None
        if state["R"] is not None:
            state["R"].mul_(shampoo_beta).add_(G.t() @ G, alpha=1 - shampoo_beta)
            state["QR"] = _eigh_basis(state["R"])
        else:
            state["QR"] = None

    def _step_size(self, group, t):
        if not group["correct_bias"]:
            return group["lr"]
        beta1, beta2 = group["betas"]
        return group["lr"] * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t)

    def _apply_adamw_update(self, p, update, step_size, lr, wd):
        p.add_(update, alpha=-step_size)
        if wd > 0:
            p.add_(p, alpha=-lr * wd)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd, f = (group["lr"], group["eps"],
                              group["weight_decay"], group["precondition_frequency"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                G = p.grad
                state = self.state[p]

                if "step" not in state:
                    self._init_state(p, G, group)
                    continue

                M, V = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if state["use_precond"]:
                    G_rot = _project_2d(G, state["QL"], state["QR"])
                    M.mul_(beta1).add_(G_rot, alpha=1 - beta1)
                    M_rot = M
                else:
                    G_rot = G
                    M.mul_(beta1).add_(G, alpha=1 - beta1)
                    M_rot = M
                V.mul_(beta2).addcmul_(G_rot, G_rot, value=1 - beta2)

                denom = V.sqrt().add_(eps)
                N_rot = M_rot / denom
                N = _project_back_2d(N_rot, state["QL"], state["QR"]) if state["use_precond"] else N_rot
                self._apply_adamw_update(p, N, self._step_size(group, t), lr, wd)

                if state["use_precond"]:
                    sb = state["shampoo_beta"]
                    if state["L"] is not None:
                        state["L"].mul_(sb).add_(G @ G.t(), alpha=1 - sb)
                    if state["R"] is not None:
                        state["R"].mul_(sb).add_(G.t() @ G, alpha=1 - sb)
                    if t % f == 0:
                        M_orig = _project_back_2d(M, state["QL"], state["QR"])
                        state["QL"], V = _refresh_basis(state["L"], state["QL"], V, dim=0)
                        state["QR"], V = _refresh_basis(state["R"], state["QR"], V, dim=1)
                        state["exp_avg"] = _project_2d(M_orig, state["QL"], state["QR"])
                        state["exp_avg_sq"] = V
        return loss
```
