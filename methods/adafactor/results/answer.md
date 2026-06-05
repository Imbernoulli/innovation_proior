# Adafactor: adaptive learning rates with sublinear memory

## Problem

Adam-style adaptive methods rescale each parameter's step by the inverse square root of a per-coordinate exponential moving average of squared gradients, `v_t`. That accumulator has the same shape as the model — one full model-sized buffer — and with momentum a second one. For very large weight and embedding matrices this optimizer state, not compute, becomes the binding constraint on model size. Adafactor keeps the per-coordinate adaptivity but pays only `O(n+m)` extra memory for an `n×m` matrix instead of `O(nm)`.

## Key ideas

1. **Factored second moment.** For a matrix parameter, do not store the full second-moment matrix `V ∈ ℝ^{n×m}`. Store only running averages of its row sums `R ∈ ℝ^n` and column sums `C ∈ ℝ^m`, and reconstruct a rank-1 estimate `V̂_{ij} = R_i C_j / Σ_k R_k` on the fly.

   This particular factorization is the minimizer, over nonnegative rank-1 factors, of the generalized KL (I-)divergence `Σ_{ij} d(V_{ij}, [RS]_{ij})` with `d(p,q) = p log(p/q) − p + q`. On the positive entries, setting the derivatives to zero gives `R_i ∝ Σ_j V_{ij}` (row sums) and `S_j ∝ Σ_i V_{ij}` (column sums); zero rows or columns land on the same formula with zero factors, and the all-zero matrix maps to a zero reconstruction. Fixing the scale gauge yields `R = V1_m`, `C = 1_n^T V`, `V̂ = R C / (1_n^T R)` whenever the grand total is positive. Because row/column sums are **linear** in `V`, exponential smoothing commutes with the factoring — the moving average of the row sums equals the row sums of the moving average — so the small factors can be maintained directly as EMAs without ever forming `V`. (Truncated SVD, the Frobenius-optimal rank-1 fit, fails here: its factors are neither nonnegative nor linear in `V`.)

2. **No first moment.** Set `β₁ = 0` (no momentum buffer); reinstatable as an EMA on the final update if desired.

3. **Update clipping.** Removing momentum and using a slowly-decaying second moment can produce larger-than-desired updates, detectable as `RMS(U_t) = √(mean(g²/v̂))` straying above 1. Clip the *unscaled update* by its RMS: `Û_t = U_t / max(1, RMS(U_t)/d)` with `d=1`. Unlike gradient clipping, this caps the actual step, which adaptive per-coordinate scaling can otherwise inflate beyond any gradient-norm bound.

4. **Increasing decay rate.** Use `β̂₂_t = 1 − t^{-c}`. It starts at 0 (`β̂₂_1 = 0`), so it needs no bias correction — the implied EMA weights provably sum to 1 by induction, making `E[v_t] = E[g_t²]` in the stationary case. It requires `0 < c ≤ 1` so the weight on old gradients vanishes (`Σ j^{-c}` must diverge); `c = 1` is a plain running average, and `c = 0.8` is recommended.

5. **Relative step size.** Scale the step by the parameter's own RMS: `α_t = max(ε₂, RMS(X_{t-1})) · ρ_t`, so each parameter moves a fixed fraction of its magnitude. The floor `ε₂` lets zero-initialized parameters escape 0.

## Algorithm (matrix parameter `X ∈ ℝ^{n×m}`)

```
alpha_t = max(eps2, RMS(X_{t-1})) * rho_t
G_t     = grad
R_t     = beta2hat_t * R_{t-1} + (1 - beta2hat_t) * (G_t^2 + eps1) 1_m   # row sums
C_t     = beta2hat_t * C_{t-1} + (1 - beta2hat_t) * 1_n^T (G_t^2 + eps1) # col sums
V_hat_t = R_t C_t / (1_n^T R_t)                                          # rank-1 reconstruction
U_t     = G_t / sqrt(V_hat_t)
U_hat_t = U_t / max(1, RMS(U_t)/d)                                       # update clipping
X_t     = X_{t-1} - alpha_t * U_hat_t
```

Vector/scalar parameters keep an unfactored `V̂_t = β̂₂_t V̂_{t-1} + (1−β̂₂_t)(G_t²+ε₁)`; everything else is identical. No bias-correction factor anywhere.

**Recommended hyperparameters:** `ε₁ = 10⁻³⁰`, `ε₂ = 10⁻³`, `d = 1`, `ρ_t = min(10⁻², 1/√t)`, `β̂₂_t = 1 − t^{-0.8}`, `β₁ = 0`.

## Code

A drop-in optimizer. Matrices (≥2 axes) are factored into row/column-mean buffers; with `r_i = mean_j(V_ij)` and `c_j = mean_i(V_ij)`, the sum-form reconstruction is exactly `V̂_ij = r_i c_j / mean_i(r)`, so `1/√V̂_ij = rsqrt(r_i/mean(r)) · rsqrt(c_j)`.

```python
import math
import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    def __init__(self, params, lr=None, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, scale_parameter=True,
                 relative_step=True, warmup_init=False):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual `lr` with relative_step=True")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init=True requires relative_step=True")
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, beta1=beta1,
                        scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super().__init__(params, defaults)

    @staticmethod
    def _rms(t):
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _get_lr(group, state):
        rel = group["lr"]
        if group["relative_step"]:                      # rho_t = min(1e-2, 1/sqrt(t))
            min_step = 1e-6 * state["step"] if group["warmup_init"] else 1e-2
            rel = min(min_step, 1.0 / math.sqrt(state["step"]))
        scale = 1.0
        if group["scale_parameter"]:                    # relative step
            scale = max(group["eps"][1], state["RMS"])
        return scale * rel

    @staticmethod
    def _approx_sq_grad(row_mean, col_mean):
        # 1/sqrt(V_hat) = rsqrt(r_i / mean(r)) * rsqrt(c_j)  (rank-1 outer product)
        r = (row_mean / row_mean.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col_mean.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                state = self.state[p]
                factored = grad.dim() >= 2
                use_first_moment = group["beta1"] is not None

                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:                        # O(n)+O(m) buffers
                        state["row"] = torch.zeros(grad.shape[:-1]).to(grad)
                        state["col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:]).to(grad)
                    else:
                        state["v"] = torch.zeros_like(grad)
                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["row"] = state["row"].to(grad)
                        state["col"] = state["col"].to(grad)
                    else:
                        state["v"] = state["v"].to(grad)

                p_data_fp32 = p
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                lr = self._get_lr(group, state)

                beta2t = 1.0 - state["step"] ** group["decay_rate"]   # 1 - t^{-0.8}
                sq = grad ** 2 + group["eps"][0]                       # g^2 + eps1

                if factored:
                    row, col = state["row"], state["col"]
                    row.mul_(beta2t).add_(sq.mean(dim=-1), alpha=1.0 - beta2t)
                    col.mul_(beta2t).add_(sq.mean(dim=-2), alpha=1.0 - beta2t)
                    update = self._approx_sq_grad(row, col)
                    update.mul_(grad)                                  # U = G / sqrt(V_hat)
                else:
                    v = state["v"]
                    v.mul_(beta2t).add_(sq, alpha=1.0 - beta2t)
                    update = v.rsqrt().mul_(grad)

                if group["clip_threshold"] is not None:
                    update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(lr)

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                p_data_fp32.add_(update, alpha=-1.0)
                if p.dtype in {torch.float16, torch.bfloat16}:
                    p.copy_(p_data_fp32)
        return loss
```

Memory per matrix parameter drops from one full `n×m` second-moment buffer to two vectors of length `n` and `m`; with the first moment off, the extra optimizer state is sublinear in the parameter count for matrices.
