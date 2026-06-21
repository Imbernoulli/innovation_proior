Training very large neural networks with Adam works because it rescales every parameter's update by the inverse square root of its own recent squared-gradient magnitude. That per-coordinate normalization handles the huge scale differences across parameters, but it costs one full model-sized buffer for the second-moment accumulator, and another if momentum is used. For billion-parameter models with enormous embedding and weight matrices, those optimizer-state copies, not FLOPs, become the reason a model cannot fit in memory. The goal is to keep Adam's per-coordinate adaptivity while shrinking the extra state so it grows sublinearly with the parameter count of a large matrix.

A tempting shortcut is to compress the second-moment matrix with a low-rank approximation such as truncated SVD, but that fails for two reasons. The second moment is updated every step as an exponential moving average, and SVD factors do not commute with that smoothing: the moving average of the factors is not the factors of the moving average, so one would have to reconstruct the full matrix each step and refactor it. Worse, SVD factors can be negative, while a second moment is nonnegative and must stay square-rootable. What is needed is a factorization whose factors are nonnegative and whose summary statistics are linear in the second-moment matrix, so smoothing and factoring can be interchanged.

The method is Adafactor. For a matrix parameter X in R^{n x m}, Adafactor stores only running row and column summaries of the squared-gradient matrix instead of the full n-by-m second moment. The reconstruction is the rank-1 nonnegative outer product that minimizes the I-divergence, a generalized KL divergence suited to nonnegative matrices. Minimizing sum_{ij} d(V_{ij}, R_i S_j) with d(p,q) = p log(p/q) - p + q yields row factors proportional to the row sums of V and column factors proportional to the column sums. Fixing the scale ambiguity gives V-hat_{ij} = R_i C_j / sum_k R_k, which is exact when V is rank 1 and nonnegative by construction.

Because row sums and column sums are linear in V, they can be maintained directly as exponential moving averages of the squared gradient's row and column sums. This drops the second-moment storage from O(nm) to O(n+m) for matrices. For scalars and vectors there is no factorization gain, so the ordinary unfactored second moment is kept; those are small anyway. Adafactor also drops the first-moment buffer by default, so there is no momentum array at all.

Removing momentum exposes an instability that ordinary momentum had been masking. A slowly decaying second moment can become stale, making g / sqrt(v-hat) much larger than intended, and training can blow up early. Adafactor detects this through the RMS of the raw unscaled update U = g / sqrt(v-hat); when v-hat tracks g^2 correctly, that RMS is near 1, so values well above 1 indicate oversized steps. The fix is to clip the actual update by its RMS rather than the gradient norm: U-hat = U / max(1, RMS(U)/d), with d = 1. This caps the real step after adaptive rescaling, which is where the problem lives.

The staleness is also addressed at the source by using an increasing second-moment decay rate beta-hat_{2,t} = 1 - t^{-c}, with c = 0.8. It starts at zero on the first step and rises toward 1, so the estimator forgets fast early when the model is changing quickly and remembers longer later. A decay schedule with beta-hat_{2,1} = 0 makes the EMA weights sum to 1 by induction, which removes the need for a separate bias correction. The requirement c <= 1 ensures old gradients eventually lose their weight.

Finally, the step size is made relative to the parameter scale: alpha_t = max(eps2, RMS(X_{t-1})) * rho_t. Each parameter moves by a fixed fraction of its own magnitude, so a single schedule works across differently scaled parameter groups. A small floor eps2 lets near-zero initialized parameters escape from zero.

Putting this together, the update for a matrix is: compute the squared gradient plus a tiny floor eps1, update row and column mean buffers with the increasing decay, reconstruct 1 / sqrt(v-hat) from the row and column means, multiply by the gradient to get U, clip U by its RMS, multiply by the relative step size, and apply the update. Vector and scalar parameters follow the same pattern but with an unfactored v buffer. Optional first-moment momentum can be re-enabled if desired, at the cost of a full extra buffer.

```python
import math
import torch
from torch.optim import Optimizer


class Adafactor(Optimizer):
    """Adaptive optimizer with sublinear extra memory for matrices."""

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
        if group["relative_step"]:
            min_step = 1e-6 * state["step"] if group["warmup_init"] else 1e-2
            rel = min(min_step, 1.0 / math.sqrt(state["step"]))
        scale = 1.0
        if group["scale_parameter"]:
            scale = max(group["eps"][1], state["RMS"])
        return scale * rel

    @staticmethod
    def _approx_sq_grad(row_mean, col_mean):
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
                    if factored:
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

                beta2t = 1.0 - state["step"] ** group["decay_rate"]
                sq = grad ** 2 + group["eps"][0]

                if factored:
                    row, col = state["row"], state["col"]
                    row.mul_(beta2t).add_(sq.mean(dim=-1), alpha=1.0 - beta2t)
                    col.mul_(beta2t).add_(sq.mean(dim=-2), alpha=1.0 - beta2t)
                    update = self._approx_sq_grad(row, col)
                    update.mul_(grad)
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

Recommended settings are eps1 = 1e-30, eps2 = 1e-3, clip threshold d = 1, relative schedule rho_t = min(1e-2, 1/sqrt(t)), increasing decay beta-hat_{2,t} = 1 - t^{-0.8}, and no first moment by default. With these choices, Adafactor preserves Adam-style per-coordinate adaptive updates while the extra optimizer state for a matrix shrinks from a full n-by-m buffer to two vectors of length n and m.
