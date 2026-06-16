**Problem.** On the diagonal-net `w = u² − v²`, the optimizer's implicit bias — not the loss — decides
which zero-loss solution is selected when `n < d`, and so decides the sample complexity of recovering a
`k`-sparse ground truth. From the near-zero start the parameterization grows coordinates
multiplicatively, escaping the support coordinates first; the first rung must measure what
*coordinate-wise adaptivity* does to that bias, against the vanilla-GD default.

**Key idea.** AdaGrad gives each coordinate of `u` and `v` its own rate by dividing the step by the
square root of its accumulated squared gradient, `lr / (sqrt(Σ_τ g_τ²) + eps)` — the hindsight-optimal
diagonal preconditioner from the mirror-descent regret bound, provably strong on sparse online data.
Here it is applied full-batch to the square-root coordinates: an accumulator per parameter vector, a
square root, a divide.

**Why (and the worry).** The regret theory predicts a sparse-data win, but on this non-convex
parameterization the same denominator *damps* the support coordinates (whose accumulated gradient mass
grows fast) and relatively *inflates* the off-support ones, blurring the saddle-to-saddle escape
ordering that makes the diagonal-net sparse — and the monotone denominator decays the effective rate.
The bet is that adaptivity hurts the implicit bias here; the number will say.

**Hyperparameters.** Canonical AdaGrad, untuned: `lr = 0.01`, `eps = 1e-6`; two zero-initialised
`float64` accumulators `state_sum_u`, `state_sum_v` of length `d`; identical update on `u` and `v`. No
per-setting tuning, no momentum — a clean read on adaptivity alone.

```python
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """AdaGrad hyperparameters: lr=0.01, eps=1e-6."""
    return {"lr": 0.01, "eps": 1e-6}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """AdaGrad state: accumulated squared gradients."""
    d = u.shape[0]
    return {
        "t": 0,
        "g_sum_u": torch.zeros(d, dtype=torch.float64),
        "g_sum_v": torch.zeros(d, dtype=torch.float64),
    }


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """AdaGrad update step."""
    lr = float(hyperparameters["lr"])
    eps = float(hyperparameters["eps"])
    g_sum_u = state["g_sum_u"] + grad_u * grad_u
    g_sum_v = state["g_sum_v"] + grad_v * grad_v
    u_new = u - lr * grad_u / (torch.sqrt(g_sum_u) + eps)
    v_new = v - lr * grad_v / (torch.sqrt(g_sum_v) + eps)
    return u_new, v_new, {"t": state["t"] + 1, "g_sum_u": g_sum_u, "g_sum_v": g_sum_v}
```
