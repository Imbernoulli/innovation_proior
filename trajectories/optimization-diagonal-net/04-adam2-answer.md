**Problem.** Uncorrected Adam beat SGD — matched at the grid floor on the small problem, edged it on
the mid settings, and won at high `d` (`350` vs `487`) — so forgetting+momentum genuinely helps the
diagonal-net geometry, not just un-breaks AdaGrad. The remaining sub-optimal knob is the second-moment
memory: `β₂ = 0.999` averages over ~1000 steps and *lags* the non-stationary support-coordinate escape.

**Key idea.** Shorten the second-moment window. `β₂ = 0.95` gives an effective `1/(1−β₂) = 20`-step
memory, so the denominator tracks each support coordinate's gradient surge as it escapes the saddle,
re-normalizing the per-coordinate rate to the *current* scale rather than the stale one — responsive,
non-damping adaptivity, the opposite of AdaGrad's compounding penalty.

**Why the paired `lr`.** Under the harness's *uncorrected* update, the `t=1` step magnitude is
`(1−β₁)/(1−β₂)^{1/2}·sign(g)`: shortening the window from `β₂=0.999` to `0.95` shrinks the raw early
step ~7× (`3.16 → 0.447`). So the configuration must raise `lr` to `0.1` (double the `β₂=0.999`
setting's `0.05`) to keep the early escape fast — the correction-omission arithmetic forces it. The
short window keeps the noise temperature up, which the noise-driven sparse bias rewards.

**Hyperparameters.** `lr = 0.1`, `β₁ = 0.9`, `β₂ = 0.95`, `eps = 1e-6`; four zero-init EMA buffers
`m_u, s_u, m_v, s_v`; **no bias correction**. Otherwise identical to the `β₂=0.999` rung.

```python
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """Adam (no bias correction) hyperparameters: lr=0.1, beta2=0.95."""
    return {"lr": 0.1, "beta1": 0.9, "beta2": 0.95, "eps": 1e-6}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Adam state: first and second moment estimates."""
    d = u.shape[0]
    return {
        "t": 0,
        "m_u": torch.zeros(d, dtype=torch.float64),
        "s_u": torch.zeros(d, dtype=torch.float64),
        "m_v": torch.zeros(d, dtype=torch.float64),
        "s_v": torch.zeros(d, dtype=torch.float64),
    }


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Adam update step WITHOUT bias correction."""
    lr = float(hyperparameters["lr"])
    beta1 = float(hyperparameters["beta1"])
    beta2 = float(hyperparameters["beta2"])
    eps = float(hyperparameters["eps"])
    t = state["t"] + 1
    m_u = beta1 * state["m_u"] + (1.0 - beta1) * grad_u
    s_u = beta2 * state["s_u"] + (1.0 - beta2) * grad_u * grad_u
    u_new = u - lr * m_u / (torch.sqrt(s_u) + eps)
    m_v = beta1 * state["m_v"] + (1.0 - beta1) * grad_v
    s_v = beta2 * state["s_v"] + (1.0 - beta2) * grad_v * grad_v
    v_new = v - lr * m_v / (torch.sqrt(s_v) + eps)
    return u_new, v_new, {"t": t, "m_u": m_u, "s_u": s_u, "m_v": m_v, "s_v": s_v}
```
