**Problem.** Bare SGD beat AdaGrad everywhere, biggest at high `d` (`487` vs `2000` at `d10000`),
confirming that AdaGrad's *monotone-accumulation* denominator — which decays the rate to zero and damps
the support escape — was the culprit, not adaptivity as such. Open question: is *all* adaptivity poison
on the diagonal-net, or only the monotone flavour?

**Key idea.** Adam without bias correction: replace AdaGrad's growing `Σ g²` with an EMA
`s_t = β₂ s_{t−1} + (1−β₂) g_t²` (forgetting cures the rate decay) and add an EMA first moment
`m_t = β₁ m_{t−1} + (1−β₁) g_t` (averages the label-noise jitter), then step
`θ ← θ − lr · m_t / (sqrt(s_t)+eps)`. Keeps per-coordinate adaptivity but throws out the
monotone-decay defect.

**Why (and the no-correction subtlety).** The harness *omits* the `1/(1−β^t)` bias correction by design,
to study the raw adaptive geometry. So the moments are used biased-toward-zero at the start: at `t=1`
the step is `≈ (1−β₁)/(1−β₂)^{1/2}·sign(g) ≈ 3.16·sign(g)` with these betas — an *amplified*,
scale-normalized early step, right where the saddle escape happens. The EMA denominator settles (no
runaway, no zero rate), so the residual support-damping is bounded; the scale-normalized step can even
accelerate a small support coordinate's escape.

**Hyperparameters.** Literal baseline: `lr = 0.05` (halved vs SGD because the step is already
normalized to ~unit parameter-space size), `β₁ = 0.9`, `β₂ = 0.999`, `eps = 1e-6`; four zero-init EMA
buffers `m_u, s_u, m_v, s_v`; **no bias correction**.

```python
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """Adam (no bias correction) hyperparameters: lr=0.05, beta2=0.999."""
    return {"lr": 0.05, "beta1": 0.9, "beta2": 0.999, "eps": 1e-6}


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
