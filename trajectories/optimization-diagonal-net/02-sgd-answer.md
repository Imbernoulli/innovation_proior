**Problem.** AdaGrad's per-coordinate denominator damped the support coordinates whose multiplicative
escape is the diagonal-net's recovery engine, so it needed the most samples — worst at high `d`
(`n* = 2000` at `d10000`). The fix is to stop fighting the geometry: strip the adaptivity out and let
plain gradient descent ride the escape dynamics unimpeded.

**Key idea.** Bare SGD, `w ← w − γ g`, applied to `u` and `v`. On the diagonal-net the step is
multiplicative (`grad_u = 2u·residual`), so from the near-zero start support coordinates escape the
saddle first and reach a sparse interpolator before off-support coordinates overfit; the harness's
per-step Rademacher label noise rides this and *strengthens* the sparse bias (the provable benefit of
stochasticity here). Plain GD adds no denominator and no rate decay, so it does nothing to blur the
escape ordering.

**Why.** A *constant* step keeps the label-noise temperature alive (the noise is the regularizer, not a
defect to anneal away) and keeps the multiplicative escape fast. Pick the top of the stable step range,
`lr = 0.1`: the binding failure was too-slow/damped escape at high `d`, so push the step up, not down —
larger `γ` means faster escape and a hotter, more sparsity-favouring noise, short of destabilizing the
`u² − v²` cancellation.

**Hyperparameters.** `lr = 0.1`, constant (no annealing). No momentum, no per-coordinate state; only a
step counter. Identical update on `u` and `v`.

```python
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """SGD hyperparameters: lr=0.1."""
    return {"lr": 0.1}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """SGD requires no additional state."""
    return {"t": 0}


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Vanilla gradient descent step."""
    lr = float(hyperparameters["lr"])
    state["t"] = state.get("t", 0) + 1
    return u - lr * grad_u, v - lr * grad_v, state
```
