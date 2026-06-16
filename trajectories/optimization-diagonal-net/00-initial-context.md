## Research question

Can an optimizer recover a sparse linear predictor from *fewer* training samples when the model is a
diagonal-net? The model reparameterizes a linear predictor as `w = u**2 - v**2` (element-wise), with
`u, v ∈ R^d` the trainable parameters. Algebraically this is still a linear model — `w` ranges over all
of `R^d` — but the squared parameterization makes the loss non-convex, and that non-convex geometry
*interacts with the optimizer's implicit bias*. Two optimizers that both drive the training loss to
zero can land at very different `w`: one dense, one sparse. Since the ground truth is `k`-sparse in a
large ambient `d`, the optimizer whose dynamics favour sparse solutions recovers from far fewer
samples. The single thing being designed is the **update rule** for `(u, v)`; everything else — the
model, the data, the label noise, the stopping rule, the sample-size search — is fixed. The score is
the negative-log of the smallest training set size that recovers, so the entire contest is *sample
complexity of recovery*, decided by implicit bias.

## Prior art before the first rung (the optimizer lineage)

The first rung is plain gradient descent on `(u, v)`. It is the resolution of a long line of
first-order methods; these are the ancestors the ladder reacts to, each with the gap that motivates the
next.

- **Batch gradient descent (Cauchy 1847).** Steepest descent on the empirical loss,
  `w ← w − γ ∇R_n(w)`. On a strongly convex objective it converges linearly, but every step costs a
  full pass over the data, and — the point here — on a *non-convex* parameterization the limit it
  reaches is not pinned by the loss alone: which zero-loss minimizer it selects is an implicit-bias
  question the convergence theory says nothing about. Gap: the loss does not determine the solution.
- **Stochastic approximation / SGD (Robbins and Monro 1951).** Replace the exact gradient with a noisy
  unbiased estimate and step `w ← w − γ g`; the noise that looks like a defect turns out to *shape* the
  limit. On the diagonal-net the per-step label noise is exactly such a perturbation, and recent
  analysis (Pesme, Pillaud-Vivien, and Flammarion, NeurIPS 2021; arXiv:2106.09524) shows it
  strengthens the sparse implicit bias. Gap: a single global step size treats every coordinate the
  same, which may or may not be what the sparse geometry wants.
- **AdaGrad (Duchi, Hazan, and Singer, JMLR 2011).** Give each coordinate its own rate by dividing the
  step by the square root of its accumulated squared gradient, `γ / sqrt(Σ_τ g_τ²)`. Provably strong
  regret on sparse, heavy-tailed data — but the denominator only grows, so the effective rate decays
  toward zero, and on a non-convex landscape that per-coordinate rescaling *changes which minimizer is
  selected*. Gap: re-weighting the geometry can help or hurt the implicit sparse bias, unpredictably.
- **Adam without bias correction (after Kingma and Ba, ICLR 2015; arXiv:1412.6980).** Replace AdaGrad's
  growing sum with an EMA of squared gradients and add an EMA first moment, but *omit* the
  `1/(1−β^t)` bias-correction terms — deliberately, to study the raw adaptive geometry the
  preconditioner imposes. Gap: whether smoothing and forgetting the second moment preserves or destroys
  the diagonal-net's sparse bias is exactly what the ladder measures.

## The fixed substrate

A diagonal-net sparse-recovery harness is frozen and must not be touched. The model is
`DiagonalNet`: `u, v ∈ R^d` (`float64`), initialised to `alpha/sqrt(2d) · 1` with `alpha = 1e-3`, so
`u = v` at init and `w_hat = u² − v² = 0` — every run starts at the origin in predictor space. The
forward map is `x @ (u² − v²)`. Training is **full-batch**: each step recomputes the gradient on the
*entire* training set, but with **fresh per-step Rademacher label noise** `ζ_t ∈ {−delta, +delta}`
added to `y` before the loss `0.5·mean((Xw − y_noisy)²)` — this is the only stochasticity, and it is
the perturbation the sparse-bias dynamics ride on. Evaluation always uses clean labels; recovery means
test MSE `< 1.0`. Training halts on a two-window plateau (train and test MSE both flat over 20k steps)
or at 1,000,000 steps. PyTorch autograd supplies the gradients; the optimizer never touches the model
or the loss — only the parameter vectors and their gradients.

## The editable interface

Exactly one region is editable — three functions in `custom_optimizer.py`. Every method on the ladder
is a fill of this same contract:

- `get_hyperparameters(dim, sparsity, delta)` → a config dict (the harness passes the problem's `d`,
  `k`, and noise magnitude; note there is no separate `sigma`/`noise_scale` argument — the only noise
  knob the optimizer sees is `delta`).
- `init_state(u, v, hyperparameters)` → a mutable state dict (momentum buffers, accumulators); any
  tensors it returns are moved to the device by the harness.
- `step(u, v, grad_u, grad_v, state, hyperparameters)` → `(u_new, v_new, state_new)`, one update.

`grad_u, grad_v` are the **full-batch** MSE gradients w.r.t. `u` and `v` (with the current step's noisy
labels). All ops are `torch` on `float64`. The starting point is the scaffold default: **vanilla
gradient descent with `lr = 0.01`**. Each method on the ladder replaces exactly these three functions.

```python
# EDITABLE region of custom_optimizer.py — default fill (vanilla GD, lr=0.01)
def get_hyperparameters(
    dim: int,
    sparsity: int,
    delta: float,
) -> dict[str, Any]:
    """Return optimizer hyperparameters for this problem setting."""
    return {"lr": 0.01}


def init_state(
    u: torch.Tensor,
    v: torch.Tensor,
    hyperparameters: dict[str, Any],
) -> dict[str, Any]:
    """Initialise optimizer state from the model parameters u, v."""
    return {"t": 0}


def step(
    u: torch.Tensor,
    v: torch.Tensor,
    grad_u: torch.Tensor,
    grad_v: torch.Tensor,
    state: dict[str, Any],
    hyperparameters: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Perform one optimizer step (vanilla gradient descent)."""
    lr = float(hyperparameters["lr"])
    state["t"] = state.get("t", 0) + 1
    return u - lr * grad_u, v - lr * grad_v, state
```

## Evaluation settings

Four settings span dimension, sparsity, and noise: **d200_k5_s01** (`d=200, k=5, delta=0.5`),
**d500_k10_s01** and **d500_k10_s02** (`d=500, k=10, delta=0.5`, differing only in the clean-label
generation seed/variance label), and a larger hidden **d10000_k50** (`d=10000, k=50`). For each
setting the harness runs a coarse-to-fine search over training-set sizes
`n ∈ {50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600}` (wider for the large setting), finding
the smallest `n*` that recovers (test MSE `< 1.0`) on at least 4 of 5 seeds. The metric per setting is
`score = −log2(n*)` — higher is better, i.e. fewer samples needed. Seed for the reported runs: 42.
