# Synaptic Intelligence (SI), distilled

Synaptic Intelligence is a regularization method for continual learning. Each parameter
accumulates, *online and along the entire training trajectory*, an importance measure equal to
its contribution to the drop in the task loss; when a new task is trained, changes to important
parameters are penalized by a per-parameter quadratic anchored at their values from the end of
the previous task. It recovers curvature-like importance for free from the gradients training
already computes — no separate Fisher-estimation phase, no extra backward pass, and constant
memory in the number of tasks.

## Problem it solves

Sequential (continual) learning of tasks `μ = 1, 2, …` where only the current task's loss `L_μ`
is available, but the objective is the sum `L = Σ_μ L_μ` over all tasks ever seen. Descending
`L_μ` moves shared weights and inadvertently raises earlier losses `L_ν` (ν < μ) — catastrophic
forgetting. The fix must hold important past-task parameters in place while leaving the rest
free to learn the new task, computed cheaply, locally, and online.

## Key idea

Measure each parameter's importance by the *path integral* of the gradient against the
parameter's own motion over the task, accumulated step by step, then penalize changes to
important parameters with a per-parameter quadratic.

- **Per-parameter importance over the trajectory.** A first-order expansion gives the loss
  change of an infinitesimal step as `L(θ+δ) − L(θ) ≈ Σ_k g_k δ_k`, `g_k = ∂L/∂θ_k`. Summing
  over the whole training path is the line integral `∫ g·dθ = ∫ g(θ(t))·θ'(t) dt`; since the
  gradient field is conservative this equals the net loss change `L(θ_end) − L(θ_start)`, which
  is negative during descent, and it decomposes per coordinate. The per-parameter piece defines
  importance:

  ```
  ω_k^μ ≡ − ∫_{t_{μ-1}}^{t_μ} g_k(θ(t)) · θ'_k(t) dt
        ≈ Σ_steps ( − g_k · Δθ_k )      (discrete; Δθ_k = optimizer update at that step)
  ```

  The minus sign tracks loss *decrease*; the running sum reuses the gradients training already
  produces, so importance is obtained online with no extra pass.

- **Normalized consolidation strength.** Demand a quadratic surrogate that reproduces the same
  loss drop over the same parameter motion `Δ_k^ν = θ_k(t_ν) − θ_k(t_{ν-1})`. That forces
  dividing the path integral by `Δ²`, and accumulating across past tasks:

  ```
  Ω_k^μ = Σ_{ν<μ}  ω_k^ν / ( (Δ_k^ν)^2 + ξ )
  ```

  The `Δ²` normalization makes the penalty carry the **same units as the loss**, and (see
  Theory) strips the displacement scale so that for a quadratic loss the coefficient in the
  no-`½` quadratic penalty has the right Hessian curvature. `ξ` is a small damping constant
  that bounds the expression when `Δ_k → 0` (a parameter that did not move).

- **Quadratic penalty.** Add to the loss of every later task a per-parameter spring anchored at
  the reference weights `θ̃_k = θ_k(t_{μ-1})` (the values at the end of the previous task):

  ```
  L̃_μ = L_μ + c Σ_k Ω_k^μ ( θ̃_k − θ_k )^2
  ```

  `c` is a single dimensionless strength trading old memories against capacity for the new task.

## Defaults and why

- `c` (penalty strength). If the path integral were exact, `c = 1` would weight old and new
  equally. SGD makes the gradient noisy, and the running sum *overestimates* the true path
  integral, so `c` is typically taken **below 1** to compensate (smaller on noisier/harder
  problems; e.g. ~1 on split MNIST, ~0.1 on permuted MNIST, `1e-3`–`0.1` swept on CIFAR).
- `ξ` (damping). A small constant in the denominator so a parameter with `Δ_k ≈ 0` does not get
  divergent importance (e.g. `1e-3` on split MNIST, `0.1` on permuted MNIST).
- **ReLU on running importance.** Noisy gradients can drive a normalized increment negative. A
  negative final stiffness would *reward* moving the weight away from its anchor, so the running
  accumulator is floored after adding the new increment: `Ω ← ReLU(Ω + W/(Δ²+ξ))`.
- **Timing / memory.** The path-integral credit accumulates continuously during a task; `Ω` and
  the reference `θ̃` update only at task boundaries; after folding the normalized increment into
  `Ω`, the per-task accumulator is reset. `Ω` is a single running sum, so memory is constant in
  the number of tasks.

## Theory: importance vs. curvature (quadratic loss)

For `E(θ) = ½(θ−θ*)ᵀ H (θ−θ*)` under continuous gradient descent `τ dθ/dt = −H(θ−θ*)`, the
solution is `θ(t) = θ* + e^{−Ht/τ}(θ(0)−θ*)`. Since `τ dθ/dt = -g`, the importance
`ω_k = -∫ g_k θ'_k dt` is the diagonal of `Q = τ ∫_0^∞ (dθ/dt)(dθ/dt)ᵀ dt`. In the Hessian eigenbasis (`λ^α, u^α`,
`d^α = u^α·(θ(0)−θ*)`):

```
Q_ij = Σ_{αβ}  u_i^α d^α ( λ^α λ^β / (λ^α + λ^β) ) d^β u_j^β       (τ cancels)
```

Averaging over random initial conditions (`⟨d^α d^β⟩ = σ² δ_{αβ}`):

```
⟨Q_ij⟩ = ½ σ² Σ_α u_i^α λ^α u_j^α = ½ σ² H_ij
```

so the path-integral matrix reduces to one half of the Hessian up to scale `σ²` — and the `Δ²`
denominator (which averages to `σ²` at `ξ = 0`) removes the displacement scale. The resulting
coefficient is `½ H_kk`; because the penalty is written as `Ω_k(θ_k−θ̃_k)^2` with no leading
`½`, its local curvature is `H_kk`.
Without averaging, the entries actually stored as per-parameter importances still have clean
forms: for a **diagonal** Hessian, `Q_ii = ½(d^i)² H_ii`, so dividing by the squared net motion
leaves the same `½ H_ii` coefficient. The off-diagonal `Q_ij` for one trajectory need not vanish,
but those entries are not used by the diagonal regularizer. For a **rank-1** Hessian the full
matrix is `Q_ij = ½(d^1)² H_ij`, hence the unnormalized diagonal entries carry the low-rank
Hessian profile; literal per-coordinate `Δ_i²` normalization cancels the eigenvector factor at
`ξ = 0` on coordinates that moved, so the exact statement is about `Q`, not a normalized full
Hessian. The low-rank case is relevant because many directions remain flat and available.

**Contrast with the endpoint Fisher.** The empirical Fisher `F̄ = E[g gᵀ]` evaluated at the
converged point vanishes for a quadratic at its minimum (the gradient is zero there), whereas
the path integral accumulated curvature information *while descending*. SI thus recovers useful
trajectory curvature signal that an endpoint estimate discards, and does it with no extra
gradient evaluation.

## Relation to prior methods

- **Elastic weight consolidation (Kirkpatrick et al. 2017).** Same quadratic-anchor shape,
  `L_B + Σ_i (λ/2) F_i (θ_i − θ*_{A,i})²`, but stiffness `F_i` is the diagonal Fisher computed
  as a *point estimate at the endpoint* in a *separate phase* after each task; SI computes
  importance online over the trajectory, no extra pass, constant memory.
- **Functional / distillation regularizers (Li & Hoiem 2016).** Constrain the network's
  outputs via the old network; require a forward pass through a stored old network per datapoint.
- **Architectural methods (Rusu et al. 2016).** Grow the network per task; memory grows with
  the number of tasks.

## Working code

The method fills two slots a continual-learning training loop exposes: `estimate_importance`,
run once after each task, turns the per-step accumulated path integral `W_k = Σ (−g_k Δθ_k)`
into the increment needed for the running importance update; `compute_regularization_loss`, run
every step, returns the quadratic penalty. The per-step accumulation of `W` is maintained by the
loop and read here.

```python
import torch


def estimate_importance(model, dataset, prev_params, device):
    """SI importance, computed once after a task finishes.

    The surrounding loop adds this return value into model._custom_importance.
    In the pathint optimizer protocol the nonnegative floor is applied to the
    running task-boundary update; this hook returns the current context's raw
    normalized increment.
    """
    epsilon = getattr(model, 'epsilon', 0.1)     # damping ξ: bounds ω when Δ_k -> 0
    omega = {}
    W = getattr(model, '_custom_W', {})          # accumulated Σ -g·Δθ over the task

    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                theta = p.detach().clone()
                theta_ref = prev_params.get(n, theta)        # θ̃: snapshot at task start
                delta = theta - theta_ref                    # Δ_k: net motion over the task
                w = W.get(n, torch.zeros_like(theta))
                omega[n] = w / (delta ** 2 + epsilon)        # W/(Δ^2+ξ)

    return omega


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """SI consolidation penalty, added at every training step (scaled by c outside):
        Σ_k Ω_k (θ_k - θ̃_k)^2 ,
    with Ω_k the importance accumulated across past tasks and θ̃_k the reference weights.
    """
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    omega = importance_dict[n]               # Ω_k: consolidation strength
                    ref = prev_params_dict[n]                # θ̃_k: reference weights
                    losses.append((omega * (p - ref) ** 2).sum())
    if losses:
        return sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
```

In the canonical implementation the per-step path integral and the cross-task accumulation are
explicit: each step adds `−g_k · Δθ_k` to a running accumulator using the unregularized task
gradient and the optimizer's weight update; at each task boundary the normalized increment
`W_k/((Δ_k)² + ξ)` is added to the running importance, the result is passed through ReLU, the
reference weights are re-snapshotted, and the per-step accumulator is reset to zero. In this
MLS-Bench signature the training loop already supplies `_custom_W` and performs the cross-context
summing, so `estimate_importance` returns the raw normalized increment and
`compute_regularization_loss` returns the no-`½` quadratic penalty.
