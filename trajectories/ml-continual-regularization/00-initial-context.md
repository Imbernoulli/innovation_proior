## Research question

A fixed-capacity network trains on a sequence of contexts, one at a time, and must keep performing on all earlier ones. During training on a context, the learner sees only that context's loss; the data of earlier contexts is gone, and replaying it would cost memory growing with the number of contexts. Ordinary SGD on the current context drags the shared weights wherever the new loss demands, and those same weights encoded the old contexts, so old accuracy collapses—catastrophic forgetting. The design task is a **regularization strategy**: an *importance estimator* (which parameters mattered for each finished context) and a *penalty form* (how their changes are penalized while later contexts train), added on top of an otherwise-fixed training loop.

## Prior art / Background / Baselines

- **Catastrophic interference.** A backprop network trained on a second set of associations abruptly loses the first, because the representation is shared and distributed—any weight move that helps the new context tends to corrupt the old one. Gap: it names the problem, not a solution.
- **Joint / replay training.** Interleave all contexts' data and forgetting vanishes, because the weights are jointly fit. Gap: forbidden here—the old data is gone, and storing it costs memory linear in the number of contexts.
- **Uniform-stiffness L2 anchor.** After a context finishes, snapshot `theta*` and add `sum_i (theta_i - theta*_i)^2` while later contexts train. No data, constant memory. Gap: one global stiffness cannot be right—large enough to hold the old context freezes the network so the new one cannot be learned; small enough to learn the new one fails to hold the old.
- **Curvature framing.** Near a context's optimum, the loss locally approximates a quadratic whose Hessian describes which directions are stiff and which are free. The diagonal Fisher gives a cheap, positive-semidefinite proxy for that Hessian. Gap: it is a local description of one optimum, not yet a prescription for how to estimate, accumulate, or apply importance across a long sequence of contexts with changing data.

## Fixed substrate / Code framework

The training loop is frozen. It sweeps contexts in sequence; on each step it forms `task_loss + (reg_strength * reg_strength_scale) * R(theta)` and backpropagates, where `R` is the penalty the editable region returns. It snapshots parameters at every context boundary, accumulates the returned importance additively into `model._custom_importance`, and exposes per-step bookkeeping a method may use. The hooks:

- `model.param_list` — list of generators yielding `(name, param)` over the regularized parameters (here, `[model.named_parameters]`). Names are stored with `.` replaced by `__`.
- `model._custom_W` — dict the loop accumulates each step as `W[n] += -grad * (p - p_old[n])`: the running gradient-weighted parameter change.
- `model._custom_p_old` — dict of per-step parameter snapshots.
- `model._custom_importance` / `model._custom_prev_params` — the carried importance and the latest-boundary parameter anchor.
- `model.gamma` — decay factor for Fisher accumulation (framework default `1.0`).
- `model.epsilon` — damping constant (default `0.1`, used to bound importance when a weight barely moved).
- `CONFIG_OVERRIDES` — a dict whose only honored key is `reg_strength_scale`, a multiplier on the per-benchmark `reg_strength`.

The benchmark networks are ordinary classifiers: a small MLP for Split-MNIST, a larger MLP for Permuted-MNIST, a CNN for Split-CIFAR100; all multi-head, Adam, cross-entropy. The loop computes the task loss only over the classes present in the current context.

## Editable interface

Exactly one region of `continual-learning/custom_regularization.py` is editable—two functions and the `CONFIG_OVERRIDES` dict. The contract:

- `estimate_importance(model, dataset, prev_params, device) -> dict` — called once after each context finishes. Returns `{param_name: importance_tensor}`. May do forward/backward passes over `dataset` and may read the per-step bookkeeping. The loop sums the returned dict into the carried importance.
- `compute_regularization_loss(model, importance_dict, prev_params_dict) -> Tensor` — called at every training step; must be cheap. Returns the scalar penalty added to the task loss.

The starting point is the scaffold default: diagonal-Fisher importance with a `0.5 * F * (theta-theta*)^2` quadratic penalty.

```python
# EDITABLE region of custom_regularization.py — default fill (diagonal Fisher + quadratic penalty)
def estimate_importance(model, dataset, prev_params, device):
    """Default: diagonal Fisher Information (EWC-style)."""
    est_fisher = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = min(len(data_loader), 200)

    for idx, (x, y) in enumerate(data_loader):
        if idx >= n_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)
        for label_index in range(output.shape[1]):
            label = torch.LongTensor([label_index]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (label_index + 1) < output.shape[1] else False
            )
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher[n] += label_weights[0][label_index] * (p.grad.detach() ** 2)

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}

    model.train(mode=mode)
    return est_fisher


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """Default: EWC quadratic penalty 0.5 * sum(F * (param - prev_param)^2)."""
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    fisher = importance_dict[n]
                    prev = prev_params_dict[n]
                    losses.append((fisher * (p - prev) ** 2).sum())
    if losses:
        return 0.5 * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: reg_strength_scale (multiplier on the per-benchmark reg_strength).
CONFIG_OVERRIDES = {}
```

## Evaluation settings

Three benchmarks span the difficulty range. **Split-MNIST** — task-incremental, 5 binary tasks (2 classes each), a small MLP. **Permuted-MNIST** — domain-incremental, 10 contexts, each a fixed pixel permutation of the same digit classes, a larger MLP; the long sequence of uncorrelated-input contexts is where an over-strong or drifting penalty does the most damage. **Split-CIFAR100** — task-incremental, 10 ten-way tasks, a CNN. One seed (42). Primary metric: **average accuracy across all contexts after training completes** (higher is better), reported per benchmark.
