# Online EWC

Online EWC is a regularisation-based continual-learning method that prevents catastrophic forgetting at a
cost that is **constant** in the number of contexts. It replaces Elastic Weight Consolidation's per-context
stack of quadratic penalties with a *single* penalty, anchored at the most recent optimum, whose stiffness is
a running, exponentially-decayed sum of the per-context diagonal Fisher information matrices. Memory and
per-step compute do not grow with the number of contexts seen.

## Problem it solves

A network of fixed capacity is trained on a sequence of contexts and must retain all of them. EWC adds a
quadratic spring per past context — anchor `theta*_t` plus diagonal Fisher `F_t` each — so storage and
per-step penalty cost grow *linearly* in the number of contexts, and the accumulating constraints eventually
over-constrain the fixed-capacity network so new contexts can no longer be learned. Online EWC keeps EWC's
protection while making both costs constant in the number of contexts and adding a mechanism that frees
capacity for the future.

## Key idea

Two moves on top of EWC:

1. **Single re-centred penalty (consistent recursive Laplace).** Merely completing the square on EWC's
   existing springs gives a constant-memory quadratic with coordinate-wise anchor
   `(sum_t F_{t,i} * theta*_{t,i}) / (sum_t F_{t,i})` where the denominator is nonzero, but that compactly
   preserves the same multi-anchor object. The Bayesian recursion
   `p(theta | T_{1:k}) ∝ p(theta | T_{1:k-1}) · p(T_k | theta)` only ever needs the running posterior.
   Laplace-approximating the *whole running posterior* (not each likelihood separately) collapses EWC's stack:
   from the third context on, the consistent approximation is one quadratic anchored at the *latest* optimum
   `theta*_{i-1}` with stiffness equal to the *sum* of past Fishers. Keeping the old per-context anchors as
   well double-counts the early contexts (a systematic bias toward them); dropping them is both cheaper and
   the consistent choice, at the cost that the most distant contexts are remembered slightly less well.
   (This single-penalty insight is due to Huszár 2017.)

2. **Decayed running Fisher (stochastic-EP partial removal == graceful forgetting).** Instead of an
   ever-growing sum, down-weight the shared summary by `gamma < 1` before folding in each new context's Fisher:

   ```
   F*_i = gamma * F*_{i-1} + F_i,        0 <= gamma <= 1, with gamma < 1 for forgetting.
   ```

   Unrolled, `F*_i = sum_{t<=i} gamma^{i-t} F_t`: a context `k` boundaries ago contributes weight `gamma^k`
   inside the stored summary, and one more factor of `gamma` when that summary is used in the next loss.
   This bounds the stored summary to an effective `~1/(1-gamma)` recent contexts, with loss-side stiffness
   scaled by one additional `gamma` (room to keep learning on fixed capacity), fades old contexts gracefully
   rather than catastrophically, and — being a single scalar applied identically to every context — handles
   **recurring** contexts (partial removal of the previous presentation) without storing per-context factors
   and **without needing task identities**, only boundaries.
   `gamma = 1` recovers the strict undecayed single-penalty sum. With the loss-side `gamma` convention below,
   `gamma = 0` stores only the most recent Fisher after an update but applies zero old-task penalty on the next
   context. Borrowed from stochastic expectation propagation (Li et al. 2015): a single shared factor updated
   by a fractional remove/add.

The regularisation term for context `K > 1`, anchored at the most recent optimum, is

```
L_reg = (1/2) * gamma * sum_i  F*_{K-1, i} * (theta_i - theta*_{K-1, i})^2,
```

added to the task loss at every step. The `gamma` multiplies the regularising shared factor, matching the
`gamma F*_{i-1}` form; the code below keeps `gamma` in both the accumulator update and the loss-side penalty.

## Importance estimator (diagonal Fisher)

The per-context importance `F_i` is the diagonal of the Fisher information at that context's optimum, the
model's own expected squared score:

```
F_{i, j} = E_x E_{y ~ p_theta(.|x)} [ ( d log p_theta(y|x) / d theta_j )^2 ].
```

Properties that make it the estimator of choice: it equals the expected curvature of the negative
log-likelihood near a minimum, is computable from first-order gradients alone (cheap on large nets), and is
positive semidefinite (a valid convex penalty). It is estimated by passing examples from the just-finished
context through the trained network in eval mode and, for each, summing the squared gradient of `-log p(c|x)`
over classes `c` weighted by the model's predicted probability `p_theta(c|x)` (the "all labels" / true
Fisher; the empirical Fisher would use only the observed label), averaged over a capped number of examples.
Each context's Fisher may be **normalised** before accumulation so deterministic contexts (large-norm Fisher)
and soft contexts (small-norm Fisher) are protected equally rather than by Fisher scale.

## Defaults and why

- `gamma = 0.9` in the hook below: keeps roughly the last ~10 contexts strongly represented in the stored
  summary while fading older ones, balancing retention against capacity for new contexts.
- Fisher estimated from a capped number of single-example passes for speed; the hook below uses 200.

## Relation to prior methods

- **EWC** (Kirkpatrick et al. 2017) keeps one spring per past context, anchored at each `theta*_t`, so its
  memory and penalty evaluation are linear in the number of contexts. Setting `gamma = 1` in the single-spring
  form recovers the strict undecayed summed-Fisher version, not the original multi-anchor bookkeeping.
- **Huszár 2017** supplies the consistent single-penalty (undecayed) form; online EWC adds the explicit
  forgetting `gamma` and the stochastic-EP recurrence handling on top.
- **Synaptic Intelligence** (Zenke et al. 2017) also uses a single accumulated importance and a latest-optimum
  anchor, but estimates importance as a trajectory path-integral rather than a local Fisher, and has no
  built-in cross-context re-weighting.

## Working code

Fills the two slots of the continual-learning regularisation harness — the per-context importance estimator
and the per-step penalty.
`estimate_importance` returns the *increment* relative to the stored importance so the loop's additive
accumulator lands exactly on `F* = gamma * F*_old + F_new`.

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def estimate_importance(model, dataset, prev_params, device):
    """Online EWC: diagonal Fisher with exponential-decay accumulation across contexts.

    Returns {param_name: increment} so the loop's additive accumulator ends up
    holding  F* = gamma * F*_old + F_new.
    """
    model.gamma = 0.9          # gamma < 1: EP partial-removal fraction == graceful-forgetting knob
    gamma = model.gamma        # gamma = 1 gives the strict undecayed single-penalty sum

    est_fisher = {}
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher[n] = p.detach().clone().zero_()

    mode = model.training
    model.eval()                                   # clean curvature estimate (no dropout noise)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    max_samples = min(len(data_loader), 200)
    n_samples = 0

    for idx, (x, y) in enumerate(data_loader):
        if idx >= max_samples:
            break
        x = x.to(device)
        output = model(x)
        with torch.no_grad():
            label_weights = F.softmax(output, dim=1)        # p_theta(y=c|x): inner-expectation weights
        for c in range(output.shape[1]):
            label = torch.LongTensor([c]).to(device)
            negloglikelihood = F.cross_entropy(output, label)
            model.zero_grad()
            negloglikelihood.backward(
                retain_graph=True if (c + 1) < output.shape[1] else False)
            for gen_params in model.param_list:
                for n, p in gen_params():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            est_fisher[n] += label_weights[0][c] * (p.grad.detach() ** 2)
        n_samples += 1

    est_fisher = {n: v / max(n_samples, 1) for n, v in est_fisher.items()}    # average over examples

    # F* = gamma * F*_old + F_new
    existing = getattr(model, '_custom_importance', {})
    for n in est_fisher:
        if n in existing:
            est_fisher[n] = gamma * existing[n] + est_fisher[n]

    # loop accumulates additively -> return increment to land on F* exactly
    result = {}
    for n in est_fisher:
        result[n] = est_fisher[n] - existing[n] if n in existing else est_fisher[n]

    model.train(mode=mode)
    return result


def compute_regularization_loss(model, importance_dict, prev_params_dict):
    """Online EWC penalty:  0.5 * gamma * sum( F* * (theta - theta*)^2 )."""
    gamma = getattr(model, 'gamma', 0.9)
    losses = []
    for gen_params in model.param_list:
        for n, p in gen_params():
            if p.requires_grad:
                n = n.replace('.', '__')
                if n in importance_dict and n in prev_params_dict:
                    fisher = importance_dict[n]               # F* (running, decayed sum)
                    prev = prev_params_dict[n]                # theta*_{i-1}: latest-boundary snapshot
                    losses.append((fisher * (p - prev) ** 2).sum())
    if losses:
        return 0.5 * gamma * sum(losses)
    return torch.tensor(0.0, device=next(model.parameters()).device)
```
