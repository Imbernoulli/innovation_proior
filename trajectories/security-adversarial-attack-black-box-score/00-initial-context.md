## Research question

A trained image classifier `f: [0,1]^d -> R^K` is exposed only as a score API: I send it an image
and it returns the per-class logits `f(x)`, predicting `argmax_k f_k(x)`. I have **no** access to its
weights and **no** gradients — only forward queries. For a correctly classified `(x, y)` I want a
nearby image the model gets wrong, i.e. `argmax_k f_k(x_adv) != y` with `||x_adv - x||_inf <= eps`
and `x_adv in [0,1]^d`.

The design target is the **attack algorithm** itself: the search procedure inside `run_attack` that,
using only forward queries, drives the correct-class score down until the prediction flips while
staying inside the `L_inf` box. The query-counting oracle, budget bookkeeping, and validity checks
are fixed. With unlimited queries an adversarial example always exists, so the objective is to find
one in as few forward passes as possible on as many images as possible. The two reported metrics are
**attack success rate** `asr` (primary, higher is better) and **average queries** `avg_queries`
(tie-break, lower is better).

## Prior art / Background / Baselines

The relevant score-based black-box attacks are the current points of comparison.

- **White-box PGD (Madry et al. 2018).** Core idea: run projected gradient-ascent steps on the margin
  loss, clipping back into the `L_inf` ball. Gap: it requires the model's gradient, which the
  black-box assumption withholds.
- **NES / finite-difference gradient estimation (Ilyas et al. 2018).** Core idea: rebuild a local
  gradient from score queries by sampling random directions, then run PGD on the estimate. Gap:
  estimator variance grows with input dimension, so each step costs many queries and full attacks
  often need tens of thousands of queries; following the local gradient also makes the method
  brittle against gradient-masking defenses.
- **SimBA — orthonormal-basis random search (Guo et al. 2019).** Core idea: greedily add or subtract a
  fixed step along a fixed orthonormal basis (pixel or DCT) whenever it lowers the true-class logit.
  Gap: accepted moves are small `L2` steps, and basis directions cannot be undone later in the
  opposite sign, so budget committed to a wrong region is unrecoverable.
- **Discrete corner / sign-search attacks (Moon et al. 2019; Al-Dujaili & O'Reilly 2019).** Core idea:
  restrict perturbations to the corners of the `L_inf` cube and search over a pre-defined discrete
  grid of sign flips. Gap: the coordinate grid is fixed ahead of time and only coarsely refined, so
  the search has little freedom to decide where to spend its query budget.

## Fixed substrate / Code framework

The harness in `torchattacks/bench/run_eval.py` is frozen. It wraps the clean model in a
`QueryLimitedBlackBox` oracle: every call to `model(x)` consumes `x.shape[0]` queries; once the
running count exceeds `batch_size * n_queries` the oracle flips `budget_exhausted` and returns
**zeros** for the rest of the batch, and the **entire batch is scored as a failure**. The wrapper
exposes logits only and carries no gradient path (`torch.no_grad`). After the attack returns, the
harness checks `L_inf` deviation (`<= eps + 1e-6`) and the `[0,1]` range; any violation marks that
sample as a failure. It evaluates on the subset of test images the clean model already classifies
correctly, so `clean_acc = 1.0` by construction, and reports
`ATTACK_METRICS asr=... clean_acc=... robust_acc=... avg_queries=...` with `robust_acc = 1 - asr`.

Practical consequences: (i) one forward pass on a batch of `B` candidates costs `B` queries; (ii)
overshooting the budget on any batch loses the whole batch, so the loop must stop before it runs out;
(iii) the returned tensor must already be projected into the `L_inf` box and `[0,1]`, or the sample
is counted as a miss regardless of whether it fooled the model.

## Editable interface

Exactly one region is editable — the body of `run_attack` in `torchattacks/bench/custom_attack.py`.
Every method fills the same contract:

`run_attack(model, images, labels, eps, n_queries, device, n_classes) -> adv_images`

where `model` is the query-only oracle, `images` is `(N,C,H,W)` in `[0,1]`, `labels` is `(N,)`, `eps`
is the `L_inf` budget, `n_queries` is the per-sample budget, and the return is the adversarial batch
with the same shape and values in `[0,1]`. The starting scaffold is a minimal score-based search:
propose uniform noise inside the `L_inf` box, query it, and greedily keep the candidate when the
correct-class logit drops.

```python
import torch
import torch.nn as nn

# =====================================================================
# EDITABLE: implement run_attack below
# =====================================================================
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_queries: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """
    Score-based query black-box attack under Linf constraint.

    model: black-box wrapper that only exposes forward logits.
    images: (N, C, H, W) in [0, 1], on device.   labels: (N,), on device.
    eps: Linf budget.   n_queries: per-sample query budget.
    """
    _ = (device, n_classes)
    model.eval()

    # A simple default that already performs score-based search.
    # Baselines will replace this block with stronger algorithms.
    adv = images.detach().clone()
    step = eps / 4.0
    iters = max(1, min(int(n_queries), 16))

    with torch.no_grad():
        for _ in range(iters):
            logits_old = model(adv)
            true_old = logits_old.gather(1, labels.view(-1, 1)).squeeze(1)

            noise = torch.empty_like(adv).uniform_(-step, step)
            cand = adv + noise
            cand = torch.clamp(images + torch.clamp(cand - images, -eps, eps), 0.0, 1.0)

            logits_new = model(cand)
            true_new = logits_new.gather(1, labels.view(-1, 1)).squeeze(1)
            improve = true_new < true_old

            if improve.any():
                mask = improve.view(-1, 1, 1, 1)
                adv = torch.where(mask, cand, adv)

    delta = torch.clamp(adv - images, min=-eps, max=eps)
    adv = torch.clamp(images + delta, 0.0, 1.0)
    return adv.detach()
# =====================================================================
# END EDITABLE REGION
# =====================================================================
```

## Evaluation settings

Each scenario is a (model, dataset) pair from {ResNet20, VGG11-BN, MobileNetV2} x {CIFAR-10,
CIFAR-100}, using publicly available pretrained checkpoints, on 200 correctly-classified test images
at batch size 50, seed 42, `eps = 8/255`, and a per-sample budget `n_queries`. Untargeted attacks.
The primary metric is `asr` (higher is better) across scenarios; `avg_queries` (lower is better) is
the tie-break.
