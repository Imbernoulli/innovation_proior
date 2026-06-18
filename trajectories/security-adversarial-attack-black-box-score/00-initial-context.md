## Research question

A trained image classifier `f: [0,1]^d -> R^K` is exposed only as an inference API: I send it an
image and it returns the per-class logits `f(x)`, predicting `argmax_k f_k(x)`. I have **no** access to
its weights and **no** gradients — I can only query forward. For a correctly classified `(x, y)` I want
a nearby image the model gets wrong, formally `argmax_k f_k(x_adv) != y` with
`||x_adv - x||_inf <= eps` and `x_adv in [0,1]^d`. The single thing being designed is the **attack
algorithm** — the search procedure inside `run_attack` that, using only forward queries, drives the
correct-class score down until the prediction flips, while staying inside the `L_inf` box. Everything
else — the query-counting oracle, the budget bookkeeping, the validity checks — is fixed.

The objective is not "can an adversarial example be found" (with unlimited queries it always can) but
"find one in as few forward passes as possible, on as many images as possible." Every call to the
oracle costs queries, and there is a hard per-sample budget `n_queries`. So the two numbers that matter
are **attack success rate** `asr` (primary, higher is better) and **average queries** `avg_queries`
(tie-break, lower is better).

## Prior art before the first rung (black-box search lineage)

The first rung reacts to the established families of score-based black-box attacks. These are the
methods the ladder climbs past; the fixed substrate below is the harness they all plug into.

- **White-box PGD (Madry et al. 2018).** Take projected gradient-ascent steps on the margin loss and
  clip back into the `L_inf` ball. The reference for attack *strength*, but it needs the model's
  gradient — unavailable here by assumption. Gap: assumes gradient access.
- **NES / finite-difference gradient estimation (Ilyas et al. 2018).** Rebuild the gradient from score
  queries — sample random directions `u_i`, estimate `grad ~ (1/sigma) E[L(x+sigma u_i) u_i]`, then run
  PGD on the estimate. Gap: the estimator's variance grows with the input dimension `d`, so each step
  costs many queries and whole attacks run into the tens of thousands; and because it follows the
  *local gradient*, it is exactly the family defeated by gradient masking.
- **SimBA — orthonormal-basis random search (Guo et al. 2019).** Maintain a perturbation; each step
  pick a direction from a fixed orthonormal basis (pixel or DCT), try `+alpha`/`-alpha` along it, keep
  whichever lowers the loss. Forward-only and masking-immune. Gap: each accepted move is a small `L2`
  step (slow), and because basis directions are orthogonal, a move that later proves wrong **cannot be
  undone** — budget committed to a region is unrecoverable.
- **Discrete corner / sign-search attacks (Moon et al. 2019; Al-Dujaili & O'Reilly 2019).** Exploit
  that successful `L_inf` perturbations sit at corners (`+/- eps` componentwise): restrict to the
  discrete cube and run a combinatorial / sign-flip search over a coarse a-priori grid. Gap: the grid
  fixing *where* changes may happen is frozen in advance and only coarsely refined — the search cannot
  freely choose where to spend budget.

## The fixed substrate

The harness in `torchattacks/bench/run_eval.py` is frozen and not editable. It wraps the clean model in
a `QueryLimitedBlackBox` oracle: every call to `model(x)` consumes `x.shape[0]` queries; once the
running count exceeds `batch_size * n_queries` the oracle flips `budget_exhausted` and returns **zeros**
for the rest of the batch, and the **entire batch is scored as a failure**. The wrapper exposes logits
only and carries no gradient path (`torch.no_grad`). After the attack returns, the harness checks each
sample's `L_inf` deviation (`<= eps + 1e-6`) and `[0,1]` range; any sample that violates either is
marked a failure. It evaluates on the subset of test images the clean model already classifies
correctly, so `clean_acc = 1.0` by construction, and reports
`ATTACK_METRICS asr=... clean_acc=... robust_acc=... avg_queries=...` with `robust_acc = 1 - asr`.

The practical consequences the attack must respect: (i) one forward pass on a batch of `B` candidates
costs `B` queries, so batching candidates across samples is the same total budget as looping; (ii)
overshooting the budget on *any* batch loses that whole batch — the loop must stop querying before it
runs out; (iii) the returned tensor must already be projected into the `L_inf` box and `[0,1]`, or the
sample is counted as a miss regardless of whether it actually fooled the model.

## The editable interface

Exactly one region is editable — the body of `run_attack` in `torchattacks/bench/custom_attack.py`
(lines 7-56). Every method on the ladder is a fill of this same contract:

`run_attack(model, images, labels, eps, n_queries, device, n_classes) -> adv_images`

where `model` is the query-only oracle, `images` is `(N,C,H,W)` in `[0,1]`, `labels` is `(N,)`, `eps`
is the `L_inf` budget, `n_queries` is the per-sample budget, and the return is the adversarial batch
with the same shape, values in `[0,1]`. The starting point is the scaffold default: a minimal
score-based search — propose uniform noise inside the `L_inf` box, query, and greedily keep the
candidate when the correct-class logit drops. Each method on the ladder replaces exactly this body.

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
CIFAR-100}, using publicly available pretrained checkpoints, on 200 correctly-classified test images at
batch size 50, seed 42, `eps = 8/255`, and a per-sample budget `n_queries`. Untargeted attacks. The
primary metric is `asr` (higher is better) across scenarios; `avg_queries` (lower is better) is the
tie-break.
