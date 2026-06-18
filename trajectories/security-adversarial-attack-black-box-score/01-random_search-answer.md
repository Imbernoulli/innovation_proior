**Problem.** Score-based black-box `L_inf` attack: for a correctly classified `(x, y)`, drive the
correct-class logit down until the prediction flips, using only forward queries, while staying in the
`L_inf` box and `[0,1]`. Primary metric `asr`; tie-break `avg_queries` (lower better). This rung is the
floor — the crudest use of the query oracle.

**Key idea (greedy uniform random search).** Hill-climb the correct-class score `f_y` by accept-if-
better: each step propose isotropic uniform noise in a small box `uniform(-eps/2, eps/2)` around the
current best, re-project into the `L_inf` ball and `[0,1]`, query, and keep the candidate per-sample
only where the correct-class logit dropped. No structure (the model is treated as a generic function,
not a CNN), no corner bias (moves land inside the box, not at `+/- eps`), no step decay.

**Why it is the floor.** Two deliberate weaknesses set the number: (i) it caps itself at
`min(n_queries, 64)` candidate queries, so with a 1000 budget it uses ~65 queries and leaves ~935
unused; (ii) isotropic interior noise in a ~3000-dim space is almost orthogonal to any useful descent
direction, so most proposals are rejected and accepted ones crawl. Easy (model, dataset) pairs flip a
fair fraction; harder/CIFAR-100 pairs sag well below half.

**Hyperparameters.** `step = eps/2`; `n_steps = max(1, min(n_queries, 64))`; one initial clean query +
one candidate query per step; per-sample masked accept on `cand_score < best`.

**What to watch.** A wide spread across scenarios and a modest mean `asr` at a tiny `avg_queries`. The
self-imposed query cap and the unstructured interior moves are what step 2 attacks: spend the real
budget and reconstruct an actual descent direction from forward queries.

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    n_queries: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    _ = (device, n_classes)
    model.eval()

    adv_images = images.detach().clone()
    step = eps / 2.0
    n_steps = max(1, min(int(n_queries), 64))

    with torch.no_grad():
        best = model(adv_images).gather(1, labels.view(-1, 1)).squeeze(1)

        for _ in range(n_steps):
            noise = torch.empty_like(adv_images).uniform_(-step, step)
            cand = adv_images + noise
            cand = torch.clamp(images + torch.clamp(cand - images, -eps, eps), 0.0, 1.0)

            cand_score = model(cand).gather(1, labels.view(-1, 1)).squeeze(1)
            improve = cand_score < best

            if improve.any():
                mask = improve.view(-1, 1, 1, 1)
                adv_images = torch.where(mask, cand, adv_images)
                best = torch.where(improve, cand_score, best)

    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
```
