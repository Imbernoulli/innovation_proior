**Problem.** The floor (`random_search`) read `asr = 0.474` at only 65 of 1000 queries — it quit early
and its isotropic interior nudges were almost orthogonal to any descent direction, collapsing on the
harder VGG11-BN pairs. The fix: spend the real budget and reconstruct an actual gradient of the margin
from forward queries alone.

**Key idea (SPSA: simultaneous-perturbation gradient estimation + Adam).** Descend the margin
`J(x) = f_y - max_{k!=y} f_k` (it keeps slope where cross-entropy saturates). Estimate its gradient from
two queries regardless of dimension: perturb *all* coordinates at once by a single Rademacher vector
`v in {+/-1}^D`, take the two-sided difference `df = J(x+cv) - J(x-cv)`, and read the gradient as
`ghat = (df / 2c) . v`. Rademacher is *forced* by the finite-inverse-moment condition (Gaussian's
`E[1/v_i]` diverges); `v_i = +/-1` makes division equal multiplication. Average `n` independent estimates
per step to kill cross-talk and query noise (variance `~1/n`), feed the result to **Adam** (per-
coordinate rescaling handles the noisy, unevenly-scaled estimate), and project back into the `L_inf` box
and `[0,1]` after each step.

**Why it beats the floor.** It descends a genuine, masking-immune estimated direction instead of guessing
one, and it spends the whole budget: a step costs `2*nb_sample` queries, and `nb_iter = n_queries //
(2*nb_sample)` uses ~768 of 1000 (3 steps at 256). The cost: only a handful of steps fit, so
`avg_queries` jumps from 65 to ~768 — the SPSA bargain (a real direction, but an expensive one).

**Hyperparameters.** `delta (probe c) = 0.01` (inside the `eps` ball, above the noise floor);
`lr = 0.01`; `nb_sample = 128`; `max_batch_size = 64`; `nb_iter = max(1, n_queries // (2*nb_sample))`.

**What to watch.** Mean `asr` 0.474 -> ~0.8s; the VGG11-BN pairs (the floor's worst, 0.255/0.300) should
roughly double if their failure was directional; ResNet20 pairs saturate toward the low-0.9s; the headline
regression is `avg_queries` 65 -> ~768.

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
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import torchattacks

    _ = (device, n_classes)
    model.eval()

    nb_sample = 128
    nb_iter = max(1, int(n_queries) // (2 * nb_sample))

    attack = torchattacks.SPSA(
        model=model,
        eps=eps,
        delta=0.01,
        lr=0.01,
        nb_iter=nb_iter,
        nb_sample=nb_sample,
        max_batch_size=64,
    )
    adv_images = attack(images, labels)
    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
```
