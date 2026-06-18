**Problem.** SPSA reached `asr = 0.816` but paid a flat `avg_queries = 768` everywhere — it spends 256
queries per descent step and so gets only ~3 steps, leaving the hard VGG11-BN boundaries step-starved at
~0.61. The fix: many cheap, *structured* one-query moves instead of a few expensive estimated steps, and
no gradient at all (so it is also immune to gradient masking).

**Key idea (Square Attack: greedy random search with structured square updates).** Hill-climb the margin
`J = f_y - max_{k!=y} f_k` with one query per candidate, designing the proposal to carry the structure
the floor lacked:
- **Corners, not interiors.** Updates of value `+/-2eps` re-project onto a box corner — the full per-
  component budget is spent every step (the floor wasted half with `eps/2`), and later squares can
  overwrite earlier regions.
- **Square support.** For a fixed pixel budget, the square maximizes the number of first-layer
  convolutional receptive fields fully covered, i.e. the worst-case change in first-layer activations.
  Position sampled freely each step (no frozen grid).
- **One sign per channel, shared spatially.** Against a piecewise-constant gradient, a shared sign
  correlates at the `||v||_1` scale vs `||v||_2` for independent signs — a factor-`h` better alignment.
- **Coarse-to-fine size.** `p` (fraction of pixels) halves on a schedule rescaled to the budget — the
  step-size decay Adam gave SPSA, built into the move size.
- **Vertical-stripe boundary init**; force-accept a flip (`margin <= 0`); stop querying a flipped sample.

**Why it wins both axes.** One query per step buys hundreds of structured moves where SPSA got three, so
`asr` rises on the hard boundaries; and because each step is one query and flipped samples exit early,
`avg_queries` falls *far below* the budget — the opposite of SPSA's flat 768.

**Hyperparameters.** `norm = "Linf"`, `p_init = 0.8`, `loss = "margin"`, `n_restarts = 1`,
`resc_schedule = True`, `n_queries = max(1, n_queries)`, `seed = SEED`.

**What to watch.** Mean `asr` 0.816 -> mid-0.9s; the VGG11-BN pairs (SPSA's worst) lift into high-0.8/
low-0.9; ResNet20 pairs near-saturate ~0.99; `avg_queries` inverts from a flat 768 to ~70-260.

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

    attack = torchattacks.Square(
        model=model,
        norm="Linf",
        eps=eps,
        n_queries=max(1, int(n_queries)),
        n_restarts=1,
        p_init=0.8,
        seed=int(os.environ.get("SEED", "42")),
        verbose=False,
        loss="margin",
        resc_schedule=True,
    )
    adv_images = attack(images, labels)
    delta = torch.clamp(adv_images - images, min=-eps, max=eps)
    return torch.clamp(images + delta, 0.0, 1.0).detach()
```
