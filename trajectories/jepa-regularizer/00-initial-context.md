## Research question

Joint-embedding self-supervised learning on images trains an encoder so that two augmented views of
one image land at the same place in embedding space. That invariance objective, written by itself,
admits a trivial global minimum: map every input to one constant vector and every pair of views agrees
perfectly. **Collapse.** The single thing being designed here is the **anti-collapse regularizer** — a
loss term, computed from a batch of projected embeddings, that forbids that constant solution while
shaping the embedding geometry so the frozen representation transfers well to a downstream linear
probe. Everything else about the pipeline (encoder, projector, optimizer, augmentations, the linear
probe and how it is read out) is fixed substrate.

## Prior art before the first rung (joint-embedding lineage)

The first rung — the MSE-only invariance loss — is the *bare* objective the whole field reacts to. It
is the resolution of nothing; it is the thing every later regularizer exists to fix. The lineage that
frames it:

- **Siamese metric learning (Bromley et al. 1993; Hadsell, Chopra, LeCun 2006, DrLIM).** A shared-weight
  twin network maps two inputs to a common embedding and a *pair loss* shapes the output distance: pull
  labelled-similar pairs together, push labelled-dissimilar pairs apart up to a margin. Two distorted
  views of one image are exactly a "similar" pair. **Gap:** the attract-only half — keep the similar
  term, drop the margin repulsion — has the constant map as a global minimizer. That attract-only half
  *is* the MSE invariance term, so on its own it collapses. The repulsion is what the bare objective
  throws away.
- **Contrastive SSL (InfoNCE / SimCLR / MoCo, 2018-2020).** Restore the repulsion as a softmax over
  in-batch negatives: pull the positive pair together, push every other image away. Collapse dies
  (a constant maximizes every negative similarity). **Gap:** the repulsion is a non-parametric estimate
  of embedding spread from pairwise sample distances — sample-hungry, so it needs huge batches or a
  memory queue, and it does not improve with embedding width.
- **Asymmetric SSL (BYOL / SimSiam, 2020).** Drop negatives; break the symmetry of the two branches
  with a predictor MLP and a stop-gradient (BYOL adds an EMA target). It does not collapse, but there
  is no single scalar being minimized — non-collapse is a dynamics accident, and the branches are
  welded together. **Gap:** no clean objective, branches coupled.
- **Information-maximization SSL (Barlow Twins, Whitening, 2021).** State what a good representation
  *is* — decorrelated, non-redundant dimensions — and make non-collapse a *consequence* of the
  objective rather than a bolt-on. **Gap (the open question this task picks up):** which exact statistic
  on the embeddings to penalize, and how much, so that collapse is impossible at the optimum while the
  geometry transfers across backbone scales.

## The fixed substrate

A self-contained CIFAR-10 joint-embedding loop is frozen and must not be touched. A backbone (one of
ResNet-18 / ResNet-34 / ResNet-50, each modified for CIFAR-10: a 3×3 stride-1 `conv1`, no maxpool, `fc`
replaced by identity) feeds a **projector** — `features_dim → proj_hidden_dim → proj_hidden_dim →
proj_output_dim` MLP, BatchNorm+ReLU on the hidden layers, plain Linear at the end (default
`2048 → 2048`). Two augmented views per image (`RandomResizedCrop(32, scale 0.2-1.0)`, color jitter,
grayscale, a light solarize, horizontal flip, normalize) go through backbone+projector to give two
projected embedding batches `z1, z2`, each `[B, D]`. The loop trains with **LARS** (`lr=0.3`,
`eta=0.02`, `clip_lr=True`, momentum 0.9, weight decay 1e-4, bias/norm excluded), a warmup-cosine
schedule (10 warmup epochs), batch size 256, AMP bf16. An **online linear probe** is trained jointly on
the *frozen detached features* every step; its CIFAR-10 validation accuracy `val_acc` is the metric.
The harness also exposes a `CONFIG_OVERRIDES` dict with a whitelist — only `proj_output_dim` and
`proj_hidden_dim` may be set — so a method may reshape the projector but nothing else.

## The editable interface

Exactly one region is editable: the `CustomRegularizer` class plus the `CONFIG_OVERRIDES` dict in
`custom_regularizer.py` (lines 33-58). The contract is one method:

- `forward(z1, z2) -> dict` — `z1, z2` are the two views' projected embeddings, each `[B, D]`; return a
  dict with at least a scalar `"loss"` key (extra keys are logged). Any `__init__` parameters and helper
  methods are allowed; `torch`, `torch.nn` as `nn`, `torch.nn.functional` as `F` are imported.

Every method on the ladder is a fill of this same contract. The starting point is the scaffold default:
a zero loss (no anti-collapse, no invariance) — a true placeholder. Each later method replaces exactly
this class (and optionally `CONFIG_OVERRIDES`) and nothing else.

```python
# EDITABLE region of custom_regularizer.py (lines 33-58) -- default fill (placeholder)
class CustomRegularizer(nn.Module):
    """Anti-collapse regularizer for self-supervised JEPA learning.

    Takes two projected embedding tensors from different augmented views
    and returns a loss dict that prevents representation collapse while
    encouraging useful feature learning.

    Args:
        z1: [B, D] projected embeddings from view 1
        z2: [B, D] projected embeddings from view 2

    Returns:
        dict with at least a "loss" key (scalar tensor)
    """

    def __init__(self):
        super().__init__()

    def forward(self, z1, z2):
        loss = torch.tensor(0.0, device=z1.device, requires_grad=True)
        return {"loss": loss}


# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: proj_output_dim, proj_hidden_dim.
CONFIG_OVERRIDES = {}
```

## Evaluation settings

- **Metric:** `val_acc` — linear-probe top-1 accuracy on the CIFAR-10 validation set (10k images),
  higher is better.
- **Benchmarks:** three backbones — **ResNet-18**, **ResNet-34**, **ResNet-50** — run the *same*
  regularizer, to test whether it generalizes across model scale (the projector input width is 512 for
  the two smaller backbones and 2048 for ResNet-50).
- **Projector:** `features_dim → 2048 → 2048` by default; a method may reshape `proj_output_dim` /
  `proj_hidden_dim` through `CONFIG_OVERRIDES`.
- **Training:** the loop pretrains for the full upstream schedule (the script's `main()` sets the long
  schedule the SSL baselines need to converge and rank reliably), batch size 256, LARS (`lr=0.3`,
  `eta=0.02`, `clip_lr=True`), warmup-cosine, single seed 42.
- **Dataset:** CIFAR-10, 50k train / 10k val.
