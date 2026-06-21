## Research question

Joint-embedding self-supervised learning trains an encoder so that two augmented views of one image map to the same embedding. That invariance objective alone has a trivial global minimum: the constant map. The design task here is the **anti-collapse regularizer** — a scalar loss term computed from a batch of projected embeddings that forbids the constant solution while preserving the geometry that transfers to a downstream linear probe. Everything else in the pipeline is fixed substrate.

## Prior art / Background / Baselines

- **Siamese metric learning (DrLIM and earlier work).** A shared-weight twin network maps two inputs to a common embedding and a pair loss pulls similar pairs together while pushing dissimilar pairs apart up to a margin.
- **Contrastive SSL (InfoNCE, SimCLR, MoCo).** Repulsion is restored by treating in-batch negatives as dissimilar pairs in a softmax over pairwise similarities.
- **Asymmetric SSL (BYOL, SimSiam).** Negatives are dropped and collapse is avoided through an asymmetric pair of branches — a predictor plus stop-gradient, with an EMA target in BYOL.
- **Information-maximization SSL (Barlow Twins, whitening methods).** Non-collapse is made a consequence of a batch-level representational criterion rather than pairwise repulsion.

## Fixed substrate / Code framework

A self-contained CIFAR-10 joint-embedding loop is frozen. A ResNet backbone (18 / 34 / 50, modified for CIFAR-10) feeds a projector MLP `features_dim -> proj_hidden_dim -> proj_hidden_dim -> proj_output_dim` (default `2048 -> 2048`), with BatchNorm+ReLU hidden layers and a plain linear output. Two augmented views per image are produced by `RandomResizedCrop(32, scale=0.2-1.0)`, color jitter, grayscale, light solarize, horizontal flip, and normalization. Training uses LARS (`lr=0.3`, `eta=0.02`, `clip_lr=True`, momentum 0.9, weight decay 1e-4, bias/norm excluded), warmup-cosine schedule (10 warmup epochs), batch size 256, AMP bf16. An online linear probe is trained jointly on frozen detached features; its CIFAR-10 validation accuracy `val_acc` is the metric. The harness exposes `CONFIG_OVERRIDES`, but only `proj_output_dim` and `proj_hidden_dim` may be set.

## Editable interface

Only the `CustomRegularizer` class and the `CONFIG_OVERRIDES` dict in `custom_regularizer.py` (lines 33-58) may be edited. The contract is:

- `forward(z1, z2) -> dict` — `z1, z2` are projected embeddings, each `[B, D]`; return a dict with at least a scalar `"loss"` key. Any `__init__` parameters and helpers are allowed; `torch`, `torch.nn` as `nn`, and `torch.nn.functional` as `F` are imported.

The default scaffold is a zero-loss placeholder. Each method replaces exactly this class and optionally `CONFIG_OVERRIDES`.

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

- **Metric:** `val_acc` — linear-probe top-1 accuracy on the CIFAR-10 validation set (10k images), higher is better.
- **Benchmarks:** the same regularizer is run with ResNet-18, ResNet-34, and ResNet-50 to test generalization across model scale (projector input width is 512 for the first two and 2048 for ResNet-50).
- **Projector:** `features_dim -> 2048 -> 2048` by default; a method may reshape `proj_output_dim` / `proj_hidden_dim` via `CONFIG_OVERRIDES`.
- **Training:** full upstream pretraining schedule, batch size 256, LARS (`lr=0.3`, `eta=0.02`, `clip_lr=True`), warmup-cosine, single seed 42.
- **Dataset:** CIFAR-10, 50k train / 10k val.
