## Research question

CIFAR-100 ships its 100 fine classes pre-grouped into 20 coarse superclasses, so a single image
carries two labels at two granularities. The single thing being designed is the **multi-task loss
combination strategy** — how the fine-head loss and the coarse-head loss are folded into one scalar
that the optimizer descends — with one objective only: maximize **fine-class test accuracy**. The
coarse task is purely auxiliary; the coarse label is a free, semantically structured extra signal,
and the question is whether and how it can be made to *help* the fine objective rather than steal
capacity from it. Everything else — the backbone, the two heads, the data pipeline, the optimizer,
the schedule — is fixed. Only the combination rule moves.

## Prior art before the first rung

The first rung reacts to the way joint multi-task training is normally done and to its standard
failure. The lineage it answers:

- **Hard parameter sharing (Caruana, *Multitask Learning*, 1997).** A shared trunk with small
  task-specific heads, trained on the summed loss `L = Σ_i L_i`. The premise is that an inductive
  bias from a related task improves generalization. Gap: the bare sum is acutely sensitive to the
  implicit relative weighting, and "just add them" is rarely where the primary task peaks.
- **Hand-tuned / grid-searched weighting `L = Σ_i w_i L_i`.** Sweep `w_i` and keep the best. Gap:
  one full training run per grid point, combinatorial in the number of tasks, and the good band is
  narrow; the static weight also cannot adapt to where training currently is.
- **Loss-magnitude balancing (the family the auxiliary-weighting baselines come from).** Observe
  that fine cross-entropy (100-way) and coarse cross-entropy (20-way) sit at different magnitudes
  and at different difficulties, so a fixed sum lets one dominate the shared gradient; rescale per
  task to equalize them. Gap: scaling the losses changes *how much* each task counts but not the
  *geometry* of how their gradients interact on the shared trunk.
- **Gradient-interference accounts (Caruana; later multi-objective MTL).** The shared trunk sees
  `g = Σ_i g_i`; when two task gradients point partly against each other the sum cancels in the
  overlap, and joint training can land *below* single-task training. This is the observation the
  gradient-surgery rung picks up: the cancellation is *directional*, so a magnitude rescale cannot
  reach it.

## The fixed substrate

The whole pipeline is frozen and must not be touched. A CIFAR-adapted backbone (ResNet-20 = `[3,3,3]`
basic blocks, ResNet-56 = `[9,9,9]`, or VGG-16-BN) ends in a global average pool feeding **two linear
heads** off the *same* features: `fc_fine` (100-way) and `fc_coarse` (20-way). The coarse label of
each image is derived from its fine label by the fixed CIFAR-100 superclass map. Training is SGD
(`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`) under cosine annealing over **200 epochs**, batch
size 128, with `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`. Each step the loop computes
`fine_loss = CE(fine_logits, fine_targets)` and `coarse_loss = CE(coarse_logits, coarse_targets)`,
hands both (plus `epoch`, `total_epochs`) to the combination module, calls `.backward()` on the
returned scalar, and steps. One detail is load-bearing for everything that follows: the optimizer is
built over `model.parameters() + mtl_loss.parameters()`, so **any learnable tensor the module
registers as a `Parameter` is trained jointly with the network by the same SGD**. Test-time accuracy
is read off `fc_fine` only.

## The editable interface

Exactly one region is editable — the `MultiTaskLoss` class in `pytorch-vision/custom_mtl.py`
(lines 195–216). Every method on the ladder is a fill of this same contract:

- `__init__(self, num_tasks=2)` — may register learnable parameters (log-variances, weights) or
  non-learnable buffers, and may hold auxiliary state such as a loss-history buffer.
- `forward(self, fine_loss, coarse_loss, epoch, total_epochs)` — receives the **two scalar task
  losses** plus the current `epoch` (0-indexed) and `total_epochs`, and returns **one scalar** for
  `.backward()`. It must stay differentiable and must not touch labels, heads, datasets, backbones,
  or the outer loop.

What the interface does and does not expose decides the whole space. The module is handed the two
*already-reduced scalar* losses, not the logits, not the targets, not the shared features — so a
pure loss-weighting rule reads only those scalars; a gradient-geometry rule must *reach back into
the graph* from `fine_loss`/`coarse_loss` (e.g. via `torch.autograd.grad`) to recover the shared
parameters, since the module is never given them directly. The `epoch`/`total_epochs` arguments are
the only curriculum/scheduling handle.

The starting point is the scaffold default: **equal weighting** — sum the two losses.

```python
# EDITABLE region of pytorch-vision/custom_mtl.py (lines 195-216) — default fill (equal weighting)
class MultiTaskLoss(nn.Module):
    """Multi-task loss combination for fine + coarse classification.

    Args:
        num_tasks: int (always 2)
    """

    def __init__(self, num_tasks=2):
        super().__init__()

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        """Combine fine and coarse classification losses.

        Args:
            fine_loss: scalar tensor, CE loss for 100-class fine prediction
            coarse_loss: scalar tensor, CE loss for 20-class coarse prediction
            epoch: int, current epoch (0-indexed)
            total_epochs: int, total number of training epochs
        Returns:
            combined scalar loss
        """
        return fine_loss + coarse_loss
```

## Evaluation settings

Three backbones span the capacity range — **ResNet-20** (small), **ResNet-56** (deeper residual),
and **VGG-16-BN** (a different family, larger heads) — each trained on CIFAR-100 with the fine+coarse
two-head setup for 200 epochs, seed 42. The reported metric is **best fine-class test accuracy
(%, higher is better)** reached during training, per backbone; the task score is the geometric mean
across the three. The combination module must remain differentiable and may not change labels, heads,
datasets, backbones, or the outer training loop.
