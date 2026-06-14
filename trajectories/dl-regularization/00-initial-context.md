## Research question

A deep convolutional image classifier has far more parameters than the data can constrain, so it
overfits: training accuracy marches toward perfect while held-out accuracy stalls. The single thing
being designed is an **additional regularization term** — one differentiable scalar, computed every
training step from the model, the batch, and the current logits, and *added* to the cross-entropy loss
before backpropagation. Everything else about the pipeline is frozen: the architectures, the weight
initialization, the data and augmentation, the optimizer (SGD, `lr=0.1`, `momentum=0.9`,
`weight_decay=5e-4` — so plain L2 weight decay is *already on*), the cosine schedule, and the
evaluation procedure. The term must generalize across three architecture/dataset pairs, must stay
differentiable, and must be cheap enough to run on every step.

## Prior art before the first rung

The first rung reacts to the standard regularizers that already act *inside* the network, and to the
fact that the most successful one stops working in exactly the place this task cares about — the
convolutional layers.

- **Weight decay / L2 (Krogh & Hertz 1991).** Penalize `||W||^2`, shrinking every weight toward zero.
  It controls overall magnitude (the *scale* of the spectrum) but not its *shape*, and it is already
  applied here through the optimizer (`weight_decay=5e-4`), so any new term must do something L2 does
  not. Gap: scale-only, and already spent.
- **Dropout (Srivastava et al. 2014).** Zero each hidden unit independently with probability `1-p` and
  rescale survivors, so one forward pass samples a thinned sub-network and test time averages the
  ensemble — killing co-adaptation in dense layers. Gap: on a convolutional feature map the activations
  are spatially correlated, so zeroing one scalar removes nothing (its neighbors carry the same signal
  onward); dropout has quietly disappeared from conv layers.
- **Batch normalization (Ioffe & Szegedy 2015).** Normalize activations per channel; a strong implicit
  regularizer, and present in all three architectures here. Gap: it normalizes *activations*, leaving
  the weight spectrum and the *output distribution* untouched.
- **Label smoothing (Szegedy et al. 2016).** Soften the one-hot target toward uniform, which curbs
  over-confident outputs. Gap: it acts by *replacing the target / the loss itself*, which this edit
  surface forbids — the base cross-entropy is fixed; a regularizer here can only *add* a term.

These act on weights (L2), hidden units (dropout), or activations (BN); none is an *additional*,
adaptive, loss-side penalty layered on a pipeline that already has L2 and BN. That is the gap the ladder
fills, each rung a different answer to "what extra scalar should I add to the loss?".

## The fixed substrate

A standard CIFAR-style training loop is frozen and must not be touched. CIFAR-adapted architectures
(ResNet-56 with `BasicBlock` `[9,9,9]`, VGG-16-BN with adaptive average pooling, MobileNetV2 with a
stride-1 stem), Kaiming weight init, `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`, per-dataset
normalization, batch size 128. The optimizer is `SGD(lr=0.1, momentum=0.9, weight_decay=5e-4)` with
`CosineAnnealingLR(T_max=200)` over 200 epochs. The training step is exactly:

`loss = CrossEntropy(outputs, targets) + compute_regularization(model, inputs, outputs, targets, config)`

so the new term is summed straight onto the cross-entropy. Best test accuracy over training is recorded.

## The editable interface

Exactly one region is editable — the body of `compute_regularization` in `pytorch-vision/custom_reg.py`
(the file's lines 246-273). `torch`, `torch.nn as nn`, and `torch.nn.functional as F` are already
imported at module scope, so the body uses them directly (no imports inside the function). The contract:

- `model`: the full `nn.Module`; iterate `model.named_parameters()` / `model.named_modules()` for
  weight- or activation-based penalties.
- `inputs`: `[B, 3, 32, 32]` input batch (for input-dependent penalties).
- `outputs`: `[B, num_classes]` logits (for output-based penalties such as entropy/confidence).
- `targets`: `[B]` integer labels.
- `config`: dict with `num_classes` (int), `epoch` (int, 0-indexed), `total_epochs` (int) — for
  epoch-dependent scheduling.

The function returns one differentiable scalar `torch.Tensor` added to the cross-entropy loss; return
`0` for no regularization. The starting point is the scaffold default: not implemented. Each method on
the ladder replaces exactly this body and nothing else.

```python
# EDITABLE region of pytorch-vision/custom_reg.py (lines 246-273) — default fill
# torch, torch.nn as nn, torch.nn.functional as F are already imported at module scope.
def compute_regularization(model, inputs, outputs, targets, config):
    """Compute regularization loss term added to classification loss.

    Called every training step. The returned scalar is added to the
    cross-entropy loss before backpropagation.

    Args:
        model: the network (nn.Module, ResNet / VGG / MobileNetV2)
        inputs: [B, 3, 32, 32] input batch
        outputs: [B, num_classes] model logits
        targets: [B] integer class labels
        config: dict with keys:
            - 'num_classes': int (10 or 100)
            - 'epoch': int (current epoch, 0-indexed)
            - 'total_epochs': int (total training epochs)

    Returns:
        scalar torch.Tensor -- regularization loss (added to CE loss).
        Return 0 for no regularization.

    Design considerations:
        - Weight-based penalties (L1, L2, orthogonality, spectral)
        - Output-based penalties (confidence, entropy, label smoothing)
        - Activation-based penalties (sparsity, diversity)
        - Gradient-based penalties (gradient penalty, Jacobian)
        - Epoch-dependent scheduling (warm-up, annealing)
    """
    raise NotImplementedError("Implement your regularization strategy")
```

## Evaluation settings

Three architecture/dataset pairs spanning the difficulty range — **ResNet-56 on CIFAR-100** (deep,
residual, 100 classes), **VGG-16-BN on CIFAR-100** (plain, BN-heavy, large dense classifier head), and
**MobileNetV2 on FashionMNIST** (depthwise-separable, grayscale-replicated-to-3-channel, 10 classes,
the held-out pair). One seed, `42`. One metric, higher is better: **best test accuracy (%)** reached at
any epoch during the 200-epoch run. The regularizer must not alter the dataset, architecture, base
loss, optimizer, scheduler, or evaluation procedure.
