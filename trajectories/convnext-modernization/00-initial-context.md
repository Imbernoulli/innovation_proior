## Research question

Vision Transformers (ViT, and hierarchical variants like Swin Transformer)
now match or beat standard ConvNets on ImageNet-1K classification at
comparable compute. Swin Transformer in particular reports 81.3% top-1 at
about 4.5x10^9 FLOPs (the "Swin-T" size class), while a standard ResNet-50
trained with its original recipe reaches only 76.1% top-1 at a similar 4.09
GFLOPs. The question: starting from ResNet-50 and moving only through a
sequence of individually testable, incremental changes to training procedure
and to standard convolutional-network building blocks (no attention, no
dynamic/content-dependent weights, no tokenization beyond a strided stem
conv), how much of the gap to Swin-T can be closed while remaining a
pure ConvNet, and does one land on some particular pure-ConvNet
architecture worth taking as an endpoint?

## Evaluation protocol (fixed for the whole exploration)

- **Dataset**: ImageNet-1K, standard 1.28M-image train / 50K-image val split,
  224x224 resolution unless stated otherwise.
- **Metric**: top-1 accuracy on the validation set. Each reported number is
  the mean over three independent training runs with different random seeds
  (std is also tracked).
- **Compute regime**: models are kept near the "ResNet-50 / Swin-T" FLOPs
  class, roughly 4-5x10^9 FLOPs, so that accuracy comparisons are not
  confounded by capacity differences. FLOPs are reported alongside accuracy
  at every step since architectural edits can shift them.
- **Given baseline numbers** (pre-dating this exploration, from public
  released weights / prior publications): ResNet-50 (torchvision release,
  original ImageNet training recipe, 90 epochs, SGD) reaches 76.13% top-1 at
  4.09 GFLOPs. Swin-T reaches 81.30% top-1 at 4.50 GFLOPs. These two numbers
  bound the exploration: the ConvNet starts at 76.1 and Swin-T at 81.3 is the
  benchmark being chased, though nothing requires matching it exactly.

## Prior art / background (known before this exploration starts)

- **ResNet (He et al., 2015).** Residual mapping `F(x) + x` with an identity
  shortcut, organized into stages of "bottleneck" blocks (1x1 reduce -> 3x3
  conv -> 1x1 expand) at decreasing spatial resolution and increasing
  channel width. The de facto standard ConvNet backbone.
- **ResNeXt (Xie et al., 2016).** Replaces the 3x3 conv in the ResNet
  bottleneck with a *grouped* convolution (filters split into groups, each
  group only sees a subset of input channels). The guiding principle:
  grouping reduces FLOPs, so use the savings to widen the network back up.
  A grouped conv with groups = channels is the degenerate case usually
  called depthwise convolution (popularized separately by MobileNet and
  Xception).
- **MobileNetV2 (Sandler et al., 2018).** The "inverted residual" block:
  instead of wide -> narrow -> wide (bottleneck), go narrow -> wide -> narrow
  — a cheap 1x1 conv expands channels, a depthwise conv operates on the wide
  representation, and a second 1x1 conv projects back down. Designed for
  efficiency on mobile hardware.
- **Vision Transformer / ViT (Dosovitskiy et al., 2020).** Splits an image
  into non-overlapping patches via a single strided "patchify" convolution
  (or equivalent linear projection), then applies a standard Transformer
  encoder: pre-norm attention and MLP sublayers, each MLP with a hidden
  dimension four times the model width (an inverted-bottleneck-shaped
  block), GELU activation, and exactly one activation function inside the
  MLP sublayer (none elsewhere in the block).
- **Swin Transformer (Liu et al., 2021).** A hierarchical ViT variant with
  four stages of decreasing spatial resolution and increasing channel width,
  local-window (and shifted-window) self-attention instead of full global
  attention, a 4x4-patch stem, an explicit "patch merging" (downsampling)
  layer inserted *between* stages rather than folded into the first block of
  each stage, and a stage compute ratio of 1:1:3:1 blocks for the "Tiny"
  size class (1:1:9:1 for larger variants).
- **LayerScale (CaiT / Touvron et al., 2021).** A learnable per-channel
  diagonal scaling applied to a residual branch's output before it is added
  back, initialized to a small value (commonly 1e-6), used to stabilize
  training of deeper Transformer stacks.
- **Known training-technique differences.** Transformer-family papers
  (DeiT, Swin) do not just use a different architecture; they also use a
  materially different training recipe from classic ResNet training:
  AdamW instead of SGD, much longer schedules (hundreds of epochs instead
  of ~90), and a heavier bundle of data augmentation (Mixup, CutMix,
  RandAugment, Random Erasing) and regularization (Stochastic Depth, Label
  Smoothing). It is documented in recent studies that such training recipes
  alone can substantially lift plain ResNet-50 accuracy, independent of any
  architecture change — meaning a naive ResNet-vs-Transformer comparison
  under each family's own default recipe confounds architecture with
  optimization procedure.
- **Known failure mode.** Directly substituting LayerNorm for BatchNorm in
  an otherwise-unmodified ResNet is documented to produce suboptimal
  performance — a fact that predates and complicates any plan to adopt
  Transformer-style normalization wholesale.

## Fixed substrate

Every rung trains and evaluates strictly within the protocol above. Only the
network architecture and/or the training recipe are edited; the dataset,
resolution, evaluation metric, and reporting convention (mean top-1 over
three seeds, with FLOPs reported alongside) are fixed throughout.
