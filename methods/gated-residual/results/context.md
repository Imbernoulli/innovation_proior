# Context: training very deep residual-style networks (circa 2019-2020)

## Research question

A network of `L` layers and width `w` typically has expressive power that grows exponentially
in depth but only polynomially in width, so depth is the lever practitioners reach for. Yet
depth is exactly where gradient-based training breaks down. During training a signal must
propagate forward from input to output, and the cost-function gradient must propagate
backward to produce a useful weight update. If each layer changes the magnitude of a
perturbation by a factor `r`, then after `L` layers both the forward signal and the backward
gradient are scaled by `r^L`: any `r` away from 1 makes them vanish or explode exponentially,
and the network is untrainable in practice. The precise goal is an architecture (or
architectural change) that lets the *same* gradient-based recipe train residual-style networks
to large depth — fully connected stacks, convolutional ResNets, and the self-attention stacks
that had recently become dominant — with these properties at once: (1) forward and backward
signals neither vanish nor explode through many layers; (2) the change is *simple* and
*architecture-agnostic*, a few lines that drop into any residual block rather than a scheme
re-derived per architecture; (3) it adds negligible compute and few or no extra
hyperparameters; (4) it works without the crutches deep models had come to rely on — no
learning-rate warm-up, no auxiliary intermediate-layer losses, and not necessarily a
normalization layer. Each prior approach below buys a subset of these; none delivers all four
in one cheap, generic move. Closing that gap is the problem.

## Background

By this time the field had three broad families of fixes for deep signal propagation —
**careful initialization, normalization, and residual connections** — and a body of theory
explaining when each works.

**Signal propagation and the edge of chaos.** Mean-field analyses of randomly initialized
deep networks (Poole et al. 2016; Schoenholz et al. 2016; Yang & Schoenholz 2017) show that
for a feed-forward net `x^l = phi(h^l)`, `h^l = W^l x^{l-1} + b^l`, the cosine distance
between two distinct input signals approaches a depth fixed point that is either 1 (the ordered
phase: all inputs collapse to one output, weight updates vanish) or 0 (the chaotic phase:
similar inputs diverge wildly, weight updates explode). The boundary between them is the *edge
of chaos*. The relevant object is the input-output Jacobian `J = ∂x_L/∂x_0 = ∏_{l=1}^L D^l W^l`,
where `D^l = diag(phi'(h^l))`. Its mean squared singular value is `chi^L`, where
`chi = (1/N) ⟨Tr (DW)^T DW⟩ = sigma_w^2 ∫ Dh [phi'(sqrt(q*) h)]^2`, with `q*` the
fixed-point pre-activation variance and `Dh` the standard Gaussian measure. Training proceeds
efficiently at the edge of chaos, `chi = 1`, where on average signal strength is preserved.

**Dynamical isometry — a stronger requirement.** Pennington, Schoenholz & Ganguli (2017, 2018)
used free-probability theory to compute the *entire* singular-value distribution of `J`, and
showed that `chi = 1` (a unit *mean* squared singular value) is necessary but not sufficient:
if the singular vectors attached to very large or very small singular values happen to align
with the perturbations present in the data, training is still inefficient. The stronger
condition is **dynamical isometry** (Saxe et al. 2013): *all* singular values of `J` are close
to 1, so every perturbation of the input propagates through the network equally well. They
established the consequences sharply. For deep *linear* nets, orthogonal weight initialization
achieves dynamical isometry and makes learning time (in epochs) essentially independent of
depth, while Gaussian initialization does not. For nonlinear nets the nonlinearity's derivative
distribution enters through `D^2`; tanh/sigmoid with orthogonal weights can be tuned to
isometry, but — proven rigorously — **ReLU networks cannot achieve dynamical isometry at all**:
a ReLU zeroes its derivative for some inputs, forcing vanishing singular values into `J`.
Empirically, networks that achieve dynamical isometry learn orders of magnitude faster than
those that merely sit at `chi = 1`.

**Why residual stacks explode in variance at initialization.** A residual network without
normalization accumulates `x_l = x_0 + sum_{i<l} F_i(x_i)`. Take the blocks zero-mean at init
with `E[F_i(x_i) | x_i] = 0`; then the residual branch and the skip path are uncorrelated and
variances add, `Var[x_{l+1}] = Var[x_l] + Var[F(x_l)]`. Standard fan-in (He) initialization
makes a block roughly preserve its input variance, `Var[F(x_l)] ≈ Var[x_l]`, so
`Var[x_{l+1}] ≈ 2 Var[x_l]` and `Var[x_l] ≈ 2^l`: the hidden-activation variance grows
*exponentially* with depth at initialization (De & Smith 2020). Two consequences follow: the
output scale blows up with depth, and — since branch and skip contribute equally — the function
each block computes is far from the identity. For positively-homogeneous (ReLU, conv, linear)
blocks one can further show (Zhang et al. 2019) that the gradient norm of certain
weight tensors is *lower-bounded* by a quantity that grows with depth, so deep unnormalized
residual nets suffer exploding gradients at the start of training. Batch normalization happens
to cure this: it downscales the residual branch by a factor on the order of `sqrt(depth)` at
initialization (De & Smith 2020), so deep normalized blocks are dominated by the skip path and
sit close to the identity — which is *why* batch normalization made very deep ResNets trainable.

**The diagnostic for self-attention stacks.** The same lens applied to a Transformer layer is
unflattering. A layer is `x_{i+1} = LayerNorm(x_i + sublayer(x_i))`. LayerNorm,
`LayerNorm(x) = gamma (x - E[x]) / sqrt(Var[x]) + beta`, is invariant to any perturbation that
purely shifts an input's mean or its variance, so each normalized token contributes two
vanishing singular values to `J` (`2n` per layer for a length-`n` sequence). Scaled dot-product
attention `softmax(QK^T/sqrt(d)) V` adds its own: when the pre-softmax logits are small the
softmax returns a near-uniform `n×n` matrix with entries `≈ 1/n`, projecting all `n` embedding
vectors onto a single direction, so of the `n×d` Jacobian singular values only `d` survive. A
residual connection restores some signal but still amplifies some perturbations and attenuates
others. Empirically, the input-output Jacobian of randomly initialized Transformer encoders is
peaked near 1 at shallow depth but, by 64 layers, has a large mass of singular values that have
vanished to machine precision — consistent with the common experience that deep Transformers
are extremely hard to train, requiring learning-rate warm-up or auxiliary losses to converge.

## Baselines

These are the prior approaches a new method would be measured against and reacts to.

**ResNet (He et al. 2016, arXiv:1512.03385).** Reformulate a block to fit a residual function:
`y = sigma(F(x) + x)`, with `F = W_2 sigma(W_1 x)` and an identity shortcut (a `1×1`
projection `W_s x` only when channel count or stride changes), and a second ReLU *after* the
addition. The motivation is the *degradation problem* — empirically, deeper plain networks have
higher *training* error than shallower ones, which is not overfitting; if the added layers
could realize identity maps the deep net could be no worse, so making it *easy* to represent
identity (drive `F → 0`) preconditions the problem. **Gap:** the block is not initialized at the
identity — `W_1, W_2` are random — so at depth the stacked random branches still produce the
exponential variance growth above, and the post-addition ReLU sits on the merged path.

**Pre-activation ResNet (He et al. 2016, arXiv:1603.05027).** Move the activation and
normalization *before* the weight layers, `x_{i+1} = x_i + F(x_i)` with
`F = BN-ReLU-Conv-BN-ReLU-Conv`, so the shortcut path is a clean, unmodulated identity (no
activation on the merged signal). This improves gradient flow through the identity path. **Gap:**
the branch is still randomly initialized and contributes fully from step one, so deep stacks
still do not start near the identity.

**Highway Networks (Srivastava, Greff & Schmidhuber 2015, arXiv:1505.00387).** The first
feed-forward nets with hundreds of layers. Regulate flow with learned, *input-dependent* gates:
`y = H(x) · T(x) + x · C(x)`, simplified to `C = 1 - T` with a transform gate
`T(x) = sigma(W_T^T x + b_T)`. When `T(x) = 0` the layer passes its input through (Jacobian
`I`); when `T(x) = 1` it is a plain layer. The carry bias `b_T` is initialized negative to bias
the network toward carrying its input early in training. **Gap:** each gate is a full weight
tensor `W_T` acting nonlinearly on the signal — a gating sub-network per layer, with extra
parameters and compute — and because `sigma` never reaches exactly 0 or 1, the network is *not*
initialized as the identity map; it is only biased toward it.

**Gated ResNet (Savarese, Mazza & Figueiredo 2016, arXiv:1611.01260).** Collapse the Highway
gate to a *single scalar* `k` per block: `u = g(k) f_r(x, W) + x`, with `g` a scalar gate (e.g.
clamped ReLU), reasoning that learning `k → 0` to degenerate a layer into identity is easier
than learning a whole tensor `W_g → 0`. The scalar `k` is initialized so that `g(k) = 1` — the
layer starts as a *standard residual block*. **Gap:** because it starts at the standard residual
configuration, the layer must *unlearn* a fully-contributing random branch to approach identity,
and the variance/Jacobian pathologies of the standard residual block are present at
initialization; the gate is also wrapped in a function `g(·)` that constrains its range.

**Zero-`gamma` (Goyal et al. 2017; Hardt & Ma 2016; He et al. 2019).** If the last operation in
a residual branch is a normalization layer, initialize its scale parameter `gamma` to 0, so the
branch outputs 0 at init and the block begins as the identity. **Gap:** it zero-initializes
*many* parameters per block (one per channel) and is only applicable when a normalization layer
sits at the end of the branch.

**FixUp (Zhang, Dauphin & Ma 2019, arXiv:1901.09321).** Train deep ResNets *without*
normalization by rescaling the standard initialization to make the per-step SGD update of the
whole network `Theta(eta)` and depth-independent. The recipe: (1) zero-initialize the
classifier and the last layer of each residual branch; (2) initialize other layers normally but
scale the weight layers *inside* residual branches by `L^{-1/(2m-2)}` (with `m` the number of
layers per branch) — stated to be the essential rule; (3) add a scalar multiplier (init 1) in
every branch and scalar biases (init 0) before each conv/linear/activation. **Gap:** it is a
multi-part, architecture-aware scheme — the rescaling exponent depends on `L` and on the
per-block layer count `m`, and several pieces must be combined correctly to match a normalized
baseline.

**SkipInit (De & Smith 2020, arXiv:2002.10444).** From the analysis that batch normalization
works by downscaling the residual branch by `~sqrt(d)` (with `d` the number of blocks): replace
the branch's normalization with a single learnable scalar `alpha` at the end of the residual
branch, using a small depth-dependent initial scale to reproduce that downscaling. **Gap:**
framed as a *replacement* for an existing normalization layer (so it presupposes one was there),
and its recommended scale is depth-dependent (`1/sqrt(d)` requires knowing the block count).

## Evaluation settings

The natural yardsticks already in use, across the three architecture families.

- **Fully connected (training dynamics).** Stacks of fully connected ReLU layers (e.g. 32
  layers, width 256) trained on CIFAR-10 image classification, optimized with Adagrad
  (learning rate `0.01`); interest is purely in how many iterations it takes to fit the training
  data as a function of depth, and in how deep a stack can be made trainable at all. Weights
  initialized with training-optimal variance `sigma_W^2 = 2/w` (He et al. 2015).
- **Convolutional ResNets on CIFAR-10.** ResNet-56, ResNet-110, and pre-activation
  ResNet-18/50, all with identical hyperparameters. SGD with batch size 128, momentum 0.9,
  weight decay `5×10^{-4}`; a step-down learning-rate schedule starting at `0.1` and divided by
  10 at epochs 100 and 150, for 200 epochs total (He et al. 2016). A one-cycle "superconvergence"
  schedule (Smith & Topin 2019) — learning rate ramped up to a large peak then down — is the
  fast-training variant. Metrics: validation error, number of epochs to reach a target accuracy
  (e.g. 80%), and training loss.
- **Transformers on enwiki8.** Character-level language modeling on enwiki8 (Mahoney 2009),
  measured by bits-per-byte (BPB) and by the number of iterations to reach a target BPB (e.g.
  1.2). A 12-layer encoder with hidden size 512, 2 attention heads, context 512, GELU
  feed-forward, dropout 0.2, optimized with LAMB at a fixed learning rate `0.016`, batch size
  ~1080. The natural normalization-placement baselines to compare against are Post-Norm (the
  original Transformer), Pre-Norm, and GPT2-Norm.
- Protocol throughout: identical hyperparameters across the compared methods, so differences are
  attributable to the architectural change rather than tuning.

## Code framework

The change lives inside one residual block that the surrounding backbone, data pipeline,
optimizer, loss, and training loop treat as a black box. The block must keep the existing
interface: a constructor `CustomBlock(in_planes, planes, stride)`, a class attribute
`expansion`, a `forward(x)` that returns a tensor with `planes * expansion` channels, and a
shortcut that handles a dimension mismatch when `stride != 1` or the channel count changes.
Everything outside the block — the optimizer (SGD, `lr=0.1`, momentum `0.9`, weight decay
`5e-4`), the cosine/step learning-rate schedule, the data augmentation, the global pooling and
classifier head, and the outer training loop — is fixed. What the block does *internally* with
its two convolutions, its normalizations, its activations, and how it combines the branch with
the shortcut is the open slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
    """Residual block plugged into a fixed CIFAR ResNet backbone.

    Interface the backbone relies on:
      - constructor signature CustomBlock(in_planes, planes, stride)
      - class attribute `expansion`
      - forward(x) returns a tensor with planes * expansion channels
      - the shortcut must absorb a stride / channel-count mismatch
    The internal transformation and how the branch is combined with the
    shortcut is the open design slot.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )
        # TODO: any block-level state the design ends up needing.

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))      # F(x): the residual branch
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        # TODO: define the merge rule for the transformed branch and shortcut.
        pass


# existing backbone the block plugs into (fixed)
def make_resnet(block, num_blocks_per_stage, num_classes=10):
    """Stacks `block` into the standard CIFAR ResNet stages; the stem,
    global average pooling, and linear classifier are fixed."""
    ...
```

The single open slot is the merge rule for the transformed branch and the shortcut.
