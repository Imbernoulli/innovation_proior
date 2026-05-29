# Context

## Research question

Convolutional networks for visual recognition have been getting deeper at a rapid pace: the
original LeNet had five layers, VGG had nineteen, and only in the past year have Highway Networks
and Residual Networks crossed the hundred-layer mark. Depth helps — more layers mean richer,
more abstract features — but it introduces a structural problem. As the input signal flows
forward through dozens or hundreds of nonlinear transformations, and as the gradient flows
backward through the same stack, both can attenuate and "wash out" before reaching the far end.
A plain stack of many layers becomes hard to optimize *despite* having more than enough capacity.

The precise question: what is the simplest connectivity pattern between layers that keeps both
information (forward) and gradient (backward) flowing freely across very deep networks — so that
hundreds of layers train without degradation — and that, ideally, does so without the parameter
bloat of simply making networks ever wider or ever deeper? A good solution must (a) provide short
paths between early and late layers, (b) train cleanly at 100+ layers with no optimization
breakdown, and (c) use parameters efficiently rather than re-learning redundant features.

## Background

**Depth, and why naïvely stacking layers fails.** The standard feed-forward CNN composes layers
so that each consumes only the output of its immediate predecessor: x_ℓ = H_ℓ(x_{ℓ−1}), where H_ℓ
is a small stack of operations (typically convolution, a nonlinearity, normalization). Stacking
many such layers ought to be at least as expressive as stacking few, but a robust empirical
finding is the **degradation problem**: as plain networks get deeper, their *training* error gets
*worse*, not better. This is not overfitting — training error itself rises. A shallower network
plus extra identity-copying layers is, by construction, a solution that achieves the shallow
network's error, yet gradient descent fails to find it. The bottleneck is optimization and signal
flow, not capacity.

**Normalization and activations.** Batch Normalization (Ioffe & Szegedy, 2015) renormalizes layer
inputs to stabilize training and allow higher learning rates; ReLU avoids gradient saturation.
These are the standard building blocks of a layer's internal transformation. A widely adopted
refinement orders the operations as BN → ReLU → Conv ("pre-activation"), which keeps any
skip/shortcut path clean of nonlinearities.

**The convergent insight of the time: short paths.** A cluster of recent architectures all attack
the flow problem by creating shortcuts from early layers to late layers, differing mainly in *how*
the shortcut is formed:

- Highway Networks (Srivastava et al., 2015) and Residual Networks (He et al., 2015) bypass the
  nonlinear block by carrying the input forward.
- Stochastic depth (Huang et al., 2016) randomly removes whole blocks during training, which both
  shortens the effective network and shows that many layers are redundant.
- FractalNet (Larsson et al., 2016) interleaves parallel sub-paths of different lengths to keep
  many short paths alive inside a nominally deep net.
- Deeply-Supervised Nets (Lee et al., 2015) attach auxiliary classifiers to intermediate layers so
  gradient reaches them directly.

Despite different topologies, they share one mechanism: **short connections from layers near the
input to layers near the output.**

**Concatenation as a combine operation.** The Inception module (Szegedy et al., 2015) combines the
outputs of differently-sized filters by **concatenation** rather than summation, to increase the
diversity of features feeding the next stage. Szegedy et al. (2016) also observe that features
within a single convolutional layer are strongly correlated — one can aggregate channels into
fewer without much accuracy loss — which suggests conventional layers waste capacity producing
redundant feature-maps.

**The 1×1 bottleneck.** A 1×1 convolution placed before an expensive 3×3 convolution can shrink the
number of input channels, cutting computation and parameters while mixing channels. This was used
in Inception and in deep ResNet blocks.

**Motivating diagnostic findings.** Three observed phenomena about existing systems frame the
problem: (1) deeper plain networks have higher training error (an optimization failure);
(2) randomly dropping residual layers during training does not hurt and can *help* accuracy — so
very deep networks carry substantial redundancy, with each layer's dedicated full-width weights
largely wasted; (3) features from the same layer are highly correlated and compressible. Together
these say: the gains of depth are real, but the way depth is currently spent — full-width weights
re-learned and re-carried at every layer — is wasteful.

## Baselines

**Highway Networks (Srivastava et al., 2015).** The first architectures to train end-to-end at
more than 100 layers. A highway layer computes

    y = H(x) · T(x) + x · C(x),

where H is the layer transformation, T is a learned "transform gate" and C a learned "carry gate,"
usually tied as C = 1 − T (each a sigmoid of the input). When T ≈ 0 the layer copies its input
forward unchanged along an "information highway." Gap: the gates are **data-dependent and add
parameters**; the shortcut is only as open as the learned gate chooses, so there is no guaranteed
clean path, and the gating machinery must itself be trained.

**Residual Networks (He et al., 2015).** Replace the learned gate with a fixed identity shortcut.
Instead of learning a layer's full target mapping H(x) directly, the block learns the *residual*
F(x) = H(x) − x and adds the input back:

    x_ℓ = H_ℓ(x_{ℓ−1}) + x_{ℓ−1}.

The identity shortcut introduces **no extra parameters**, and the gradient flows directly through
it from late layers to early ones, which makes 100+ layer networks trainable and fixes the
degradation problem. Gap: the shortcut and the block output are merged by **summation**. Summation
can impede information flow — the carried-forward signal and the newly computed features are added
into one tensor and can partially cancel or overwrite each other — and each residual unit still
owns a full-width set of weights, so the network re-writes its entire state at every block.

**Pre-activation ResNet (He et al., 2016).** Reorders the block to BN → ReLU → Conv so the
shortcut path is a clean identity with no intervening nonlinearity. With a clean identity,
unrolling the recursion gives, for any later layer L and earlier layer ℓ,

    x_L = x_ℓ + Σ_{i=ℓ}^{L−1} F(x_i),

so the gradient is

    ∂Loss/∂x_ℓ = ∂Loss/∂x_L · ( 1 + ∂/∂x_ℓ Σ_{i} F(x_i) ).

The constant "1" guarantees the gradient at x_L reaches x_ℓ undiminished no matter how deep the
net, which is why pre-activation ResNets train past 1000 layers. This is the most
parameter-efficient strong baseline of the time and the natural yardstick. Gap remains the
additive merge and per-layer full-width weights.

**Stochastic depth (Huang et al., 2016).** A training-time regularizer for ResNets: each residual
block survives with probability p_ℓ (decaying linearly with depth) and is otherwise replaced by
identity for that minibatch; at test time all blocks are present, rescaled by survival probability.
It trains a 1202-layer ResNet successfully and lowers error. Two facts it establishes: deep
residual nets are **highly redundant** (whole layers can be dropped at random with no harm), and
dropping the layers between two surviving layers creates a **direct connection** between them — so
short, non-adjacent connections are exactly what eases training. Gap: it only creates such
connections *randomly and transiently*; the architecture underneath is still a summation-based,
full-width residual net.

**Inception / GoogLeNet (Szegedy et al., 2015, 2016).** Builds width via "Inception modules" that
concatenate the outputs of parallel filters of different sizes within a module, and uses 1×1
bottlenecks for efficiency. It demonstrates concatenation as a viable combine operation and strong
accuracy at lower cost than VGG. Gap: concatenation is confined *within* a module, not used to wire
distant layers together; the design is comparatively complex.

## Evaluation settings

The natural yardsticks are the standard object-recognition benchmarks:

- **CIFAR-10 / CIFAR-100** — 32×32 color images, 50,000 train / 10,000 test, 10 and 100 classes.
  Evaluated both without augmentation and with the standard mirroring/shifting augmentation
  (denoted with a "+"). Preprocessing normalizes by per-channel mean and standard deviation; a
  small held-out validation split is used for model selection.
- **SVHN** — 32×32 color digit images, ~600k training (including the extra set) and 26k test;
  typically used without augmentation, pixels scaled to [0,1], model chosen by validation error.
- **ImageNet (ILSVRC 2012)** — 1.2M training and 50k validation images over 1000 classes; standard
  scale/aspect/crop augmentation at train time, single-crop and 10-crop 224×224 at test time.

Metric: classification error rate (top-1, and top-5 for ImageNet), reported against parameter
count and FLOPs so that accuracy can be compared at matched model size and compute. Training:
SGD with Nesterov momentum 0.9, weight decay 1e-4, He initialization, a step-decayed learning rate
starting at 0.1, and dropout after convolutions on the non-augmented datasets.

## Code framework

The existing primitives are convolution, batch norm, ReLU, pooling, an SGD optimizer,
cross-entropy loss, and a standard training loop. The unresolved part is the stage-level
transformation and how stages pass state between resolutions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def composite_function(num_input_channels, num_output_channels):
    # TODO: choose the per-layer transformation from standard primitives.
    pass


class FeatureStage(nn.Module):
    """A stack of transformations operating at one spatial resolution."""

    def __init__(self, num_layers, in_channels, out_channels):
        super().__init__()
        # TODO: instantiate the stage transformations and connectivity rule.
        pass

    def forward(self, x):
        # TODO: apply the stage.
        pass


class Downsample(nn.Module):
    """Reduce spatial size and adjust channels between stages."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO: cheap channel adjust + spatial reduction.
        pass

    def forward(self, x):
        pass


class Network(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # Stem: an initial convolution / pooling on the raw image.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # TODO: a sequence of FeatureStage + Downsample blocks.
        self.stages = None  # TODO
        # Head: pool to a vector and classify.
        self.classifier = None  # TODO: nn.Linear(final_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        # TODO: run the stages, global-average-pool, classify.
        pass


# Standard training loop, optimizer and loss already exist:
def train(model, loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                nesterov=True, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    for images, targets in loader:
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
```
