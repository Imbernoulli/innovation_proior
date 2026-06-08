## Research question

Design the pointwise nonlinearity for a set of deep convolutional networks so that test accuracy rises
across three architecture/dataset pairs at once, while everything else about the networks and their
training is frozen. The single thing being designed is the activation curve — the element-wise map
applied after each Conv–BatchNorm pair (and in the classifier head). It must be differentiable,
shape-preserving (tensor in, same-shape tensor out), and a drop-in for the ReLU that the architectures
were originally built around. The optimizer, learning-rate schedule, weight initialization, data
augmentation, and the model definitions themselves are fixed; the only lever is the shape of the
nonlinearity.

## Prior art before the first rung (the activation lineage)

The first rung reacts to ReLU and the smooth, self-gated units that grew out of it. These precede the
ladder; each leaves a gap the ladder's curves try to close.

- **ReLU** (Nair & Hinton 2010): `max(x,0)`. Cheap, sparse, derivative exactly 1 for `x>0`, which made
  deep nets trainable where saturating sigmoid/tanh units killed the gradient. Gap: identically zero
  output *and* zero gradient on the whole negative half-line, so a unit pushed permanently negative
  stops learning ("dying ReLU"); a non-differentiable kink at the origin; piecewise-linear, so no
  curvature; monotonic. This is the scaffold's *default* fill.
- **Leaky ReLU / PReLU** (Maas et al. 2013; He et al. 2015): `max(x, px)`, with `p` a small constant
  or a learned per-channel slope. Leak a little gradient on the negative side. Gap: still
  piecewise-linear, kinked, monotonic.
- **ELU** (Clevert et al. 2015): `x` for `x≥0`, `α(eˣ−1)` for `x<0`. Negative saturation pushes the
  mean activation toward zero. Gap: monotonic, an extra constant, and reported instability with
  residual blocks unless BatchNorm sits at the block end.
- **Self-gated smooth units** — SiLU/Swish `x·σ(βx)` (Elfwing et al. 2017; Ramachandran et al. 2017),
  GELU `x·Φ(x)` (Hendrycks & Gimpel 2016), Mish `x·tanh(softplus(x))` (Misra 2019). All multiply the
  raw input by a smooth nonlinear function of itself, so they are differentiable everywhere, take small
  negative values (no dying units), are unbounded above and bounded below, and are non-monotonic with a
  shallow negative "bump." Gap among themselves: each is a single fixed curve, and which one trains best
  on a given CNN setting — and why — is exactly what the ladder probes.

## The fixed substrate

The training pipeline and the model definitions are frozen and must not be touched:

- **Optimizer:** SGD, `lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`.
- **Schedule:** cosine annealing over `200` epochs (`CosineAnnealingLR`, `T_max=200`).
- **Initialization:** Kaiming-normal (`fan_out`, `nonlinearity='relu'`) on conv weights, `fan_in` on
  linear weights; BatchNorm weight 1, bias 0. Fixed regardless of which activation is used.
- **Data:** `Resize(32)` → `RandomCrop(32, padding=4)` → `RandomHorizontalFlip` → `ToTensor` →
  per-dataset `Normalize`; FashionMNIST is repeated to 3 channels. Test transform is resize + normalize
  only. Batch size 128, `CrossEntropyLoss`.
- **Architectures:** CIFAR ResNet-20 (BasicBlock `[3,3,3]`, 3×3 stem, global avg pool); VGG-16-BN
  (Conv–BN–activation blocks + a 512→512 classifier with the activation and dropout 0.5);
  MobileNetV2 (inverted residuals, the activation replacing the usual ReLU6). The activation module is
  instantiated and called in every one of these places.

The metric is *best test accuracy reached during the 200 epochs*, so the activation is judged on the
peak of the fixed cosine run, not on a re-tuned schedule.

## The editable interface

Exactly one region is editable — the `CustomActivation` class in `pytorch-vision/custom_activation.py`
(the `nn.Module` used everywhere a ReLU would be). The contract is a shape-preserving element-wise (or
channel-wise) `forward(x)`; learnable parameters may be registered in `__init__`. Every rung on the
ladder is a different fill of exactly this class and nothing else.

The starting point is the scaffold default: plain ReLU.

```python
# EDITABLE region of pytorch-vision/custom_activation.py (lines 32-49) -- default fill (ReLU)
class CustomActivation(nn.Module):
    """Custom activation function. Drop-in replacement for ReLU.

    Input/Output: tensor of any shape, element-wise operation.

    Design considerations:
        - Monotonicity vs non-monotonicity tradeoffs
        - Smoothness (differentiability everywhere vs piecewise)
        - Negative-domain behavior (zero, linear, bounded)
        - Computational cost (simple ops vs learned parameters)
        - Interaction with BatchNorm and residual connections
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x)  # default: ReLU
```

(`torch`, `torch.nn as nn`, and `torch.nn.functional as F` are already imported at module top.)

## Evaluation settings

Three architecture/dataset pairs, one seed, evaluated by best test accuracy (%, higher is better):

- **resnet20-cifar10** — ResNet-20 on CIFAR-10.
- **vgg16bn-cifar100** — VGG-16-BN on CIFAR-100.
- **mobilenetv2-fmnist** — MobileNetV2 on FashionMNIST (grayscale repeated to 3 channels).

Seed `42`. Each run is the full fixed pipeline above (SGD + cosine, 200 epochs) with only
`CustomActivation` changed.
