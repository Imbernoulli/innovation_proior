## research question

Swish `x·σ(βx)` was found by neural-architecture search, is widely used, and reliably improves over
ReLU — but *nobody can say why it works or what it is*, so the field keeps treating activation design as
black-box search. Two questions follow. First, is there a principled formula that explains the searched
Swish — that derives it from something familiar rather than discovering it empirically? Second, if such
a formula exists, does it generalize to a *family* of activations that strictly contains Swish, and can
the network be made to *learn, per channel and per sample, whether to behave nonlinearly (activate) or
linearly (not)* at negligible parameter and compute cost, especially in large optimized networks where
a single fixed activation has little remaining headroom?

## background

The activation lineage: ReLU `max(x,0)` (Hahnloser 2000; Nair & Hinton 2010), and its piecewise-linear
relatives Leaky ReLU `max(x, αx)` (Maas et al. 2013) and PReLU `max(x, px)` with learned per-channel
`p` (He et al. 2015). Two more general objects sit above these. **Maxout** (Goodfellow et al. 2013):
`max(η_a(x), η_b(x))` for linear `η_a, η_b` — a piecewise-linear unit that, by choosing the two linear
pieces, *generalizes* ReLU (`max(x,0)`), Leaky ReLU / PReLU (`max(x, px)`), and the absolute value, and
can approximate any convex piecewise-linear activation; it is the general "max of linear functions"
family. **Softplus** `log(1+eˣ)` (Dugas et al. 2000): a smooth approximation to ReLU `max(x,0)` built
from the LogSumExp smooth max. LogSumExp smooths the hard value `max(x,0)`; its derivative is the
sigmoid gate. It is related to, but not the same object as, the softmax-weighted value average below.

On the search side: Swish `x·σ(βx)` (Ramachandran et al. 2017), found by an RL/exhaustive search over
composed primitives; with `β=1` it equals the SiLU `x·σ(x)` (Elfwing et al. 2017; Hendrycks & Gimpel
2016). Swish is smooth, non-monotonic, unbounded above, bounded below; its first derivative has
*fixed* upper/lower bounds (≈ 1.0998 and ≈ −0.0998), with `β` only setting how fast the derivative
asymptotes to those fixed values, not the values themselves.

The mathematical tool that connects these is the **smooth maximum** (a.k.a. `α-softmax`,
softmax-weighted mean), standard in optimization and neural computation (Lange et al. 2014; Boyd &
Vandenberghe 2004): for values `x_1,…,x_n`,
`S_β(x_1,…,x_n) = (Σ_i x_i e^{βx_i}) / (Σ_i e^{βx_i})`, a smooth, differentiable surrogate for the hard
`max` with a nonnegative switching factor `β` (large positive `β` → `max`; `β=0` → arithmetic mean). The key question is what
this weighted-value smoothing gives when applied to the two-piece members of the Maxout family.

Related design directions for "let the network adapt its nonlinearity": **SENet** (Hu et al. 2018)
reweights channels with a small squeeze-and-excitation routing function `σ(W₁W₂·GAP(x))` (a cheap
channel-attention bottleneck, reduction `r`); **DY-ReLU** (Chen et al. 2020) makes a piecewise-linear
activation's coefficients a hyperfunction of the global context, at a larger routing cost. These are the
comparison points for an input-conditioned activation, and the bar is to add adaptive nonlinear behavior
without turning the activation into a heavy dynamic module.

A diagnostic motivating the problem: across model scales, the *relative* ImageNet improvement of
Swish (and of SENet) over a ReLU baseline shrinks as models get larger and deeper — the gain that is
easy on a small MobileNet is hard to keep on a ResNet-152. Any candidate activation therefore has to be
checked on both small mobile models and large optimized backbones.

## baselines

- **ReLU** `max(x,0)`: hard sign gate; dead negative half, kink, no curvature. The `max(x,0)` member of
  Maxout. Gap: not smooth, dying units, fixed and non-adaptive.
- **Leaky ReLU / PReLU** `max(x, px)` (`p` fixed small / learned per-channel, init 0.25): leak on the
  negative side. Gap: piecewise-linear, kinked, monotonic; the `max(x,px)` member of Maxout.
- **Maxout** `max(η_a(x), η_b(x))`: general max-of-linears; contains ReLU/LReLU/PReLU/abs and
  approximates convex PWL activations. Gap: non-smooth (it's a hard max), and it *multiplies* the
  parameters/feature-maps per unit — expensive — and gives no smooth, learnable interpolation toward
  linearity.
- **Swish / SiLU** `x·σ(βx)`: smooth, non-monotonic, the strongest searched incumbent. Gap: derived by
  black-box search with no explanation; first-derivative upper/lower bounds are *fixed* (≈1.0998,
  ≈−0.0998) regardless of training — `β` only sets the asymptotic speed; gains shrink on large/deep
  models.
- **Softplus** `log(1+eˣ)`: LogSumExp smoothing of ReLU. Gap: monotonic and not self-gated; it does not
  expose the softmax-weighted value form `x·σ(βx)` that Swish uses.
- **GELU / Mish** `x·Φ(x)`, `x·tanh(softplus(x))`: other strong smooth self-gated curves. Gap: each a
  single fixed curve; no learnable switch between linear and nonlinear behavior, no derivation as a
  smooth Maxout.
- **SENet** `σ(W₁W₂·GAP(x))` channel reweighting; **DY-ReLU** input-conditioned PWL coefficients. Gap:
  SENet modulates features rather than reshaping the activation derivative; DY-ReLU buys conditioned
  slopes with a heavier dynamic-activation module.

## evaluation settings

Drop-in activation swaps with architecture/training otherwise fixed, on ImageNet-1k classification
(224×224, top-1 error) across a spread of model sizes — light-weight (MobileNetV1/V2 at several width
multipliers, ShuffleNetV2 0.5×/1.5×) and deep (ResNet-18/50/101/152, and the highly-optimized
SENet-154). Light-weight training follows the ShuffleNetV2 recipe; large models use a linear-decay LR
from 0.1, weight decay 1e-4, batch 256, ~600k iterations. Static-activation comparison: ReLU vs.
Swish vs. learned smooth two-piece activations, same nets. Input-conditioned comparison: against SENet on the same
backbones, and an ablation of the routing-function design space (layer-/channel-/pixel-wise) and other
activations (Mish, ELU, Softplus) on ShuffleNetV2 0.5×. Transfer: COCO detection (RetinaNet, ResNet-50
backbone; AP at IoU thresholds/scales) and CityScapes segmentation (PSPNet, ResNet-50; mean IoU).
Metric: top-1 error / AP / mIoU, plus parameter and FLOP counts to track activation overhead and
switching-factor histograms for input-conditioned gates.

## code framework

```python
import torch
import torch.nn as nn


def smooth_maximum(values, beta):
    """S_beta(x_1,...,x_n) = sum_i x_i e^{beta x_i} / sum_i e^{beta x_i}.

    Smooth, differentiable surrogate for max with temperature beta.
    The two-argument case is the workhorse here.
    """
    # TODO: reduce two linear pieces through the weighted-value smooth maximum.
    pass


class SmoothMaxActivation(nn.Module):
    """A smooth approximation to a Maxout member max(eta_a(x), eta_b(x)).

    Reference points from the hard Maxout family:
      eta_a = x,    eta_b = 0    -> max(x, 0)   = ReLU
      eta_a = x,    eta_b = p x  -> max(x, p x) = PReLU
      eta_a = p1 x, eta_b = p2 x -> max(p1 x, p2 x)
    A smooth activation can be built by applying smooth_maximum to these two
    linear pieces, with a switching temperature controlling linear vs. nonlinear.
    """

    def __init__(self, num_channels):
        super().__init__()
        # TODO: per-channel parameters for the two linear pieces and the switch.
        pass

    def forward(self, x):
        # TODO: smooth_maximum(eta_a(x), eta_b(x); beta).
        pass


class BetaGenerator(nn.Module):
    """Optional routing function that produces the switching temperature
    from the input feature (layer-/channel-/pixel-wise).

    A cheap channel-attention-style bottleneck already exists as a primitive:
      beta_c = sigmoid(W1 W2 GAP(x))   # W1: C x C/r, W2: C/r x C, r=16
    """

    def __init__(self, num_channels, reduction=16):
        super().__init__()
        # TODO: the generator G(x) -> beta, if the switch is learned from input.
        pass

    def forward(self, x):
        # TODO: return beta conditioned on x.
        pass
```
