# Context

## Research question

Self-gated, smooth, non-monotonic activations of the form `x·σ(βx)` and `x·Φ(x)` have emerged as
strong ReLU alternatives on deep networks. The open question is which precise pointwise curve in that
neighborhood of functions should be used, and how its gradient behavior connects to optimization dynamics.

## Background

The activation lineage relevant here: sigmoid/tanh (smooth, bounded, saturating) → ReLU `max(x,0)`
(Nair & Hinton 2010; unsaturated on the positive half, fast) → a wave of fixes (Leaky ReLU, ELU,
SELU) → the self-gated units. The self-gated unit multiplies the raw input by a smooth nonlinear
function of itself: SiLU/Swish `x·σ(βx)` (Elfwing et al. 2017; Ramachandran et al. 2017) and
GELU `x·Φ(x)` (Hendrycks & Gimpel 2016). What these share is a distinctive concavity for negative
inputs that preserves a small amount of negative signal (instead of zeroing it like ReLU), a smooth
continuous profile (no kink), unboundedness above (no positive saturation), and boundedness below.

Two threads of prevailing wisdom inform this area. First, *smoothness of the output/loss landscape*
is associated with easier optimization and better generalization: smoother per-unit maps produce
smoother output landscapes, which correlate with smoother, wider-minima loss surfaces. Second, in
classical optimization a *preconditioner* — for gradient descent, an inverse-Hessian-like operator
applied to reshape the objective's geometry — accelerates convergence by making the landscape locally
more isotropic. Empirically, deeper plain (non-residual) networks degrade in trainability past
some depth, and activations differ in how far that degradation can be pushed back — a diagnostic that
distinguishes candidate curves.

## Baselines

- **ReLU** `max(x,0)`: cheap, sparse, derivative 1 for `x>0`; monotonic.
- **Leaky ReLU** `max(x, αx)` (`α=0.01`): a constant leak on the negative side; piecewise-linear,
  monotonic.
- **Swish / SiLU** `x·σ(βx)` (`β=1` ≈ SiLU): smooth, non-monotonic, self-gated; small negative
  bump; unbounded above, bounded below. The strongest incumbent and the direct reference point.
- **GELU** `x·Φ(x)` (Hendrycks & Gimpel 2016): smooth, non-monotonic, Gaussian-gated; very similar
  shape to Swish.
- **Softplus** `log(1+eˣ)`: smooth, strictly positive, monotonic ReLU-approximation.

## Evaluation settings

Drop-in swaps with everything else held fixed. A controlled ablation over candidate self-gated-style
curves on a small (~6-layer) CNN on CIFAR-10 (a few runs, ~50 epochs, RMSProp) to pick the form;
statistical-significance runs on CIFAR-10 with a fixed small architecture over many (≈20+) seeds
reporting mean accuracy, mean loss, and accuracy std; CIFAR-10 across many standard architectures
(ResNets, Wide ResNets, ShuffleNet, MobileNet, Inception, DenseNet, EfficientNet) swapping only the
activation; ImageNet-1k on ResNets; and object detection on MS-COCO. Diagnostic probes hold the
network fixed and vary one factor: trainability vs. *network depth* on MNIST MLPs; robustness vs.
*input Gaussian-noise* level; sensitivity to *weight-initializer* choice; and direct visualization of
the output landscape and the loss landscape (filter-normalized) for a fixed ResNet-20. Metric: test
accuracy / AP, plus loss-landscape smoothness and minima width as qualitative evidence.

## Code framework

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfGatedActivation(nn.Module):
    """A self-gated pointwise unit: x * h(x) for some smooth scalar gate h.

    Examples in this family:
      Swish/SiLU: h(x) = sigmoid(x)         -> x * sigmoid(x)
      GELU:       h(x) = Phi(x)             -> x * Phi(x)
    Another smooth gate h(x) can be selected from candidate gates and
    analyzed through its first derivative.
    """

    def forward(self, x):
        # TODO: x * h(x), with the selected smooth gate h(x).
        pass


# candidate gates the ablation ranges over (forms similar to the incumbents):
CANDIDATE_GATES = {
    # "name": lambda x: <smooth gate h(x)>,
    # TODO
}
```
