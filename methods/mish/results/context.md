# Context

## Research question

Given that self-gated, smooth, non-monotonic activations (`x·σ(βx)`, `x·Φ(x)`) have become strong ReLU
alternatives on deep networks, the open question is *which precise pointwise curve in that neighborhood
should be used, and why*. The desired object is a single pointwise nonlinearity that (i) keeps the properties
that make self-gated units work — smoothness everywhere, a small preserved negative region,
unboundedness above, boundedness below — (ii) remains stable as networks get deep and complex, and
(iii) comes with a mechanistic account of *why* its first-derivative behavior should ease optimization,
not just a benchmark table. It must remain a drop-in replacement: swap it for ReLU/Swish with no other
change to architecture or training.

## Background

The activation lineage relevant here: sigmoid/tanh (smooth, bounded, but saturating so gradients
vanish in deep stacks) → ReLU `max(x,0)` (Nair & Hinton 2010; unsaturated on the positive half, fast,
but with the "dying ReLU" failure where negative inputs collapse to zero output *and* zero gradient,
plus a non-differentiable kink) → a wave of fixes (Leaky ReLU, ELU, SELU) → the self-gated units.
The self-gated unit multiplies the raw input by a smooth nonlinear function of itself: SiLU/Swish
`x·σ(βx)` (Elfwing et al. 2017; Ramachandran et al. 2017) and GELU `x·Φ(x)` (Hendrycks & Gimpel 2016).
What these share, and what is believed to matter, is a distinctive concavity for negative inputs that
*preserves a small amount of negative signal* (instead of zeroing it like ReLU), a smooth continuous
profile (no kink), unboundedness above (no positive saturation), and boundedness below.

Two threads of prevailing wisdom set up the problem. First, *smoothness of the output/loss landscape*
is associated with easier optimization and better generalization: smoother per-unit maps produce
smoother output landscapes, which correlate with smoother, wider-minima loss surfaces. Second, in
classical optimization a *preconditioner* — for gradient descent, an inverse-Hessian-like operator
applied to reshape the objective's geometry — accelerates convergence by making the landscape locally
more isotropic. These two ideas (smoothness; preconditioning) are the conceptual tools a new activation
could be analyzed with. Empirically, deeper plain (non-residual) networks degrade in trainability past
some depth, and activations differ in how far that degradation can be pushed back — a diagnostic that
distinguishes candidate curves.

## Baselines

- **ReLU** `max(x,0)`: cheap, sparse, derivative 1 for `x>0`. Gap: dying units (zero output and gradient
  on the negative half), non-differentiable kink, no curvature, monotonic.
- **Leaky ReLU** `max(x, αx)` (`α=0.01`): a constant leak on the negative side. Gap: still
  piecewise-linear, monotonic, kinked.
- **Swish / SiLU** `x·σ(βx)` (`β=1` ≈ SiLU): smooth, non-monotonic, self-gated; small negative bump;
  unbounded above, bounded below. The strongest incumbent and the direct reference point. Gap: it is
  *one* particular curve in the self-gated family — whether a different smooth gate trains more stably
  and to higher accuracy on deep/complex nets is exactly the open question.
- **GELU** `x·Φ(x)` (Hendrycks & Gimpel 2016): smooth, non-monotonic, Gaussian-gated; very similar
  shape to Swish. Gap: like Swish, a single fixed curve in the family; the "best curve and why"
  question is unsettled.
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
