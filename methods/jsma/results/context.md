# Context: crafting targeted, minimally-spread adversarial inputs for feedforward DNNs (circa 2014–2015)

## Research question

A feedforward deep neural network is a function `F: R^M -> R^N` that maps an `M`-dimensional
input `X` (for images, the flattened pixel intensities, each normalized to `[0,1]`) to an
`N`-dimensional output vector, and a classifier reads off `label(X) = argmax_j F_j(X)`. At
test time, after the network is trained and frozen, an adversary holds a benign input `X`
correctly classified as `Y`, and wants to produce a perturbed input `X* = X + δ` that the very
same network assigns to a chosen *target* class `Y* ≠ Y`. The clean way to state the goal is the
constrained program

```
argmin_δ  ‖δ‖   s.t.  F(X + δ) = Y*.
```

The question is which norm `‖·‖` to control, and how to solve the program on a real DNN, where
`F` is non-linear and non-convex so the constraint `F(X+δ)=Y*` has no closed-form inverse. The
specific pressure here is on the *number of input features the adversary has to touch*: a
solution that flips the label by nudging a handful of pixels (each possibly by a large amount)
is qualitatively different from — and for several threat models more dangerous than — one that
spreads a tiny change across the entire image. The adversary has white-box access: the
architecture and trained weights of an acyclic feedforward network with differentiable
activations, but not necessarily the training data. The deliverable is an algorithm that takes
`X`, a target `Y*`, the network `F`, and a budget on how much of the input may be altered, and
returns an `X*` that the network reads as `Y*` while disturbing as little of the input as it can.

## Background

By this time deep neural networks dominate vision, speech, and language, and the security
community has begun asking what happens when a classifier faces an adversary at test time
(evasion), as opposed to a corrupted training set (poisoning). The foundational empirical fact
is that trained DNNs are *not* locally robust: although they generalize well on natural data,
there exist inputs visually indistinguishable from correctly-classified ones that the network
misclassifies with high confidence. These are *adversarial examples*. Two things about them are
load-bearing context.

First, *why they exist*. A trained network learns input–output mappings that are surprisingly
discontinuous: in the immediate neighborhood of a correctly-classified `X`, there are nearby
"pockets" of input space the network labels differently, even though a smoothness prior would
expect tiny perturbations to leave the class unchanged. A widely-held complementary explanation
is linearity. Consider a single linear unit with weights `w` acting on a perturbed input
`x̃ = x + η`:

```
w^T x̃ = w^T x + w^T η.
```

If `η` is bounded in the max norm, `‖η‖_∞ ≤ ε`, the perturbation that maximizes the activation
shift is `η = ε·sign(w)`, giving `w^T η = ε·‖w‖₁ = ε·m·n`, where `m` is the average magnitude of
a weight and `n` the input dimension. The change in the unit's output therefore grows *linearly
with the dimension* `n`: many imperceptible per-feature changes, each below the precision of the
sensor (e.g. below `1/255` of an 8-bit pixel), add up to one large change in the output. So even
shallow, near-linear models on high-dimensional inputs are vulnerable, and a small per-feature
bound is no protection in high dimensions.

Second, *transferability*: an adversarial example crafted for one network is frequently
misclassified by other networks trained with different hyperparameters or even on a disjoint
training set. This suggests adversarial examples reflect properties of the learned function
class rather than quirks of one model, and it underwrites black-box-by-transfer threats.

A separate strand of pre-existing machinery is the use of input-space derivatives to *interpret*
a trained classifier. For a fixed image `I` and a class `c`, one can backpropagate the class
score `S_c` to the input and rearrange the resulting vector `∂S_c/∂I` into an image-shaped map;
taking the per-pixel magnitude (e.g. the max over color channels of `|∂S_c/∂I|`) yields a
"class saliency map" highlighting which pixels most affect the score of class `c`. This was
introduced as a visualization / weakly-supervised-localization tool: it is single-class,
magnitude-only (it discards the sign of the derivative), and it requires only one backward
pass. It says where a class "lives" in an image; it was not built to drive an attack and carries
no notion of a target-versus-rest trade-off.

The general lesson the field has internalized: gradients computed during training, normally used
to update *weights*, can instead be used to update the *input*, because the same backpropagation
machinery differentiates a scalar objective with respect to whatever you like. The methods below
all exploit this, and all of them propagate information from an output-side scalar back to the
input.

## Baselines

**Box-constrained L-BFGS minimum-distortion attack (Szegedy et al. 2013).** The first method to
construct adversarial examples treats the formal program directly. For a classifier `f` with an
associated continuous loss `loss_f`, an input `x`, and a target label `l`, it seeks

```
minimize ‖r‖₂   s.t.   f(x + r) = l   and   x + r ∈ [0,1]^m,
```

approximated by the penalty form: line-search over `c > 0` to find the smallest `c` whose
box-constrained L-BFGS minimizer of `c·‖r‖ + loss_f(x + r, l)` actually satisfies
`f(x+r) = l`. This reliably finds visually-imperceptible adversarial examples and is where
transferability was first observed. *Limitation:* the optimizer runs an iterative,
box-constrained second-order optimization per example, which is computationally heavy; and the
objective is an `L2` magnitude penalty, so the minimizer it returns is a *dense* perturbation —
a small change smeared across essentially every input coordinate. The procedure has no term
that counts, or tries to limit, how many input features are altered.

**Fast Gradient Sign Method (Goodfellow et al. 2014).** Motivated by the linearity view above,
this replaces the per-example optimization with a single closed-form step. Linearize the
training loss `J(θ,x,y)` around the input and take the max-norm-optimal step:

```
η = ε · sign(∇_x J(θ, x, y)),     x* = x + η.
```

One backpropagation pass through the loss yields the gradient; the `sign` makes `η` the optimal
perturbation under an `L_∞ ≤ ε` constraint, and it generalizes the linear `ε·sign(w)` argument to
non-linear nets that "behave linearly enough." It is fast and exposes a wide variety of models.
*Limitation:* the step writes a nonzero perturbation into *every* input dimension at once — by
construction it is the densest possible perturbation under the `L_∞` ball — and it is organized
around the loss gradient and the untargeted "raise the loss" direction rather than around
steering the network into one specific class while touching as little of the input as possible.

**Class-saliency visualization via input gradients (Simonyan et al. 2013).** As described in
Background, this produces a single-class, magnitude-only map `M_ij = max_c |(∂S_c/∂I)_{i,j,c}|`
from one backward pass. *Limitation:* it ranks pixels by how much they affect *one* class's
score in either direction, with no sign and no comparison against the other classes; it is a
visualization, not a procedure for choosing and modifying features to achieve a chosen
misclassification.

Common gap across all three: each maps an output-side scalar (a loss, or one class score) back
to the input via a gradient, and either solves for a dense magnitude-minimizing perturbation or
merely visualizes importance. None of them is built to control how many input features change,
and none provides a concrete procedure for choosing a small set of altered features that pushes
the network into a specific class.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Datasets / networks.** MNIST handwritten digits (28×28 grayscale, flattened to 784 features
  in `[0,1]`, 10 classes) with a LeNet-style convolutional network (two conv+pool stages, a
  fully-connected hidden layer, a 10-way softmax). Image classifiers on CIFAR-style inputs
  (`C×H×W`, channels in `[0,1]`) are the other standard small-image testbed.
- **Threat model.** White-box, test-time, targeted: full access to architecture and weights;
  the network is fixed; only the input is altered, after training. Inputs must stay in the valid
  range `[0,1]`.
- **Metrics.** Attack success — the fraction of (initially correctly-classified) inputs that are
  pushed to the adversary's target — and *distortion*, here measured as how much of the input is
  changed: the number (or percentage) of input features altered. A pixel/feature counts as
  modified if it differs from the original. (For multi-channel inputs, a spatial pixel counts as
  modified if any of its channels changes.)
- **Protocol.** Collect inputs the model classifies correctly; run the attack with a fixed
  budget on the amount of allowed distortion; check validity (output in range, distortion within
  budget) and whether the network now outputs the target; an attack that exceeds the budget or
  emits an invalid input is a failure.

## Code framework

The attack plugs into a standard white-box harness: a frozen, differentiable classifier and a
batch of clean inputs in `[0,1]`, and it must return adversarial inputs of the same shape, in
the same range, under a budget on changed input features. The substrate that
already exists is automatic differentiation (we can backpropagate any scalar built from the
network's outputs to the input), an `argmax` read-out for the predicted label, and clamping to
the valid range. What is *not* given — the contribution to be designed — is the rule that
decides, for a given input and target, which feature(s) to change and by how much, iterated
until the network flips or the budget is spent.

```python
import torch
import torch.nn as nn


def predicted_label(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """argmax over the network outputs — the class the classifier reports."""
    return model(x).argmax(dim=1)


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,   # (N,) true labels
    pixels: int,            # budget: number of feature changes allowed per image
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """Return adversarial images, same shape, in [0,1], with a budgeted number of features
    changed per image, that the frozen `model` misclassifies."""
    model.eval()
    adv = images.clone().to(device)

    for k in range(images.shape[0]):
        x = adv[k : k + 1]                      # one image; AD lets us differentiate model(x)
        # budget on how many feature-modifying steps we may take
        # TODO: the per-input procedure we will design.
        #   Using only forward/backward passes through `model` and the valid range [0,1],
        #   decide which feature(s) of x to modify and by how much, and iterate until the
        #   network's predicted_label(model, x) changes or the feature budget is exhausted.
        pass

    return adv.clamp(0.0, 1.0)
```

The harness supplies the frozen model, the clean batch, the budget, and the range constraint;
the single empty slot is the feature-selection-and-modification rule that turns a clean input
into a budget-respecting adversarial one.
