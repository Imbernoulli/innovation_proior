# Context: adversarial perturbations under an extreme sparsity constraint (circa 2016-2017)

## Research question

A deep image classifier `f` maps an `n`-pixel image `x = (x_1, ..., x_n)` to class
probabilities, and `f_t(x)` is the probability it assigns to the true class `t`. It is by now
well established that one can add a small perturbation `e(x)` to a correctly classified `x` and
flip the prediction. The standard way to do this constrains the *magnitude* of `e(x)` — keep
`||e(x)||_p` small under an `L2` or `L-inf` budget — and lets the perturbation touch as many
pixels as it likes. The question is what happens when the budget is instead expressed as a
*count* of modified spatial pixels (`L0`), and in particular how the attack behaves in the
extreme low-pixel regime.

## Background

By 2016-2017 the adversarial-examples literature is young but fast-moving, and a few facts
about the input geometry of deep classifiers are established and load-bearing here.

- **Imperceptible perturbations exist and are easy to find.** Szegedy et al. (2014) first
  showed that an `x` and a near-identical `x + r` can receive different labels, and that such
  `r` are not rare flukes.
- **The decision boundaries sit very close to natural images, and along few directions.**
  Fawzi, Moosavi-Dezfooli et al. (2017) report a curvature analysis: around a natural image the
  space is *flat along most directions* and curved (sensitive) along only a few; most data
  points sit near a boundary. Moosavi-Dezfooli et al. (2017) show a *single* image-agnostic
  perturbation can fool a network on most images, evidence that boundary shapes near different
  points are similar and the boundary "diversity" is low.
- **The relevant budget is contested.** `Lp` norms are a convenient but imperfect proxy for
  perceptibility; counting modified pixels (`L0`) is a more direct measure of how localized a
  change is. An `L0` budget makes the support of the perturbation — *which* pixels — part of
  the optimization, which is a discrete, combinatorial choice on top of the continuous choice
  of how much to change them.
- **Population-based optimization for hard objectives.** Differential evolution (Storn & Price,
  J. Global Optimization 1997) is a metaheuristic for nonlinear, non-differentiable, multimodal
  objectives over continuous spaces. It maintains a population of candidate vectors and, each
  generation, perturbs each member by a *scaled difference of two other members*, then keeps the
  better of parent and child. The differential step self-scales to the current spread of the
  population, and the one-to-one parent/child selection preserves diversity; together these make
  it comparatively robust to local minima while needing nothing but the ability to *evaluate*
  the objective.

## Baselines

The prior attacks a sparse, black-box attack would be measured against and reacts to.

**Box-constrained L-BFGS (Szegedy et al., ICLR 2014).** Minimize `||r||_2` subject to
`f(x + r) = target` and `x + r` in `[0,1]^n`, solved with box-constrained L-BFGS. It produced
the first adversarial examples and requires white-box access to compute gradients. The
perturbation is spread over all pixels under an `L2` norm.

**Fast Gradient Sign Method (Goodfellow, Shlens & Szegedy, ICLR 2015).** A single step along
the sign of the loss gradient, `x + eps * sign(grad_x J)`, motivated by the hypothesis that the
linearity of high-dimensional models is what makes them fragile. It is white-box, `L-inf`
bounded, and nudges every pixel by `eps`.

**DeepFool (Moosavi-Dezfooli, Fawzi & Frossard, CVPR 2016).** Iteratively linearize the
decision boundary and take the minimal step to the nearest linearized boundary, yielding a small
`L2` perturbation and an estimate of the distance to the boundary. It is white-box and dense
(`L2`).

**Jacobian Saliency Map Attack (Papernot et al., IEEE EuroS&P 2016).** Build an adversarial
saliency map from the forward derivatives (the Jacobian of the network outputs), then greedily
perturb the few highest-saliency pixels to drive the target-class score up, a pixel or two at a
time. It is a sparse, `L0`-flavored attack that uses the Jacobian. In practice it modifies on
the order of a few percent of the pixels (e.g. ~4% of a 32x32 image, ~40 pixels).

**Universal adversarial perturbations (Moosavi-Dezfooli et al., CVPR 2017).** A single
image-agnostic vector that, added to almost any natural image, fools the network. It is
white-box to construct and modifies every pixel, providing a contrasting geometric picture to
sparse attacks.

**Simple black-box local search (Narodytska & Kasiviswanathan, CVPR Workshops 2017).** A
score-based local search that uses one-pixel changes as a starting point for a greedy local
search that ultimately perturbs many pixels (around 30 of 1024).

**Random sparse search.** The obvious non-learning control: repeatedly pick a random pixel and
random value, keep whatever changes the label, report the best over a fixed number of tries.

## Evaluation settings

The natural yardsticks already in use for adversarial robustness.

- **CIFAR-10** (32x32 RGB, 10 classes), both the original test set and the noisier "Kaggle
  CIFAR-10" variant (which contains random duplication / rotation / blur / bad-pixel noise and so
  simulates real-world corruption). Fixed random subsets of test images are drawn per network for
  attack, with labels handled according to the targeted or non-targeted protocol.
- **ImageNet (ILSVRC 2012)**, images converted to lossless PNG and resized to 227x227 for input
  to AlexNet — about 50x more spatial locations than 32x32 CIFAR-10, to test whether a tiny
  modification still works at scale.
- **Target networks**: All-Convolutional network, Network-in-Network, VGG16, and BVLC AlexNet,
  trained to standard accuracy.
- **Threat models**: targeted (perturb toward a chosen class) and non-targeted (away from the
  true class). For black-box attacks, only the output probability vector is observed.
- **Metrics**: success rate (targeted = probability of reaching a chosen class; non-targeted = %
  of images perturbed to *any* wrong class), average confidence of the wrong prediction, the
  number of distinct target classes a given image can be pushed to, original-target class-pair
  frequency (heatmaps), and the cost in number of model evaluations and in average per-channel
  pixel distortion.
- **Budgets**: fixed small numbers of modified spatial pixels, including the one-pixel extreme
  and small few-pixel comparisons.

## Code framework

The attack plugs into a standard score-based-attack harness: a wrapped classifier exposing only
a forward pass to probabilities, and a generic black-box search routine that takes a real-valued
objective, box bounds on each coordinate, an optional early-stop callback, and an
iteration/population budget. The open design slots are how to represent a sparse perturbation as a
vector the search routine can handle, how to score a candidate from output probabilities, and how
to turn an optimized vector back into a perturbed image.

```python
import numpy as np
import torch
import torch.nn.functional as F

# A generic black-box search primitive already exists: it proposes real-valued vectors for
# `func`, respects per-coordinate `bounds`, and can stop early when `callback` signals
# success. It queries `func` only by value (no gradients).
from black_box_optimizer import population_search


class ScoreBasedAttack:
    """Harness for an attack that sees only the model's output probabilities."""

    def __init__(self, model, pixels, popsize, steps, inf_batch=128, targeted=False):
        self.model = model
        self.pixels = pixels          # the L0 budget: number of pixels allowed to change
        self.popsize = popsize
        self.steps = steps
        self.inf_batch = inf_batch
        self.targeted = targeted

    def _get_prob(self, images):
        # batched forward pass -> softmax probabilities, the only model access we get
        with torch.no_grad():
            outs = [self.model(b) for b in torch.split(images, self.inf_batch)]
        return F.softmax(torch.cat(outs), dim=1).detach().cpu().numpy()

    def _encode_bounds(self, image):
        # TODO: the box bounds for one searchable perturbation vector — what the optimizer
        #       evolves. (How a sparse perturbation is represented is to be designed.)
        raise NotImplementedError

    def _decode(self, image, vector):
        # TODO: turn an optimized vector back into a perturbed image.
        raise NotImplementedError

    def _objective(self, image, label, vector):
        # TODO: score a candidate vector from the model's probabilities.
        raise NotImplementedError

    def forward(self, images, labels):
        adv = []
        for i in range(len(images)):
            image, label = images[i:i+1], labels[i:i+1]
            bounds = self._encode_bounds(image)
            # TODO: search for a perturbation with the population-based optimizer,
            #       then decode the best vector into the adversarial image.
            best = population_search(
                func=lambda v: self._objective(image, label, v),
                bounds=bounds,
                maxiter=self.steps,
                popsize=self.popsize,
                callback=None,   # TODO: early stop on success
            ).x
            adv.append(self._decode(image, best))
        return torch.cat(adv)
```

The three open slots are the representation, the probability-based score, and the decode, wrapped by
the generic black-box search primitive that already exists.
