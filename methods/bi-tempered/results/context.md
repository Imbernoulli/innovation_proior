## Research question

Softmax cross-entropy — the logistic loss — is the standard training objective for neural-net classifiers. It is convex in the activations of the last layer, and the softmax maps activations to a probability vector through an exponential. The construction has two pieces: a transfer function from activations to probabilities, and a divergence that scores those probabilities against the target. The question is how to choose this pair when the training labels are not perfectly clean — in particular, whether the softmax-and-log construction can be generalized, without abandoning the properties that make it work, so that the loss still behaves as a *proper* loss whose minimizer recovers the true class posterior.

## Background

**The logistic loss as a matching loss.** For activations `â ∈ ℝ^k` over `k` classes, the softmax `ŷ_i = exp(â_i)/Σ_j exp(â_j)` maps activations to a probability vector, and the loss is the relative entropy (KL divergence) between the one-hot target `y` and `ŷ`. There is a clean reason these two pieces go together: the softmax is the gradient of the convex dual of the negative entropy `F(y) = Σ_i (y_i log y_i − y_i)`, and pairing a transfer function with the Bregman divergence induced by the corresponding convex function gives a "matching loss" (Auer et al.; Helmbold et al.) that is provably convex in the activations. The logistic loss is exactly the matching loss for the softmax: the two pieces are dual halves of a single convex object.

**Bregman divergences.** For a strictly convex `F`, the Bregman divergence is `Δ_F(y, ŷ) = F(y) − F(ŷ) − (y − ŷ)·∇F(ŷ)`; it is nonnegative, zero iff `y = ŷ`, convex in its first argument, and invariant to adding affine terms to `F`. Squared Euclidean distance (`F = ½‖·‖²`) and KL divergence (`F` = negative entropy) are both Bregman divergences. The negative-entropy/KL/softmax triple is the special case the logistic loss uses.

**Behavior of convex, light-tailed losses under label noise.** Convex potential losses are non-robust to label noise (Long & Servedio): the per-example loss grows without bound as the activation moves the wrong way. The softmax tail decays exponentially, so probabilities saturate quickly toward 0/1. Heavy-tailed alternatives to the softmax have been studied as a way to soften the probability assignment near the decision boundary.

**Tempered logarithm and exponential.** A temperature-deformed logarithm (Naudts; Tsallis statistics) is `log_t(x) = (x^{1−t} − 1)/(1 − t)`, monotonically increasing and concave, recovering `log` as `t → 1`. For `0 ≤ t < 1` it is *bounded below* by `−1/(1−t)`. Its inverse is the tempered exponential `exp_t(x) = [1 + (1−t)x]_+^{1/(1−t)}` (with `[·]_+ = max(·,0)`), recovering `exp` as `t → 1`; for `t > 1` it has a heavier (polynomial) tail than `exp`. These are standard one-parameter deformations of the elementary `log`/`exp`.

**Diagnostic illustration on synthetic 2-D data.** On a two-dimensional binary problem with a small feed-forward net, two kinds of label corruption are studied. Small-margin label flips place noisy points near the clean boundary; large-margin flips place mislabeled points far from the clean boundary; random label noise mixes both regimes. Visualizing the fitted boundary under each regime shows how the trained classifier responds to each kind of corruption.

## Baselines

**Logistic / softmax cross-entropy.** `Δ_F(y, softmax(â))` with `F` the negative entropy; the matching loss for the softmax, convex in the activations, minimizer is the true posterior.

**Label smoothing (Szegedy et al., 2016).** Replaces the one-hot target with `(1−ε) one-hot + ε/k`, putting a floor under every class so the correct logit has no incentive to run to `+∞`. Bounds overconfidence and helps generalization and calibration. It is a static modification of the *target*, identical for every example and class.

**Focal loss (Lin et al., 2017).** `−(1−P_t)^γ log(P_t)`; down-weights easy (well-classified) examples to address class imbalance. It reshapes the loss by confidence while keeping the standard softmax and the `log`.

**Mean absolute error / truncated cross-entropy (Ghosh et al., 2017; Feng et al., 2020).** MAE on the softmax probabilities is bounded and provably noise-robust; truncating the Taylor series of cross-entropy interpolates between cross-entropy and MAE.

**Tsallis-divergence two-temperature loss (Amid et al., 2018).** Generalizes logistic regression with two temperatures via the Tsallis divergence, containing earlier tempered variants as special cases. Because `log_t(a/b) ≠ log_t(a) − log_t(b)` in general, the natural Monte-Carlo estimator of the Tsallis divergence references the conditional `P(y|x)`.

## Evaluation settings

The natural yardsticks are standard image-classification datasets with *controlled* label corruption, plus large-scale clean classification. For moderate scale: MNIST (CNN) and CIFAR-100 (ResNet-56), each with a fraction of training labels artificially flipped at noise levels from 0 up to 0.5, reporting top-1 accuracy on a *clean* test set and selecting checkpoints on an identically-corrupted validation set. For large scale: ImageNet-2012 with ResNet-18 and ResNet-50, where the labels are inherently somewhat noisy, reporting top-1 accuracy. Optimizers, schedules, and augmentation are each model's standard recipe, left unchanged so the loss is the only variable; any loss-specific knobs are selected on validation data. Higher accuracy is better.

## Code framework

The available primitives are the raw activations (logits) over `k` classes, integer or one-hot targets, the standard softmax, and the standard cross-entropy criterion. What is open is the transfer function that maps activations to probabilities and the divergence that scores them against the target.

```python
import torch
import torch.nn.functional as F

# logits:  [B, k] raw activations (pre-softmax)
# targets: [B] integer labels, or [B, k] one-hot / soft targets
# F.softmax(logits, dim=-1)              -> standard (light-tailed) probabilities
# F.cross_entropy(logits, targets)       -> KL(one_hot, softmax), convex & unbounded


def transfer_function(activations):
    """Map activations to a probability vector over classes.

    The standard choice is the softmax. The open question is whether a
    heavy-tailed alternative is better for noisy data, and how to normalize it.
    """
    # TODO: choose the activation -> probability map (and its normalization)
    raise NotImplementedError


def classification_loss(activations, targets):
    """Score the predicted probabilities against the target.

    The standard choice is the relative entropy (KL), giving the convex
    logistic loss. The open question is which divergence to use and how it
    relates to the choice of transfer function.
    """
    # TODO: choose the divergence between target and predicted probabilities
    raise NotImplementedError
```
