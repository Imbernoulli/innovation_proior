## Research question

Softmax cross-entropy — the logistic loss — is the standard training objective for neural-net classifiers. It is convex in the activations of the last layer, which is comforting from an optimization standpoint. But that same convexity, together with the exponential tail of the softmax, makes the loss fragile in two specific ways when the training labels are not perfectly clean. A large-margin mislabeled example (one the model is confident about, far from the boundary) incurs an unboundedly large loss and remains a persistent source of pressure on the fitted boundary. And a small-margin mislabeled example (one sitting near the boundary) is chased by the short-tailed softmax, which saturates its probability toward 0/1 and forces the classifier to fit it. The question is whether the softmax-and-log construction can be generalized — without abandoning the things that make it work — so that the loss is *bounded* (capping the influence of large-margin outliers) and the probability assignment is *heavy-tailed* (refusing to chase small-margin noise), while still being a *proper* loss whose minimizer recovers the true class posterior.

## Background

**The logistic loss as a matching loss.** For activations `â ∈ ℝ^k` over `k` classes, the softmax `ŷ_i = exp(â_i)/Σ_j exp(â_j)` maps activations to a probability vector, and the loss is the relative entropy (KL divergence) between the one-hot target `y` and `ŷ`. There is a clean reason these two pieces go together: the softmax is the gradient of the convex dual of the negative entropy `F(y) = Σ_i (y_i log y_i − y_i)`, and pairing a transfer function with the Bregman divergence induced by the corresponding convex function gives a "matching loss" (Auer et al.; Helmbold et al.) that is provably convex in the activations. The logistic loss is exactly the matching loss for the softmax. This duality is the structural fact any generalization should try to preserve.

**Bregman divergences.** For a strictly convex `F`, the Bregman divergence is `Δ_F(y, ŷ) = F(y) − F(ŷ) − (y − ŷ)·∇F(ŷ)`; it is nonnegative, zero iff `y = ŷ`, convex in its first argument, and invariant to adding affine terms to `F`. Squared Euclidean distance (`F = ½‖·‖²`) and KL divergence (`F` = negative entropy) are both Bregman divergences. The negative-entropy/KL/softmax triple is the special case the logistic loss uses.

**Why convex, light-tailed losses are fragile.** Convex potential losses are known to be non-robust to label noise (Long & Servedio): because the per-example loss grows without bound as the activation moves the wrong way, a handful of mislabeled large-margin points can dominate the empirical risk and bend the learned boundary. Separately, the softmax tail decays exponentially, so probabilities saturate quickly; a mislabeled point near the boundary therefore gets assigned a near-extreme probability and the classifier must distort itself to satisfy it. Heavy-tailed alternatives to the softmax have been shown to soften this. These two failure modes — large-margin outliers and small-margin boundary noise — are distinct and call for distinct fixes.

**Tempered logarithm and exponential.** A temperature-deformed logarithm (Naudts; Tsallis statistics) is `log_t(x) = (x^{1−t} − 1)/(1 − t)`, monotonically increasing and concave, recovering `log` as `t → 1`. Crucially, for `0 ≤ t < 1` it is *bounded below* by `−1/(1−t)`. Its inverse is the tempered exponential `exp_t(x) = [1 + (1−t)x]_+^{1/(1−t)}` (with `[·]_+ = max(·,0)`), recovering `exp` as `t → 1`; for `t > 1` it has a *heavier* (polynomial) tail than `exp`. These two deformations are the raw material: the bounded-below `log_t` can make a loss bounded, and the heavy-tailed `exp_t` can make a softmax that does not saturate.

**Diagnostic illustration on synthetic 2-D data.** On a two-dimensional binary problem with a small feed-forward net, small-margin label flips expose the light-tail problem: the logistic loss stretches the boundary toward noisy points near the clean boundary. Large-margin flips expose the convex unbounded-loss problem: mislabeled points far from the clean boundary pull the classifier strongly despite being implausible under the rest of the data. Random label noise mixes both regimes. The diagnostic lesson is the separation itself: the probability assignment needs a tail cure, and the per-example loss needs a boundedness cure.

## Baselines

**Logistic / softmax cross-entropy.** `Δ_F(y, softmax(â))` with `F` the negative entropy; the matching loss for the softmax, convex in the activations, minimizer is the true posterior. Gap: convex and unbounded in the activations (large-margin outliers dominate), and the softmax tail is exponentially light (small-margin boundary noise is chased). Robust only when labels are clean.

**Label smoothing (Szegedy et al., 2016).** Replaces the one-hot target with `(1−ε) one-hot + ε/k`, putting a floor under every class so the correct logit has no incentive to run to `+∞`. Bounds *overconfidence* and helps generalization and calibration. Gap: it is a static modification of the *target*, identical for every example and class; the loss is still convex and unbounded in the activations and the softmax is still light-tailed, so it does not address the two outlier/noise failure modes at their source, and it cannot adapt to how wrong or how confident a given example is.

**Focal loss (Lin et al., 2017).** `−(1−P_t)^γ log(P_t)`; down-weights easy (well-classified) examples to fight class imbalance. Gap: it reshapes the loss by confidence but is still built on the standard softmax and an unbounded `log`, so it does not bound the loss for large-margin outliers nor heavy-tail the probabilities; it targets imbalance, not label noise.

**Mean absolute error / truncated cross-entropy (Ghosh et al., 2017; Feng et al., 2020).** MAE on the softmax probabilities is bounded and provably noise-robust; truncating the Taylor series of cross-entropy interpolates between cross-entropy and MAE. Gap: MAE-like losses are bounded but train slowly and under-fit on hard, many-class problems, and they do not touch the light-tailed softmax.

**Tsallis-divergence two-temperature loss (Amid et al., 2018).** Generalizes logistic regression with two temperatures via the Tsallis divergence, containing earlier tempered variants as special cases. Gap: the Tsallis-based construction does *not* yield a *proper* loss — because `log_t(a/b) ≠ log_t(a) − log_t(b)` in general, the natural Monte-Carlo estimator requires access to the unknown conditional `P(y|x)`, so approximating it by `1` gives a biased estimator. Properness — that minimizing the expected loss recovers the true posterior — is required for real applications, and this construction lacks it.

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

    The standard choice is the relative entropy (KL), giving the convex,
    unbounded logistic loss. The open question is which divergence bounds the
    per-example loss while remaining a proper loss.
    """
    # TODO: choose the divergence between target and predicted probabilities
    raise NotImplementedError
```
