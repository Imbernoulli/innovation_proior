# Context: differentially private deep learning and the cost of the clipping threshold (circa 2021-2022)

## Research question

Training a deep network with a formal `(epsilon, delta)`-differential-privacy guarantee means the
optimizer never sees the raw minibatch gradient: it sees a *privatized* gradient, produced by
bounding the contribution of each individual training example and then adding calibrated Gaussian
noise. Concretely, the per-sample gradients `g_i` are each forced to have bounded `L2` norm (so one
example can move the aggregate by only so much — this is the *sensitivity*), they are summed, and
Gaussian noise scaled to that sensitivity is added before the optimizer steps. Two knobs control this
that the non-private optimizer does not have: the noise multiplier `sigma`, and the bound on each
per-sample gradient — the *clipping threshold* `R`.

The two knobs are not equally cheap. `sigma` can be fixed *before training ever starts*: once the
privacy budget `(epsilon, delta)`, the subsampling probability `p = B/n`, and the number of
iterations `T` are chosen, an off-the-shelf privacy accountant returns the exact `sigma` that spends
that budget — no tuning. `R` is the opposite. It cannot be read off the privacy budget; it has to be
searched for, and the model's accuracy is brutally sensitive to it. So where regular training needs
only a one-dimensional grid search over the learning rate `eta`, private training needs a
two-dimensional search over `(R, eta)`. On large models that 2D search is the dominant cost — days on
a GPT2-scale model, and it grows worse if one wants a different threshold per layer. The precise goal:
make private training as cheap to tune as regular training, without spending a separate search over
`R`, while keeping (1) the *same* privacy guarantee and (2) a convergence rate no worse than ordinary
SGD.

## Background

By this time differentially private SGD is the standard recipe for private deep learning, and it is
known to defend against the concrete attacks that motivate privacy at all: membership-inference
attacks that recover whether a record was in the training set, and verbatim-memorization attacks in
which a language model auto-completes someone's name, phone number, or address from the training
corpus. The load-bearing primitive is the *subsampled Gaussian mechanism*: include each datapoint in
the minibatch i.i.d. with probability `p`, bound the `L2` sensitivity of the per-sample gradients to
some value, add Gaussian noise proportional to that sensitivity, and account the privacy loss across
`T` iterations with a modern accountant (Renyi/moments accounting, Dwork & Roth; the privacy-loss-
distribution / Fourier accountants; or Gaussian-DP, Dong et al. 2019, under which subsampled DP-SGD
converges to a clean `mu`-GDP guarantee with `mu = (B/n) sqrt(T (e^{1/sigma^2} - 1))`, Bu et al.
2020). The single fact that drives everything below: the Gaussian mechanism's privacy depends only on
the *ratio* of noise to sensitivity. If the per-sample contribution is bounded to `Delta` and the
noise added has standard deviation `sigma * Delta`, the noise-to-sensitivity ratio is exactly `sigma`
no matter what `Delta` is — so the privacy guarantee is a function of `sigma`, `p`, `T` alone, and is
blind to the actual value of the bound.

There is a sharp, well-documented empirical regularity about *where the good private models live*:
the state of the art is consistently obtained with a **small** clipping threshold. On the E2E and
classification benchmarks, the best private GPT2 and RoBERTa results use a per-sample bound of `0.1`
(Li et al. 2021); on ImageNet the best private ResNets and Vision Transformers use `1` (Kurakin et
al. 2022; De et al. 2022; Mehta et al. 2022); on CIFAR-10 the best results use `0.1` (Tramer &
Boneh 2020). And the sensitivity to `R` is dramatic, not gentle: reproductions show a ResNet18 on
ImageNet falling from `45%` to `31%` accuracy when `R` is doubled, and to `0.1%` when it is
quadrupled — the same steep cliff appears even with the noise turned off, so it is the *clipping*, not
the noise, that the accuracy is reacting to. A second documented phenomenon: in this small-threshold
regime the clip is *active* on most examples — for the GPT2 generation runs essentially `100%` of
per-sample gradients are clipped at every iteration, and on the classification tasks roughly
`20-60%`.

Clipping also has a known optimization failure shape. Bounding each per-sample gradient introduces a
bias; when per-sample directions from different classes disagree, the aggregated clipped gradient can
be near zero even when the true (unclipped) gradient is far from zero, so the optimizer sits still at
a non-stationary point. This "stuck even though there is descent available" behavior has been
characterized geometrically for clipped DP gradient descent (Chen et al. 2020; Song et al. 2021 for
the generalized-linear-model case), including explicit small examples (a balanced binary logistic
regression, a Gaussian-mixture mean-estimation problem) where the clipped update vanishes on a whole
interval of parameter values.

The convergence yardstick is the standard non-convex SGD analysis (Ghadimi & Lan 2013; Bottou et al.
2018): under a lower-bounded loss, `L`-smoothness, and per-sample gradient noise with mean zero and
bounded variance `xi^2`, plain SGD with step `eta ~ 1/sqrt(T)` drives `min_t E||g_t||` to zero at
rate `O(T^{-1/4})`. A frequently-used strengthening in that literature — and one verified empirically
for these gradients (Chen et al. 2020, Fig. 3) — is that the per-sample gradient noise is *centrally
symmetric* about the true gradient: `tilde g - g` and `g - tilde g` have the same distribution.

## Baselines

These are the prior approaches a new method is measured against, with the specific limitation each
leaves open.

**Abadi's DP-SGD (Abadi et al., CCS 2016).** The standard mechanism. Clip each per-sample gradient to
norm `R`, sum, add Gaussian noise scaled to `R`, step:
```
ghat_i = g_i / max(1, ||g_i|| / R)        # = g_i * min(R/||g_i||, 1), so ||ghat_i|| <= R
gtilde  = sum_i ghat_i + N(0, sigma^2 R^2 I)
w_{t+1} = w_t - eta * gtilde
```
The sensitivity is `R`; privacy is accounted with the moments accountant. **Gap:** two DP-specific
knobs, `(R, eta)`, where `R` is not derivable from the privacy budget and the accuracy is extremely
sensitive to it, forcing an expensive 2D grid search that dominates the tuning cost at scale.

**Adaptive (quantile) clipping (Andrew et al., NeurIPS 2021).** Rather than fix `R`, track a chosen
quantile of the per-sample-norm distribution online and privately. For a target quantile `q`, the
pinball loss `l(C; X) = (1-q)(C - X)` if `X <= C` else `q(X - C)` has its minimizer at the `q`-th
quantile of `X`; one descends it online from a privately-estimated noisy fraction `bbar` of
per-sample norms below the current threshold, `C <- C - eta_C (bbar - q)` (a geometric variant in
practice, `eta_C = 0.2`). The threshold tracks the moving gradient scale and spends only a small
slice of privacy budget. **Gap:** it removes the need to tune `R` only by *introducing* new things to
tune — the target quantile `q` and the split of the privacy budget between estimating the quantile
and perturbing the gradient — so DP-specific hyperparameters remain.

**Re-parameterized clipping (De et al., 2022).** Use `Clip(g_i) = min(1/R, 1/||g_i||)`, which is
Abadi's clip under a re-scaled learning rate. Observed to make accuracy *less* sensitive to `R`.
**Gap:** `R` is not actually decoupled — it still appears in *both* branches of the `min`, so doubling
`R` is not equivalent to doubling `eta`, the threshold still has to be tuned, the decoupling claim is
demonstrated only for non-adaptive DP-SGD, and shrinking `R` silently rescales the weight decay
(`lambda` effectively becomes `lambda/R`), which can itself hurt accuracy.

**Global clipping (Bu et al., 2021).** Keep a per-sample gradient unchanged if its norm is below `R`
and drop it entirely otherwise, `Clip(g_i) = I(||g_i|| < R)`, to reduce the clipping bias and the
mis-calibration of DP classifiers. **Gap:** it discards the large-norm samples outright and still
carries the threshold `R`.

Across all of these the threshold survives in one form or another, and none comes with a non-convex
convergence guarantee under the same smoothness assumptions used for ordinary SGD.

## Evaluation settings

The natural yardsticks already in use for private deep learning:

- **Image classification.** MNIST and Fashion-MNIST (28x28 grayscale, 10 classes) trained from
  scratch with a small CNN and DP-SGD with momentum; CIFAR-10 (32x32 color, 10 classes) on top of
  features from a SimCLRv2 model pretrained on unlabeled ImageNet; the ImageNette 10-class subset of
  ImageNet; and CelebA (high-resolution faces, 40 attributes) for single-attribute and multi-label
  classification. Group normalization replaces batch normalization where a normalization layer is
  needed, since batch statistics break per-sample gradients.
- **Sentence classification.** RoBERTa-base/large fine-tuned with DP-Adam on SST-2, QNLI, MNLI, QQP.
- **Table-to-text generation.** GPT2 fine-tuned with DP-AdamW on the E2E and DART datasets,
  scored by BLEU.
- **Protocol.** Privacy is calibrated with an RDP accountant: fix `(epsilon, delta)` and let the
  accountant return `sigma` given `(p, T)`. The metric is test accuracy (or BLEU) at a fixed budget,
  with the privacy spent reported alongside. For the small-model setting used here: MNIST,
  Fashion-MNIST, and CIFAR-10 at `epsilon = 3`, `delta = 1e-5`, metric test accuracy.

## Code framework

The mechanism plugs into the existing DP-SGD harness. Everything outside the per-sample-gradient
processing is fixed and already standard: a way to obtain *per-sample* gradients (a tensor of shape
`[B, *param_shape]` per parameter, from a per-sample-gradient backend such as Opacus or a
ghost-clipping library), a privacy accountant that converts `(epsilon, delta, p, T)` into the noise
multiplier `sigma`, and an ordinary optimizer/training loop that consumes one aggregated, privatized
gradient per step. The unresolved design question is how each per-sample gradient is mapped into a
bounded-sensitivity, summable quantity before noise is added. That is the single empty slot below.

```python
import torch


class DPMechanism:
    """Turns per-sample gradients into one privatized aggregate gradient.

    Fixed around this: the per-sample-gradient backend, the privacy accountant
    that fixes `sigma` from (epsilon, delta, p, T), and the optimizer/training loop.
    Open: how each per-sample gradient is processed so that the aggregate has
    bounded L2 sensitivity (one example can change it by a known amount), which is
    what makes adding sensitivity-scaled Gaussian noise a valid DP mechanism.
    """

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm        # the per-sample bound knob, if used
        self.noise_multiplier = noise_multiplier  # sigma, fixed by the accountant
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

    def clip_and_noise(self, per_sample_grads, step, epoch):
        # per_sample_grads: list of tensors [B, *param_shape]
        # returns: list of privatized aggregated gradients [*param_shape]
        # TODO: choose the per-sample gradient processing map from g_i to
        #       a bounded-sensitivity contribution, sum, and add noise scaled to that
        #       sensitivity. The exact map is the open question.
        raise NotImplementedError

    def get_effective_sigma(self, step, epoch):
        # the noise multiplier handed to the privacy accountant
        return self.noise_multiplier
```

The training loop draws a Poisson-subsampled batch, computes per-sample gradients, calls
`clip_and_noise`, and hands the privatized gradient to the optimizer; `get_effective_sigma` feeds the
accountant. Any new rule has to fill exactly that one `clip_and_noise` slot.
