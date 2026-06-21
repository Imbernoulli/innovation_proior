# Context: anti-collapse regularization for joint-embedding self-supervised learning (circa 2024-2025)

## Research question

We want to train an encoder `f_theta: R^D -> R^K` purely from unlabeled data so that its
embeddings are a good foundation for *unknown* downstream tasks. The dominant recipe is
joint-embedding self-supervised learning: form two semantically related *views* of the same
input (two augmentations of an image, two frames of a video), push the encoder so the
embedding of one view predicts the embedding of the other, and read off a representation.
The predictability objective alone is under-constrained — its global minimum is the
trivial *constant* map (*complete collapse*), and a softer failure sends all embeddings
into a low-dimensional subspace (*dimensional collapse*). Every joint-embedding method
therefore includes a second term whose job is to prevent the embeddings from degenerating.

The question is how to design that second term — an anti-collapse regularizer over a
batch of embeddings.

## Background

Joint-embedding self-supervised learning had, by this time, become a central engine for
representation learning. The mechanisms in wide use fall into a few families:
*feature whitening / decorrelation* layers that force the batch covariance toward
the identity (Ermolov et al. 2021); *negative samples* that explicitly push apart
embeddings of different inputs (SimCLR, Chen et al. 2020; MoCo, He et al. 2020);
*asymmetric architectures* — a momentum / EMA *teacher* network feeding a *student*, with a
*stop-gradient* on the teacher branch and asymmetric view generation (BYOL; DINO, Caron et
al. 2021; I-JEPA, Assran et al. 2023). The dominant theoretical lens was mutual information,
whose various bounds were shown to recover methods such as InfoNCE and relatives.

A few load-bearing facts about the design space were established and knowable before any new
regularizer. **Collapse is a real, observed failure mode** of the bare predictability
objective: without an anti-collapse term the encoder is documented to map all inputs to
near-identical embeddings, and dimensional collapse — the representation living in a
strict subspace — was diagnosed and tied to the covariance spectrum of the embeddings
(Jing et al. 2021). **The geometry of the embedding distribution controls downstream
behavior.** For a frozen encoder evaluated with a linear probe (ridge / OLS), one can write
the bias and variance of the probe as a function of the embedding covariance: for the ridge
estimator `beta_hat = (Z^T Z + lambda I)^{-1} Z^T y`, the bias is
`-lambda (Z^T Z + lambda I)^{-1} beta_true`, whose magnitude along the smallest-variance
eigendirection is amplified relative to the isotropic case, and the unregularized variance
is `tr(Var(beta_hat)) = sigma^2 sum_j 1/lambda_j`, which by convexity of `1/x` is larger
when the covariance eigenvalues `{lambda_j}` are spread out than when they are equal at
fixed total variance. These facts are about probes and embedding geometry, independent
of any particular anti-collapse rule. The same dependence appears for nonlinear probes: the leading bias of a
radius-`k`-NN or kernel (Nadaraya–Watson) estimator scales with the Fisher-information
functional `J(p) = integral ||grad log p(x)||^2 p(x) dx` of the embedding density `p`.

The relevant statistical toolbox was also mature. The **Cramér–Wold theorem** (Cramér &
Wold 1936): two random vectors are equal in distribution if and only if all of their
one-dimensional linear projections are equal in distribution — reducing a multivariate
distributional question to a family of univariate ones. **Univariate goodness-of-fit
tests** against a target distribution had been studied for a century, in three families:
*moment-based* (Jarque–Bera, comparing skewness and kurtosis); *CDF-based* / empirical-
distribution-function (Cramér–von Mises, Anderson–Darling, Watson, Kolmogorov–Smirnov,
Shapiro–Wilk), which compare the empirical CDF to the target and require sorting the
samples; and *characteristic-function-based* (Epps–Pulley 1983), which compare the
*empirical characteristic function* `phi_hat_X(t) = (1/n) sum_j exp(i t X_j)` to the
target's characteristic function in a weighted `L2` norm. The **multivariate** normality
tests built directly on the characteristic function — Baringhaus–Henze / Henze–Zirkler and
relatives — were known to take the form of a *double sum over all pairs* of samples,
`(1/n) sum_{j,k} exp(-beta^2 ||Y_j - Y_k||^2 / 2)` and similar, i.e. quadratic in the
number of samples. Finally, the idea of *slicing* a high-dimensional problem into random
one-dimensional projections to escape quadratic or exponential cost was established
elsewhere: sliced score matching (Song et al. 2020) for density/score estimation, and the
sliced Wasserstein distance (Bonneel et al. 2015) for optimal transport, both of which
average a cheap one-dimensional computation over random directions.

## Baselines

These are the prior regularizers a new anti-collapse term would be measured against and
react to.

**VICReg (Bardes, Ponce & LeCun, ICLR 2022).** A three-part loss on the two batches of
view embeddings `Z, Z'`: a per-dimension *variance* hinge that pushes each embedding
coordinate's standard deviation above a threshold `gamma` (`max(0, gamma - std(Z_j))`,
summed over coordinates `j`); a *covariance* penalty summing the squared off-diagonal
entries of the batch covariance matrix (decorrelating coordinates); and an *invariance*
term, the mean-squared distance between the two views' embeddings. It is simple, symmetric,
needs no negatives, teacher, or stop-gradient.

**Barlow Twins (Zbontar et al. 2021).** Drives the *cross-correlation matrix* between the
two views' embeddings toward the identity: diagonal entries to one (invariance), off-
diagonal to zero (redundancy reduction).

**Whitening / decorrelation methods (Ermolov et al. 2021).** Insert a whitening operation
so the batch covariance is forced to the identity directly.

**Teacher–student with stop-gradient (DINO, Caron et al. 2021; I-JEPA, Assran et al.
2023).** Prevent collapse not with an explicit statistical penalty but with an asymmetry:
a slowly-updated EMA teacher provides targets for a student, a stop-gradient blocks the
trivial solution, and centering / sharpening keeps outputs from degenerating.

**Direct multivariate goodness-of-fit (Baringhaus–Henze / Henze–Zirkler).** One could, in
principle, attach a multivariate normality test directly to the batch of embeddings to
force them toward a Gaussian. Every such test takes the form of a double sum over sample
pairs, `O(N^2)` in batch size.

## Evaluation settings

The natural yardsticks already in use for an anti-collapse regularizer:

- **Linear-probe accuracy on a frozen backbone.** Pretrain the encoder with the
  self-supervised objective, freeze it, and fit a linear classifier on top; report
  classification accuracy. The standard small-scale image benchmark is CIFAR-10 (50k
  train / 10k validation, 10 classes); larger-scale evaluation uses ImageNet-1k and its
  subsets (ImageNet-100, ImageNet-10). The probe is a frozen-features evaluation, the
  community-standard measure of representation quality.
- **Backbones across scales.** ResNet-18 / 34 / 50 convolutional networks for the small
  scale; Vision Transformers (ViT-S through ViT-H/g) and ConvNeXt / Swin / MaxViT for the
  large scale — a regularizer should generalize across architecture families and sizes.
- **Projector.** A small MLP head (e.g. `features_dim -> 2048 -> 2048 -> proj_dim`) maps
  backbone features into the space where the regularizer acts; downstream evaluation uses
  the backbone features, not the projector output.
- **Training protocol.** On CIFAR-scale: ~100 epochs, batch size 256, the LARS optimizer
  (lr ~0.3) with a warmup-then-cosine schedule; two or more augmented views per image
  generated with the standard augmentation stack (random resized crop, color jitter,
  grayscale, blur, solarize, horizontal flip). Data-parallel (multi-GPU) training is the
  default at scale, so any batch statistic the regularizer computes must be reducible
  across devices.

## Code framework

The regularizer plugs into an existing joint-embedding training harness. The data pipeline,
the augmentation stack, the backbone, the projector MLP, the optimizer, and the outer
training loop already exist; what does not exist yet is the anti-collapse term itself. The
substrate is therefore a generic module that receives the two batches of projected view
embeddings and must return a loss dictionary with a scalar `"loss"` key; an invariance term
(mean-squared distance between the views) is the uncontroversial half of the objective. The
single empty slot is the anti-collapse penalty.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def all_reduce_mean(x):
    """Average a tensor across data-parallel processes (identity on a single device).
    Available because batch statistics must be combined across GPUs."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


class CustomRegularizer(nn.Module):
    """Anti-collapse regularizer for joint-embedding self-supervised learning.

    Receives two batches of projected embeddings from two augmented views of the
    same images and returns a loss dict. Must rule out the degenerate (collapsed)
    solutions of the bare predictability objective, at linear time/memory cost in
    the batch size and embedding dimension, with bounded gradients.

    z1: [B, K]  projected embeddings of view 1
    z2: [B, K]  projected embeddings of view 2
    -> dict with at least a scalar "loss"
    """

    def __init__(self, **kwargs):
        super().__init__()
        # TODO: any state the anti-collapse regularizer we design will need.
        pass

    def forward(self, z1, z2):
        # invariance: the uncontroversial half — the two views should agree
        invariance_loss = F.mse_loss(z1, z2)

        # TODO: the anti-collapse penalty we will design — a per-batch term that
        #       prevents the embeddings from degenerating. Combine it with the
        #       invariance term and return the total.
        total_loss = invariance_loss  # placeholder
        return {"loss": total_loss}


# existing joint-embedding training loop the regularizer plugs into
def train_step(backbone, projector, regularizer, optimizer, view1, view2):
    optimizer.zero_grad()
    z1 = projector(backbone(view1))          # [B, K] projected embeddings, view 1
    z2 = projector(backbone(view2))          # [B, K] projected embeddings, view 2
    out = regularizer(z1, z2)                # anti-collapse + invariance
    out["loss"].backward()
    optimizer.step()
    return out
```

The training loop supplies the two view-embedding batches; `forward` is where the
anti-collapse rule will live.
