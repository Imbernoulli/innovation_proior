## Research question

We want to learn useful visual representations from unlabeled images. The dominant recipe is
*joint-embedding self-supervised learning*: take an image, produce two distorted views of it via a
random augmentation pipeline, push both through a network, and train so that the two views land on
similar representations. The hope is that whatever survives the distortions is the semantic content,
so the learned features transfer well to downstream tasks (classification, detection).

The central obstacle is **collapse**. "Make the two views agree" is satisfied perfectly by a network
that ignores its input and emits a constant vector: every pair of views agrees trivially, the
representation carries zero information, and the loss is minimized. So the bare objective
"maximize similarity between views" is degenerate. Every working method bolts on some extra
mechanism whose job is to make collapse unattractive. The open question is whether
there is a *single, symmetric objective whose optimum is non-trivial by construction* — one that
does not need negative samples, a special predictor branch, a momentum/target network, a
stop-gradient, or a clustering step to stay away from the constant solution. A solution would have
to (a) define a clear quantity to optimize, (b) provably exclude the constant solution at that
optimum, and (c) scale to ImageNet-sized training.

## Background

**Siamese / joint-embedding learning.** The shared-weights "Siamese" setup for learning invariances
goes back to contrastive loss formulations for dimensionality reduction (Hadsell, Chopra & LeCun
2006). Modern SSL inherits this: two augmented views, one shared encoder, an objective that rewards
agreement between the views. The augmentation distribution encodes the invariances we want (crops,
color jitter, grayscale, blur, solarization, flips); the standard pipeline became the one used by
BYOL (Grill et al. 2020), with asymmetric blur/solarize probabilities for the two views.

**The collapse phenomenon and the diagnostics around it.** Because "agree" is trivially solvable by a
constant, the field has accumulated evidence about *what* keeps a method off the constant: contrastive
methods repel negatives; SimSiam's ablation (Chen & He 2020) isolates the stop-gradient as the
critical ingredient (removing it collapses immediately); analyses of BYOL implicate batch
normalization (Tian et al. 2020; Fetterman & Albrecht 2020) or group normalization (Richemond et al.
2020). Two empirical regularities about *existing* systems are especially relevant. First, contrastive
methods that draw negatives from the minibatch degrade as the batch shrinks — SimCLR loses several
points of ImageNet linear-probe accuracy going from large batches down to 256 — whereas the
asymmetric no-negative methods are batch-size robust because their loss has no cross-sample term.
Second, increasing the dimensionality of the projector output *saturates* performance for contrastive
and asymmetric methods alike: bigger embeddings buy nothing past a point.

**Redundancy reduction (Barlow 1961).** A separate, much older idea comes from neuroscience. H. Barlow
hypothesized that the purpose of early sensory processing is to recode highly redundant sensory input
into a *factorial code* — a representation whose components are statistically independent. This
"redundancy-reduction" principle has been used to explain organization in the visual pathway (Barlow
2001) and has seeded learning algorithms based on decorrelating units (Földiák 1990; Linsker 1988;
Redlich 1993; Schmidhuber 1996; Deco & Parra 1997). The principle is a statement about single-signal
coding — it concerns the internal statistics of one representation — and was not posed in the
two-view joint-embedding setting.

**Information-theoretic framing.** The Information Bottleneck (Tishby, Pereira & Bialek 2000; Tishby &
Zaslavsky 2015) formalizes "keep what matters, drop the nuisance" as a trade-off between mutual
informations. In the two-view setting, with original image `X`, distorted view `Y`, and representation
`Z = f(Y)`, minimizing `I(Z,Y) - beta I(Z,X)` penalizes information about the distorted input while
rewarding information about the underlying sample. For a deterministic encoder, `H(Z|Y)=0`, and under
a Gaussian model the entropy of a representation is `(1/2)log|Cov|`, a *parametric* quantity that
can be estimated from second moments — unlike non-parametric entropy estimators, which suffer the
curse of dimensionality and need many samples.

**Whitening.** Classical whitening (ZCA, Cholesky-based) transforms features to have identity
covariance — zero mean, unit variance, zero cross-correlation. It is classically applied as an
explicit, hard transform of the data.

## Baselines

**SimCLR / InfoNCE (Chen et al. 2020; van den Oord et al. 2018; He et al. 2019 — MoCo).** Contrastive
learning. With feature-normalized embeddings (cosine similarity), the InfoNCE loss for a positive pair
`(z^A_b, z^B_b)` scored against the other examples in the opposite-view batch is

```
L_infoNCE = - sum_b  <z^A_b, z^B_b> / (tau ||z^A_b|| ||z^B_b||)
            + sum_b  log sum_{b' != b} exp( <z^A_b, z^B_b'> / (tau ||z^A_b|| ||z^B_b'||) )
```

a *similarity term* pulling positives together and a *contrastive term* pushing each sample away from
the other samples in the batch (interpretable as a non-parametric entropy / uniformity estimate over
the embedding distribution, Wang & Isola 2020). Collapse is avoided because the constant solution
cannot make a positive pair more similar than the negatives: all logits are equal, so the loss stays
at the uniform-classification value rather than approaching its minimum. **Gaps:** the entropy
estimate is non-parametric, so it needs many negatives — large batches (SimCLR) or a momentum encoder
with a ~65k-entry queue (MoCo) — and it gains nothing from high-dimensional embeddings (curse of
dimensionality). Temperature `tau` is an extra knob weighting the hardest negatives.

**BYOL / SimSiam (Grill et al. 2020; Chen & He 2020).** No negatives, no contrastive term — just a
cosine similarity between the two views:

```
L_cosine = - sum_b  <z^A_b, z^B_b> / (||z^A_b|| ||z^B_b||)
```

By itself this collapses. They avoid it with *asymmetry*: an extra "predictor" MLP on one branch
breaks the symmetry, and the other branch is a detached/fixed target (BYOL additionally maintains it
as an exponential moving average of the online weights). **Gaps:** these methods are batch-size robust
(no cross-sample term) but they are *not* the minimization of any single objective — there exist
trivial solutions that the procedure avoids only through implementation choices and learning dynamics;
why exactly stop-gradient + predictor prevents collapse is the subject of ongoing study. Conceptually
unprincipled, even if empirically strong.

**Clustering — DeepCluster / SwAV / SeLa (Caron et al. 2018, 2020; Asano et al. 2019).** Assign views
to clusters/codes and enforce that two views of an image get consistent assignments, alternating with
a (often non-differentiable) clustering step. **Gaps:** prone to collapse via empty clusters, avoided
only by balance constraints / careful engineering; needs to store features when #clusters >> batch.

**Hard whitening — W-MSE (Ermolov et al. 2020, concurrent).** Differentiably whiten each batch of
embeddings (Cholesky) so their covariance is the identity, then take cosine similarity between the
whitened views. **Gap:** the whitening constraint is enforced as an explicit matrix operation on every
batch rather than as a soft penalty inside a simple scalar objective.

**IMAX (Becker & Hinton 1992; Zemel & Hinton 1990).** An early twin-network objective
`log|Cov(Z^A - Z^B)| - log|Cov(Z^A + Z^B)|` that maximizes information between the two
representations under an additive-independent-Gaussian-noise-on-a-shared-signal model. **Gaps:** it is
*directly* an information quantity with no trade-off knob, and it was not shown to scale to
large-scale vision.

## Evaluation settings

- **Pretraining:** ImageNet-1k (1.28M images), no labels. ResNet-50 encoder. Two augmented views per
  image with the standard SSL augmentation pipeline (random resized crop to 224x224, horizontal flip,
  color jitter, grayscale, Gaussian blur, solarization), the last few applied stochastically with
  view-dependent probabilities.
- **Linear evaluation:** freeze the encoder, train a linear classifier on top of the 2048-d features,
  report ImageNet top-1 / top-5.
- **Semi-supervised:** fine-tune on 1% and 10% of ImageNet labels.
- **Transfer:** linear classification on Places-205, VOC07, iNaturalist-2018; object detection /
  instance segmentation on VOC07+12 (Faster R-CNN) and COCO (Mask R-CNN) via detectron2, following
  the protocol of He et al. 2019.
- **Optimization yardstick:** LARS optimizer (You et al. 2017), learning rate scaled by batch/256 with
  a warmup and cosine decay (Loshchilov & Hutter 2016), as used by the standard SSL training protocol.

## Code framework

The primitives that already exist: a ResNet-50 backbone (drop the classifier head), an MLP, batch
normalization, the LARS optimizer, the two-view augmentation pipeline, and a standard distributed
training loop. The open slot is the scalar objective computed from the two projected views.

```python
import random

import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms


class GaussianBlur:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        # existing image augmentation primitive
        pass


class Solarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        # existing image augmentation primitive
        pass


class Transform:
    def __init__(self):
        # TODO: instantiate the two stochastic view pipelines.
        pass

    def __call__(self, x):
        # TODO: return one transformed image from each pipeline.
        pass


def build_projector(projector):
    # TODO: choose the projector depth, width, normalization, and nonlinearity.
    pass


def off_diagonal(x):
    # TODO: return the off-diagonal entries if the objective uses a square feature matrix.
    pass


def pairwise_feature_objective(z1, z2, normalizer, batch_size, weight):
    # TODO: combine the two projected batches into one scalar self-supervised loss.
    pass


class TwinEmbeddingModel(nn.Module):
    def __init__(self, projector="TODO", objective_weight=1.0, batch_size=2048):
        super().__init__()
        self.objective_weight = objective_weight
        self.batch_size = batch_size
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = build_projector(projector)
        self.feature_normalizer = None  # TODO: fill only if the objective needs one.

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        return pairwise_feature_objective(
            z1, z2, self.feature_normalizer, self.batch_size, self.objective_weight
        )


def train_step(model, optimizer, y1, y2):
    # TODO: add the ordinary optimization step around the model loss.
    pass
```
