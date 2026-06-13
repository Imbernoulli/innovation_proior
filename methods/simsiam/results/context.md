# Context

## Research question

We want to learn good visual representations from images *without any labels*, and the structure
that has come to dominate this problem is the **Siamese network**: take one image, produce two
randomly augmented views, push their features to agree. Agreement alone, however, has a trivial
minimizer — map *every* image (and every view) to the same constant vector. Agreement is perfect,
information is zero. This is **collapse**, and it is the central obstacle: any method built on
"two views should match" must carry some additional mechanism whose job is to keep the solution
off that constant.

Every successful recipe pays for collapse prevention with substantial machinery — a large pile of
negative examples, or an online clustering step with a balance constraint, or a second
slowly-updated encoder. The precise question is: *which* of those moving parts is actually
responsible for preventing collapse? Is any of them necessary? Concretely, can a bare Siamese
network — one encoder with weights shared between the two branches, ordinary batch sizes, no
negatives, no second encoder — learn meaningful representations without collapsing, and if so,
what is the minimal ingredient that makes that possible?

## Background

**The Siamese principle and the collapse failure mode.** A Siamese network applies the *same*
weight-sharing function to two or more inputs and compares the outputs. Used for label-free
representation learning, the two inputs are two augmentations of one image and the objective
maximizes their similarity. The well-known degenerate solution is that all outputs collapse to a
constant: the similarity objective is then trivially maximized while the representation carries no
information. Diagnostically, collapse is visible in two signatures at once — the training loss
races to its minimum possible value, and the per-channel standard deviation of the
ℓ₂-normalized outputs goes to zero (all samples map to the same unit vector). As a
reference point, if the outputs were instead a zero-mean isotropic Gaussian, the std of the
ℓ₂-normalized output would sit near \(1/\sqrt{d}\) for output dimension \(d\); a healthy,
non-collapsed model should hover near that value rather than at zero.

**Negatives as the classical anti-collapse device.** Contrastive learning (Hadsell et al. 2006)
attracts positive pairs (two views of one image) and repulses negative pairs (views of different
images). The negatives are what excludes the constant solution: if all images mapped to one point,
the repulsion term would be maximally violated. The InfoNCE / NT-Xent loss writes this as a
softmax over similarities, and empirically the representation improves with the *number* of
negatives — which is exactly why surrounding machinery exists to supply many of them.

**Clustering as the other classical device.** Instead of pairwise repulsion, group images and use
the group as a target (DeepCluster, Caron et al. 2018). The cross-entropy over \(K\) clusters
implicitly contrasts. Collapse here is "all images into one cluster"; the standard guard is a
balanced-partition constraint, e.g. via the entropy-regularized optimal-transport / Sinkhorn-Knopp
solution that forces assignment mass to spread evenly across clusters.

**Batch normalization, projection heads, predictor heads.** Three architectural primitives are in
the air. BN (Ioffe & Szegedy 2015) stabilizes optimization of deep nets. A *projection MLP* head
placed before the loss (popularized in the contrastive setting) improves the learned features. A
*prediction MLP* head on one branch — a small network that transforms one view's embedding to
match the other — was introduced in the momentum-target line. Whether any of these has anything to
do with *collapse* (as opposed to plain optimization quality) is an open, testable question.

**Augmentations carry the invariances.** The transformation set — random resized crop, color
jitter, grayscale, Gaussian blur, horizontal flip — defines what the representation becomes
invariant to. The augmentation distribution is the implicit supervision.

**Alternating optimization / k-means / EM.** A separate, classical tool: when a loss couples two
sets of variables, one can minimize by *alternating* — fix one set and solve for the other, then
swap. k-means is the canonical instance (alternate cluster centers and assignments); EM is the
probabilistic version (E-step computes expected assignments, M-step updates parameters).

## Baselines

**SimCLR (Chen et al. 2020).** Siamese network with *direct weight-sharing* between branches.
Negatives are the other elements of the current batch; NT-Xent loss with temperature \(\tau\), a
projection MLP head before the loss, strong augmentation. Collapse is prevented by the negatives.
*Gap:* needs *very large batches* (e.g. 4096) to have enough negatives, and a large-batch
optimizer (LARS).

**MoCo (He et al. 2019).** Siamese network with *indirect* weight-sharing: one branch is a
**momentum encoder** (an exponential moving average of the trained encoder), and a **queue** of
~65k features from recent batches supplies a large, consistent negative set without a huge batch.
*Gap:* a second encoder plus a long queue — extra memory and moving parts; still fundamentally
relies on negative pairs.

**Clustering / SwAV (Caron et al. 2020).** Incorporates online clustering into a Siamese network:
compute a cluster assignment ("code") from one view and predict it from the other. The assignment
is solved per batch under a balanced-partition constraint via the Sinkhorn-Knopp transform, which
prevents collapse; stop-gradient is applied on the assignment (target) branch. No explicit
negative exemplars, but the cluster centers act as negative prototypes. *Gap:* requires enough
samples per assignment — large batches (4096) or a queue — and carries the online-clustering /
Sinkhorn machinery.

**BYOL (Grill et al. 2020).** Drops negatives entirely. Directly predicts the output of one view
from the other using a **prediction MLP** head on the online branch, and the target branch is a
**momentum encoder**; the loss is the (symmetrized) negative cosine similarity of ℓ₂-
normalized vectors, equivalently the MSE of normalized vectors up to a scale of 2. It does not
collapse. The reported account attributes this to the momentum encoder: removing it is reported to
fail badly (0.3% accuracy in that work's ablation). *Gap:* the explanation is given at the level of
the momentum encoder as a whole, and the method still carries a full second encoder and large
batches.

## Evaluation settings

- **Pre-training data:** the 1000-class ImageNet training set, used *without labels*.
- **Primary protocol — linear evaluation:** freeze the pre-trained backbone (features from the
  global average pooling layer), train a supervised linear classifier on top on the training set,
  report top-1 accuracy on the validation set (center crop).
- **Progress monitor:** a k-nearest-neighbor classifier on frozen features, used during
  pre-training to track learning cheaply.
- **Collapse diagnostics:** the training-loss curve (does it hit the minimum possible value?) and
  the per-channel std of the ℓ₂-normalized output (does it go to zero, vs. the \(1/\sqrt{d}\)
  reference?).
- **Transfer:** fine-tune the pre-trained backbone for VOC object detection (Faster R-CNN) and
  COCO detection / instance segmentation (Mask R-CNN, C4 backbone).
- **Backbone & recipe yardsticks:** ResNet-50; SGD vs. large-batch optimizers (LARS); linear lr
  scaling \(lr \times \text{BatchSize}/256\); cosine decay schedule; batch sizes from 64 to 4096.

## Code framework

The pre-method primitives that already exist: a ResNet backbone, MLP heads built from
`Linear`/`BatchNorm1d`/`ReLU`, an `ℓ₂`-normalized cosine similarity, an SGD optimizer with
cosine-decayed learning rate, and a two-view augmentation pipeline. The data loader yields two
independently augmented views per image. The model and loss are left as the empty slot the method
will fill.

```python
import torch
import torch.nn as nn

class TwoCropsTransform:
    """Take one image, return two independently augmented views."""
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

class SiameseModel(nn.Module):
    """A weight-sharing network over two views."""
    def __init__(self, base_encoder, dim=2048):
        super().__init__()
        self.encoder = base_encoder(num_classes=dim)
        # TODO: define the model.

    def forward(self, x1, x2):
        # TODO: process both views with the shared encoder.
        pass

def loss_fn(*outputs):
    # TODO: define the training objective.
    pass

def train_one_epoch(loader, model, optimizer):
    for images, _ in loader:
        x1, x2 = images[0], images[1]
        outputs = model(x1, x2)
        loss = loss_fn(*outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
