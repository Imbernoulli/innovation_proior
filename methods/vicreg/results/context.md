# Context

## Research question

We want to learn useful visual representations from unlabeled images. The dominant recipe is
*joint-embedding self-supervised learning*: take an image, produce two distorted views of it with a
random augmentation pipeline, push both through a network, and train so that the two views land on
similar embeddings. Whatever survives the distortions is taken to be the semantic content, so the
learned features should transfer to downstream tasks (classification, detection, segmentation).

The central obstacle is **collapse**. "Make the two views agree" is satisfied perfectly by a network
that ignores its input and emits a constant vector: every pair of views agrees trivially, the loss
hits its minimum, and the representation carries no information. So the bare objective "maximize
similarity between views" is degenerate, and every working method adds some extra mechanism whose
only job is to make the constant solution unattractive. Those mechanisms come with costs and
constraints: large batches or memory banks, asymmetric branches, momentum/target networks,
stop-gradients, predictors, batch- or feature-wise normalization, clustering steps, quantization.
Most of them also *couple the two branches* — through shared weights, a moving-average dependency, or
a cross-branch statistic — which is what forces the two branches to be identical.

The open question is whether collapse can be prevented by an **explicit, interpretable constraint
applied to each branch separately**, with none of that machinery: no negatives, no momentum encoder,
no stop-gradient, no predictor, no normalization of the embeddings, no quantization. A method that
constrained each branch independently would, as a bonus, free the two branches to differ — different
architectures, different weights, even different input modalities — opening joint-embedding SSL to
multi-modal data. A solution would have to (a) name a clear quantity to optimize, (b) make the
constant solution impossible at the optimum *by construction*, not by training dynamics, and
(c) scale to ImageNet-sized training.

## Background

**Siamese / joint-embedding learning.** The shared-weights "Siamese" setup for learning invariances
goes back to contrastive loss formulations for dimensionality reduction (Bromley et al. 1994;
Hadsell, Chopra & LeCun 2006): two inputs, one shared encoder, an objective that pulls related pairs
together. Modern SSL inherits this skeleton — two augmented views, an encoder, an objective that
rewards agreement. The augmentation distribution encodes the invariances we want (random crops, color
jitter, grayscale, Gaussian blur, solarization, flips); the standard pipeline is the one introduced
by SimCLR (Chen et al. 2020) and refined by BYOL (Grill et al. 2020).

**The collapse phenomenon and its diagnostics.** Because "agree" is trivially solvable by a constant,
the field has accumulated evidence about *what* keeps each method off the constant, and these are
pre-method facts about existing systems:
- Contrastive methods that draw negatives from the minibatch degrade as the batch shrinks (SimCLR
  loses several points of ImageNet linear-probe accuracy going from large batches down to 256),
  because the repulsive term is a sample-space spread estimate that needs many samples.
- The asymmetric no-negative methods are batch-size robust (no cross-sample term) but their
  no-collapse behavior is dynamics-dependent and not well understood: SimSiam's own ablation isolates
  the **stop-gradient** as the load-bearing ingredient — remove it and the representation collapses to
  a constant immediately. Analyses of BYOL implicate **batch normalization** (and group normalization)
  as crucial to its stability.
- A persistent observation about decorrelation-based methods: even BYOL and SimSiam, which have *no*
  explicit decorrelation term, end up with low average correlation between representation
  dimensions — i.e. the asymmetric tricks have an implicit decorrelation effect. And their per-
  dimension standard deviation, measured at the embedding, hugs 1/√d (the value forced by projecting
  onto the unit sphere), suggesting these methods keep variance alive only indirectly, sometimes
  imperfectly (slow drift toward collapse).
- There are two distinct failure modes to keep separate. **Trivial collapse**: all embeddings shrink
  to one point (zero variance). **Informational collapse**: embeddings keep some variance but the
  dimensions are redundant — variance is concentrated in a low-dimensional subspace, so the
  representation carries less information than its dimension suggests.
- Performance of decorrelation/information-maximization methods *rises* with the dimensionality of the
  projector output (and saturates only at very large widths), in contrast to contrastive methods,
  which saturate early.

**The redundancy-reduction / information-maximization principle.** An old idea from sensory coding
(Barlow): a good code recodes redundant input into components that are statistically independent — a
factorial code. If two units always carry the same information, one is wasted. Operationally,
"informative" becomes "don't let the components duplicate each other": decorrelate them.

## Baselines

**Contrastive learning (SimCLR, MoCo, CPC, CMC).** For a positive pair (two views of one image) the
embeddings are pulled together; all other samples in the batch (or a memory bank) are pushed away,
typically via the InfoNCE loss on l2-normalized embeddings,
`L = − log [ exp(sim(z, z⁺)/τ) / Σ_k exp(sim(z, z_k)/τ) ]`. The repulsion makes the constant solution
catastrophic, so collapse is avoided. Cost: it is a sample-space spread estimate — it needs many
negatives, hence very large batches (SimCLR) or a momentum-updated queue of past embeddings (MoCo).
Gap: the dependence on batch size / memory, and the requirement to mine negatives.

**Clustering methods (DeepCluster, SwAV).** Instead of treating each sample as its own class, group
samples into clusters and require two views of an image to get the same cluster code. DeepCluster runs
k-means on previous-iteration features as pseudo-labels (expensive, asynchronous, hard to scale).
SwAV learns clusters online and keeps the assignment balanced with the Sinkhorn–Knopp transform,
which prevents the empty-cluster collapse. Gap: this is contrastive at the level of clusters — it
still needs many prototype vectors and balanced-assignment plumbing.

**Distillation / asymmetric methods (BYOL, SimSiam).** Drop negatives entirely and use a plain
similarity between the two views, which *should* collapse — and doesn't, because the two branches are
made asymmetric. One branch gets a **predictor** MLP `g_ψ` and the other branch is a detached target:
SimSiam stops the gradient through the target branch; BYOL additionally makes the target an
exponential moving average of the online weights. The invariance loss becomes a symmetrized
prediction loss, `s = (1/2n) Σ D(z_i, p'_i) + (1/2n) Σ D(z'_i, p_i)`, where `D` is the MSE between
l2-normalized vectors (BYOL) or the negative cosine similarity (SimSiam). Effective and batch-size
robust. Gap: there is no single function being minimized — trivial solutions to the similarity loss
exist and are avoided only by the asymmetry and the learning dynamics; *why* it works is unclear, and
the asymmetry forces a dependency (shared weights / EMA) between the branches.

**Information-maximization methods (Barlow Twins, W-MSE).** Prevent collapse by maximizing the
information content of the embeddings via decorrelation. **Barlow Twins** computes the normalized
**cross-correlation** matrix between the two branches' embeddings (each dimension standardized across
the batch) and drives it toward the identity: `Σ_i (1 − C_ii)² + λ Σ_{i≠j} C_ij²`, where
`C_ij = Σ_b z^A_{b,i} z^B_{b,j} / (√Σ_b (z^A_{b,i})² √Σ_b (z^B_{b,j})²)`. Diagonal-to-1 is invariance;
off-diagonal-to-0 is decorrelation. **W-MSE** transforms the embeddings into the eigenspace of their
covariance and forces the whitened vectors to be uniform on the unit sphere. Gaps: Barlow's
cross-correlation *couples the two branches* (it favors branches with similar output statistics) and
requires standardizing the embeddings (batch-normalization-like; the standardization scalar in the
denominator is needed because the pre-standardization embeddings can shrink to numerical zero). W-MSE
requires inverting a covariance matrix — costly and unstable — and computes the whitening operator over
several batches, biasing the MSE, fixed only with a batch-slicing trick.

## Evaluation settings

Self-supervised pretraining is done on the unlabeled ImageNet-1k training set; representations are the
output of the frozen encoder backbone (the projection/expander head is discarded after pretraining).
The standard yardsticks that already exist:
- **Linear evaluation on ImageNet**: train a linear classifier on the frozen representations; report
  Top-1 / Top-5 accuracy.
- **Semi-supervised on ImageNet**: fine-tune on 1% and 10% of labels (the SimCLR splits); Top-1 / Top-5.
- **Transfer**: linear classification on Places205 (scene), VOC07 (multi-label, linear SVM via
  LIBLINEAR), iNaturalist2018 (fine-grained); object detection on VOC07+12 with Faster R-CNN (C4
  backbone); detection and instance segmentation on COCO with Mask R-CNN (FPN backbone).
- **k-NN** on ImageNet (k = 20, 200) on frozen representations.
- **Retrieval**: image-to-text and text-to-image on MS-COCO (R@1/5/10), in the VSE++ setting (text
  encoder = word embedding + GRU, image encoder = ResNet).
- **Audio**: ESC-50 environmental sound classification (50 classes), linear probe on frozen features.
- Backbone is a ResNet-50 (with wider/aggregated ResNet and ViT-S variants available); standard
  augmentation pipeline; LARS optimizer with cosine learning-rate decay and warmup.

## Code framework

The primitives below already exist: a data pipeline that emits two augmented views per image, an
encoder backbone (ResNet), an MLP head, a standard distributed training loop, and the LARS optimizer.
What does **not** exist yet is the objective — the slot marked `# TODO`.

```python
import torch
import torch.nn.functional as F
from torch import nn, optim
import torchvision.datasets as datasets


class TwoViewTransform:
    """Augmentation pipeline producing two distorted views of one image."""
    def __init__(self):
        # random resized crop, flip, color jitter, grayscale, blur, solarize, normalize
        ...
    def __call__(self, image):
        return self.view(image), self.view(image)


def encoder(arch="resnet50"):
    """Backbone f_theta: image -> representation (kept for downstream)."""
    backbone, repr_dim = build_resnet(arch)
    return backbone, repr_dim


def head(in_dim, spec):
    """MLP head h_phi on top of the representation, where the loss is computed.
    Linear/BN/ReLU stack; final linear. The output dimension is a hyperparameter."""
    ...


class JointEmbedding(nn.Module):
    """Two views -> two embeddings, plus the (yet-to-be-designed) SSL objective."""
    def __init__(self, args):
        super().__init__()
        self.backbone, repr_dim = encoder(args.arch)
        self.head = head(repr_dim, args.mlp)

    def forward(self, x, y):
        z = self.head(self.backbone(x))
        z_prime = self.head(self.backbone(y))
        # TODO: the self-supervised objective that makes z, z_prime agree
        #       WITHOUT collapsing to a constant. This is the contribution.
        loss = self.objective(z, z_prime)
        return loss

    def objective(self, z, z_prime):
        # TODO
        pass


def train(args):
    transform = TwoViewTransform()
    dataset = datasets.ImageFolder(args.data_dir / "train", transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, ...)

    model = JointEmbedding(args).cuda()
    optimizer = LARS(model.parameters(), lr=0, weight_decay=args.wd)

    for epoch in range(args.epochs):
        for (x, y), _ in loader:
            x, y = x.cuda(), y.cuda()
            adjust_learning_rate(optimizer, ...)   # warmup + cosine decay
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
    # after pretraining: discard the head, keep model.backbone for downstream
```
