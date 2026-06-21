# Context: few-shot image classification landscape (circa 2016-2017)

## Research question

A classifier is shown a handful of labeled examples of brand-new classes it never saw during
training — say five classes with five (or even one) example each — and must immediately label
fresh query images of those same classes. Re-training a deep network on five-times-five images
overfits catastrophically: there is nowhere near enough data to fit millions of weights, and
yet humans recognize a new visual concept from a single example. So the problem is to build a
classifier that *generalizes to unseen classes from a tiny support set*, without per-task
retraining, and whose generalization comes from the right **inductive bias** rather than from
data volume.

Concretely the protocol is *episodic*: each task ("episode") samples N classes, gives K labeled
support examples per class, and presents query images to be sorted into one of the N classes.
The contribution should be a reusable algorithmic component — a way to summarize a support set
and to compare a query against it — that can be trained end-to-end.

## Background

Few-shot classification was studied long before deep features
(Miller et al. 2000; Fei-Fei et al. one-shot learning; Lake et al. 2011 on Omniglot, showing
humans do one-shot recognition with high accuracy). The modern wave reframed it around two
ideas that were in the air by 2016-2017:

- **Learned embeddings + metric comparison.** Instead of training a fresh classifier per task,
  learn a single embedding network once, then do classification in embedding space by *distance*
  or *similarity*. This descends from the metric-learning literature
  (Kulis 2012; Bellet et al. 2013): learn a transformation of the input so that a simple
  nearest-neighbor rule works well in the transformed space. The seminal instance is
  Neighbourhood Components Analysis (Goldberger, Hinton, Roweis, Salakhutdinov 2004), which
  learns a linear map A (equivalently a Mahalanobis metric Q = AᵀA) by maximizing a *soft*
  leave-one-out nearest-neighbor objective; its nonlinear extension replaces A by a neural
  network (Salakhutdinov & Hinton 2007). Large-margin variants (LMNN, Weinberger et al. 2005;
  DNet-KNN, Min et al. 2009) optimize the same KNN-accuracy target with hinge losses.

- **Episodic training: make the training condition match the test condition.** Rather than
  training on ordinary minibatches and hoping the embedding transfers, sample mini-tasks during
  training that look exactly like test tasks (N classes, K shots, a query set), and optimize the
  classifier's performance *on the query set of each sampled episode*. This makes the training
  objective faithful to the few-shot test environment and is a strong regularizer in the
  limited-data regime.

- **The meta-learning framing.** A complementary line treats few-shot learning as "learning to
  learn": train a procedure that, given an episode, *produces* a good classifier for it.
  Ravi & Larochelle (2017) train an LSTM meta-learner to emit the parameter updates of a
  classifier for each episode, exploiting that LSTM cell dynamics and gradient-descent updates
  have the same algebraic form; Finn et al. (2017) instead learn an initialization from which a
  few gradient steps adapt to any new task. These methods learn a custom model *per episode*,
  which is powerful but heavy.

A few load-bearing facts about the design space matter here. First, a Bregman divergence is any
d_φ(z, z') = φ(z) − φ(z') − (z − z')ᵀ∇φ(z') for a strictly convex φ of Legendre type; squared
Euclidean distance (φ = ‖·‖²) and Mahalanobis distance are Bregman, but cosine distance is not.
Banerjee, Merugu, Dhillon & Ghosh (2005) proved a sharp property of this family: for any Bregman
divergence, the point minimizing the total divergence to a set of points is exactly their
arithmetic mean — argmin_z Σ_{x∈X} d_φ(x, z) = (1/|X|) Σ_x x — and, conversely, *any* smooth
distortion function with this mean-as-minimizer property must be a Bregman divergence (the family
is exhaustive for it). They also established a bijection between regular exponential-family
densities and regular Bregman divergences: every density p_ψ(z|θ) = exp{zᵀθ − ψ(θ) − g_ψ(z)} can
be rewritten as exp{−d_φ(z, μ(θ)) − g_φ(z)}. Second, the *choice of distance* is not a cosmetic
knob: NCA is built from negative squared Euclidean/Mahalanobis distances, nearest-class-mean
classifiers use Mahalanobis distances to class means, and Matching Networks default to cosine
similarity. These competing choices make the distance an open design variable, not a harmless
implementation detail. Third, the prior episodic work typically chooses the episode "way" and
"shot" to resemble the few-shot test task; whether those knobs must match test conditions exactly
remains an open design question.

## Baselines

**Matching Networks (Vinyals, Blundell, Lillicrap, Wierstra et al. 2016).** Embed every support
and query image, then classify a query x̂ as a weighted vote over support labels,
ŷ = Σ_i a(x̂, x_i) y_i, where the attention weight a(x̂, x_i) = exp(c(f(x̂), g(x_i))) /
Σ_j exp(c(f(x̂), g(x_j))) is a softmax of the cosine similarity c between embedded query and
embedded support points. This is a differentiable weighted nearest-neighbor / kernel-density
classifier in embedding space. Matching Networks introduced episodic training — optimizing
max_θ E_{L∼T}[ E_{S,B∼L}[ Σ_{(x,y)∈B} log P_θ(y | x, S) ] ] over sampled N-way K-shot episodes —
which is its most durable contribution. It also proposed Full Context Embeddings (FCE): a
bidirectional LSTM over the support set to contextualize g, and an attention-LSTM (unrolled a
fixed number of steps) to contextualize f conditioned on the support set.

**Neighbourhood Components Analysis (Goldberger, Hinton, Roweis, Salakhutdinov 2004) and its
nonlinear extension (Salakhutdinov & Hinton 2007).** Learn an embedding so that a stochastic
nearest-neighbor rule classifies well. Each point i picks neighbor j with probability
p_ij = exp(−‖Ax_i − Ax_j‖²) / Σ_{k≠i} exp(−‖Ax_i − Ax_k‖²), p_ii = 0, a softmax over negative
squared Euclidean distances in the transformed space, and the objective maximizes the expected
number of correctly classified points, f(A) = Σ_i Σ_{j: c_j = c_i} p_ij. The nonlinear version
replaces the linear map A by a neural network.

**Nearest Class Mean classifier (Mensink, Verbeek, Perronnin, Csurka 2013).** Represent each
class by the mean μ_c of its examples and assign a query to the nearest mean,
c* = argmin_c d(x, μ_c), under a learned Mahalanobis metric, with a probabilistic multi-class
form p(c | x) ∝ exp(−½ d_W(x, μ_c)). Because a class is just an average, brand-new classes can
be added at near-zero cost — average their examples into a new mean — without retraining the
metric. Non-linear classification extends this with a k-means partitioning step in input space
to assign multiple centroids per class, with the metric optimized separately.

**Meta-Learner LSTM (Ravi & Larochelle 2017).** Train an LSTM to output the iterative updates of
an episode-specific classifier so that it generalizes to the episode's query set; provided the
miniImageNet 64/16/20 class split.

**Simple transfer baseline (reported by Ravi & Larochelle 2017).** Train an ordinary classifier
on the base classes, then do cosine nearest-neighbor on the penultimate features for novel
classes.

## Evaluation settings

The yardsticks already established by this prior work:

- **Omniglot** (Lake et al. 2011): 1623 handwritten characters from 50 alphabets, 20 examples
  each, grayscale resized to 28×28, classes augmented with 90° rotations; the standard split
  (Vinyals et al. 2016) trains on 1200 characters (×4 rotations = 4800 classes) and tests on the
  rest. Evaluated 5-way and 20-way, 1-shot and 5-shot.
- ***mini*ImageNet** (Vinyals et al. 2016): 100 ImageNet classes, 600 color images each at 84×84;
  the Ravi & Larochelle (2017) split fixes 64 train / 16 validation / 20 test classes for direct
  comparison. Evaluated 5-way 1-shot and 5-shot, with accuracy averaged over 600 test episodes
  and reported with 95% confidence intervals; 15 query points per class.
- **CIFAR-FS** and **CUB-200**: 100 classes from CIFAR-100, and 200 fine-grained bird species
  (Welinder et al. 2010), used as further few-shot benchmarks (and CUB as a zero-shot benchmark
  with 312-dim attribute vectors as class meta-data).
- **Metric:** mean classification accuracy over many randomly sampled test episodes (higher is
  better). The natural experimental knobs to study are the distance function (cosine vs.
  Euclidean), the number of classes per training episode ("way"), and whether the training shot
  matches the test shot.
- A standard backbone in this setting is a four-block convolutional embedding
  (each block: 3×3 conv -> batch norm -> ReLU -> 2×2 max-pool); on 28×28 Omniglot inputs this
  yields a 64-dimensional embedding, while 84×84 miniImageNet inputs produce a larger flattened
  embedding under the same block pattern. Optimization is by SGD-style episodic updates, commonly
  with Adam and a step schedule.

## Code framework

The method plugs into an existing episodic harness: a data pipeline samples N-way K-shot
episodes (support images/labels, query images/labels), a shared backbone embeds images into
feature vectors, a training loop processes the support set, scores the queries, computes a loss
on the query labels, and backpropagates through the backbone. All of that already exists. What
does *not* yet exist is the part that turns a support set into something a query can be compared
against, and the rule that scores a query — that is the empty slot to be designed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def make_backbone(use_pooling: bool = True) -> nn.Module:
    """Shared embedding network: image -> feature vector."""
    ...


class FewShotClassifier(nn.Module):
    """Existing base class. Owns the backbone and a place to stash support-set info."""

    def compute_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    # generic primitives the harness already provides:
    # - a way to compute one per-class summary vector from a labeled set of features
    # - L2 and cosine distance from a batch of features to a stored set of per-class vectors
    def compute_class_summary(self, features: Tensor, labels: Tensor) -> Tensor: ...
    def l2_distance_to_class_summary(self, features: Tensor) -> Tensor: ...
    def cosine_distance_to_class_summary(self, features: Tensor) -> Tensor: ...


class CustomFewShotMethod(FewShotClassifier):
    """The few-shot method to be designed. Fill in the three slots below."""

    def __init__(self):
        backbone = make_backbone(use_pooling=True)
        super().__init__(backbone=backbone)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        # TODO: from the labeled support set, build whatever we will compare queries against,
        #       and store it for forward().
        pass

    def forward(self, query_images: Tensor) -> Tensor:
        # TODO: score each query against the stored support-set information;
        #       return scores of shape (n_query, n_way).
        pass

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        # TODO: the training loss on query scores vs. query labels.
        pass


# existing episodic training loop the method plugs into
def training_epoch(model, data_loader, optimizer):
    for support_images, support_labels, query_images, query_labels, _ in data_loader:
        optimizer.zero_grad()
        model.process_support_set(support_images, support_labels)   # build support representation
        scores = model(query_images)                                # score queries
        loss = model.compute_loss(scores, query_labels)             # loss on query labels
        loss.backward()                                             # backprop through backbone
        optimizer.step()
```

The harness supplies the episodes, the backbone, and distance utilities; the three method
bodies — how to summarize the support set, how to score a query, and the loss — are what remains
to be filled in.
