## Research question

The goal is to build a classifier that recognises *novel* visual categories from only a
handful of labelled examples each — one or five images per class — and does so without
collapsing to overfitting. Formally there are three datasets with two disjoint label spaces: a
large *training* set with its own classes used to learn transferable machinery, and a *support*
set plus *test* set that share a new label space disjoint from training. If the support set has
`K` labelled examples for each of `C` classes, the target is called `C`-way `K`-shot. With only
the support set in hand, the obvious move — fine-tune a deep network on those `K·C` images —
fails: a million-parameter network has nothing like enough signal in `K=1` or `K=5` examples
per class, and data augmentation or regularisation only blunts the overfitting rather than
removing it. So the problem is to extract, from the abundant training classes, a *reusable*
piece of machinery — a way of representing each example, of summarising a class from its few
support examples, and of deciding which class a query belongs to — that then transfers
feed-forward to brand-new classes it was never trained on. The piece must be a general
algorithmic component, not a dataset-specific trick, and ideally it should classify a query in
a single forward pass with **no parameter updates at test time** (low-latency, low-power
deployment), since needing to fine-tune on the support set is exactly the regime that the
data scarcity makes fragile.

## Background

Deep supervised recognition had become extremely strong in the data-rich regime
(Krizhevsky et al. 2012; He et al. 2016; Simonyan & Zisserman 2015), but those models need
large labelled sets and many gradient steps, which makes them useless for newly emerging or
rare categories where annotated images may simply never exist. Humans, by contrast, generalise
a new concept like "zebra" from a single picture. This gap drove a resurgence of *meta-learning*
("learning to learn"): extract transferable knowledge from a set of auxiliary tasks so that a
new, sparse-data task can be solved without the overfitting one would expect from applying deep
models to a few examples.

The load-bearing idea that organises this whole area is **episodic training**, introduced with
Matching Networks (Vinyals et al. 2016). The training signal is constructed to look exactly like
the test situation: in each iteration sample `C` classes from the training set, draw `K`
labelled examples per class to act as a *sample* (support) set `S = {(x_i, y_i)}`, and draw a
disjoint *query* set `Q = {(x_j, y_j)}` from the same `C` classes. The model is scored on
classifying the query given the support, and trained over many such episodes. Because every
training episode is itself a small `C`-way `K`-shot problem, the machinery the model learns is
under constant pressure to be the *transferable* part — the part that works across class
identities — rather than memorising any particular classes. This train-as-you-test principle is
the empirical backbone of every method below.

Two further pre-existing pieces matter. First, **metric-based recognition**: rather than
training a classifier head per task, learn an embedding `f` such that, in embedding space, a
query can be labelled by a simple comparison to the support examples — nearest neighbour, a
kernel/attention weighted vote, or a per-class summary. The transferable knowledge is then the
embedding `f` plus a *scoring rule*, and inference on a new task is a feed-forward
computation. Second, the observation — reported across this line of work — that the *choice of
scoring rule matters a great deal*: on the same embedding, swapping one fixed distance for
another (cosine vs. squared Euclidean) moves accuracy substantially, and the comparison rule
that happens to pair well with the way prototypes are formed (means) is not the one most people
reached for first. That fixed comparison functions are this sensitive — and that the "right" one
is dataset- and design-dependent — is a pre-method fact about the landscape, established before
this scoring slot is settled.

Also in the air, from a different problem entirely, was the idea of processing relations between
entities with a learned network. Santoro et al. (2017) proposed a module for relational
reasoning *within a single image's set of objects*: take a set of object representations
`O = {o_1,…,o_n}`, and compute `RN(O) = f_φ( Σ_{i,j} g_θ(o_i, o_j) )`, where `g_θ` is a single
shared small network applied to **every pair** of objects (the two objects concatenated as its
input), the pairwise outputs are **summed** (so the result is invariant to object ordering), and
`f_φ` reduces the pooled vector to an answer. Two properties of this construction were
emphasised: using one shared `g_θ` for all pairs is data-efficient and resists overfitting any
particular pairing, and summation gives an order-invariant aggregation over a set. This was
built to reason about object-object relations inside one scene, so it remained a different tool
from few-shot classification rather than a ready-made classifier.

## Baselines

These are the prior methods a new few-shot classifier would be measured against and would react
to. The first three are the metric-based feed-forward family most directly in scope; the last
group are the heavier alternatives.

**Matching Networks (Vinyals et al. 2016, NIPS).** The method that introduced episodic
training. It predicts a query label as an attention-weighted vote over the support labels:
`ŷ = Σ_i a(x̂, x_i) y_i`, a non-parametric, weighted-nearest-neighbour classifier in embedding
space. The attention is a softmax over the **cosine similarity** of embeddings,
`a(x̂, x_i) = softmax_i c(f(x̂), g(x_i))`, with `f`, `g` neural embeddings (optionally a
"fully-conditional embedding" that runs a bi-directional LSTM over the support set). Core idea:
learn the embedding so that a fixed cosine-attention vote classifies correctly; train it with
episodes so train matches test. **Limitation it leaves open:** all of the learning lives in the
embedding, while the comparison itself is a *fixed cosine similarity* — an element-wise,
closed-form function chosen in advance; and the optional set-conditioning leans on an RNN that
imposes an arbitrary ordering on the support set.

**Prototypical Networks (Snell et al. 2017, NIPS).** Summarise each class by a single
*prototype* `c_k`, the mean of its embedded support examples, `c_k = (1/K) Σ_{i:y_i=k} f(x_i)`,
then classify a query by a softmax over **negative squared Euclidean distance** to the
prototypes, `p(y=k | x) = softmax_k(−‖f(x) − c_k‖²)`, trained by minimising the negative
log-probability of the true class over episodes. The choice of squared Euclidean distance is
principled: for the family of *Bregman divergences* (squared Euclidean is one), the
representative that minimises total divergence to a set of points is exactly the mean, so the
mean prototype is the matched cluster centre; and expanding the distance,
`‖f − c_k‖² = ‖f‖² − 2 c_kᵀf + ‖c_k‖²`, the `‖f‖²` term is constant across classes, so the
classifier is **linear in `f`** (weights `2c_k`, bias `−‖c_k‖²`). Empirically squared Euclidean
beats cosine — cosine is not a Bregman divergence, breaking the mean-as-optimal-centre argument.
In the one-shot case `c_k` is the single support point, so the prototype step reduces to comparing
the query against one support embedding per class; it is structurally close to Matching Networks,
but it is not identical unless the embedding and scoring function are made the same.
**Limitation it leaves open:** the comparison is again a *fixed* metric, and the squared-Euclidean
softmax is exactly a *linear* classifier in embedding space — so the method works only to the
extent that the embedding alone can render every novel class linearly separable with a
mean-centred Gaussian-like cluster; whenever the natural notion of "same class" is a more
intricate, non-linear relationship between query and class summary, a fixed element-wise metric
on top of the embedding cannot express it, and the whole burden falls on the embedding network.

**Convolutional Siamese Networks (Koch et al. 2015, ICML workshop).** Twin weight-sharing CNNs
embed two images; the network is trained on a *verification* objective — predict whether a pair
is same-class or different-class — and at test time a query is labelled by comparison to support
examples through the learned representation with a fixed/linear top. Core idea: cast recognition
as pairwise same/different and learn the representation that supports it. **Limitation it leaves
open:** the top-level comparison is fixed/linear, training is pairwise rather than the episodic
`C`-way task that matches test conditions, and the representation and comparison are not
meta-learned jointly and end-to-end for the few-shot task.

**Optimisation- and memory-based alternatives.** A second family adapts to the new task instead
of comparing within a fixed embedding. MAML (Finn et al. 2017) meta-learns a network
*initialisation* from which a few gradient steps on the support set yield a good task-specific
classifier; the Meta-Learner LSTM (Ravi & Larochelle 2017) goes further and meta-learns the
*optimiser* (an LSTM that emits the parameter updates). Memory/RNN approaches (MANN, Santoro
et al. 2016; Meta Networks, Munkhdalai & Yu 2017) iterate an RNN over the support examples and
accumulate task knowledge in hidden state or external memory. **Limitations they leave open:**
the optimisation methods require *fine-tuning at test time* (gradient steps on the support set),
which is precisely the fragile regime the data scarcity creates and which adds latency; the
memory methods carry the complexity of recurrence and the difficulty of ensuring the memory
reliably retains the relevant information without forgetting.

## Evaluation settings

The natural yardsticks already in use at this time, all under the episodic `C`-way `K`-shot
protocol with mean classification accuracy over many sampled test episodes:

- **Omniglot** (Lake et al. 2011): 1623 handwritten characters from 50 alphabets, 20 examples
  each, grayscale resized to 28×28; new classes augmented by 90°/180°/270° rotations; the common
  split uses 1200 characters (plus rotations) for training and the remaining 423 (plus rotations)
  for test. Evaluated 5-way and 20-way, 1-shot and 5-shot; accuracy averaged over many test
  episodes (e.g. 1000).
- **miniImageNet** (Vinyals et al. 2016; split of Ravi & Larochelle 2017): 100 ImageNet classes,
  600 colour images each, resized to 84×84, split 64/16/20 classes for train/validation/test.
  Evaluated 5-way 1-shot and 5-shot; accuracy averaged over 600 test episodes with 95%
  confidence intervals.
- Standard embedding backbone for fair comparison across methods: four convolutional blocks of
  64-filter 3×3 conv + batch normalisation + ReLU (with max-pooling), as used by the
  metric-based baselines. Optimisation by Adam (Kingma & Ba 2015), initial learning rate `1e-3`,
  annealed during training; models trained end-to-end from scratch with no additional dataset.

## Code framework

The episodic few-shot harness already exists for the baselines. What is *not* settled is how a
query should be scored against the support set, so the substrate is only the generic episodic
machinery: an embedding backbone that turns images into features, a method object that ingests the
support set, a `forward` that returns per-class scores for queries, and a loss. The single empty
slot is how support information is summarised and how a query is compared against it.

```python
import torch
from torch import Tensor, nn


def make_backbone() -> nn.Module:
    """The shared feature extractor (e.g. the standard four-conv embedding, or a ResNet).
    Produces a feature representation per image. Whether the slot below wants pooled
    vectors or spatial feature maps is itself a design choice left open."""
    raise NotImplementedError


class EpisodicFewShotMethod(nn.Module):
    """Generic C-way K-shot classifier. Given a labelled support set it must score each
    query against the C classes. The scoring rule remains open."""

    def __init__(self):
        super().__init__()
        self.backbone = make_backbone()
        # any extra modules the scoring rule needs go here
        # TODO: the scoring machinery to be designed

    def compute_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        # extract and store whatever summary of the support set forward() will need
        # TODO: how to summarise the K support examples of each of the C classes
        pass

    def forward(self, query_images: Tensor) -> Tensor:
        # return classification scores of shape (n_query, C)
        # TODO: how to score a query against the stored support summary
        raise NotImplementedError

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        # TODO: the training objective that shapes the scores
        raise NotImplementedError


# existing episodic training loop the method plugs into
def train(method, optimizer, episode_sampler):
    for support_images, support_labels, query_images, query_labels in episode_sampler:
        optimizer.zero_grad()
        method.process_support_set(support_images, support_labels)   # summarise support
        scores = method(query_images)                                # score queries
        loss = method.compute_loss(scores, query_labels)             # objective
        loss.backward()
        optimizer.step()                                             # Adam, lr 1e-3
```

The sampler draws one episode (support + query, both `C`-way `K`-shot) at a time;
`process_support_set` / `forward` / `compute_loss` are where the scoring rule must be supplied.
