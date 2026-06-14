# Context: learning to recognize a new visual class from a single labeled example (circa 2015-2016)

## Research question

A child shown one picture of a giraffe in a book can pick giraffes out of a crowd afterward;
the best deep image classifiers of the day need hundreds or thousands of labeled examples per
class. The problem is to close that gap in the extreme regime called *one-shot* (more generally
*N-way k-shot*): given a small *support set* of N classes that were **never seen during
training**, with only k labeled examples per class (k as small as 1), classify a fresh
*query* image into one of those N classes. Random performance is 1/N, so the bar is to do
much better than chance from a handful of examples.

The hard constraint that shapes everything is *no fine-tuning at test time*. The novel classes
arrive only at evaluation, as a support set, and the system must produce sensible labels for
them **without any weight updates** — no gradient steps, no per-task retraining. So the
contribution being sought is a reusable algorithmic component — a way to summarize a support
set, to compare a query against it, and to turn that comparison into a label distribution —
that is learned once on base classes and then applied unchanged to brand-new classes. It must
be differentiable end-to-end (so the comparison and the feature extractor train jointly), and
it must accept a support set whose size and class membership change from one task to the next.

## Background

Two families of models sit at opposite ends of a trade-off, and the tension between them is
the whole setup.

**Parametric deep classifiers** (Krizhevsky et al. 2012 on vision; Hinton et al. 2012 on
speech; Mikolov et al. 2010 on language) are the state of the art on large datasets, but they
are notoriously data-hungry. The reason is structural: the knowledge lives in the weights, and
the weights are moved only slowly, by many stochastic-gradient steps over many examples. Data
augmentation and regularization soften overfitting in the low-data regime but do not remove
the dependence on many weight updates. A fresh class cannot be absorbed in one shot, because
one example yields one tiny gradient step; and pushing the network hard on a few new examples
risks catastrophic forgetting of everything else.

**Non-parametric models** — nearest neighbours, kernel density estimators, locally weighted
learning (Atkeson, Moore & Schaal 1997) — sit at the other end. They assimilate a new example
the instant it is stored: no training, no forgetting, and the effective capacity grows with the
data because the model *is* the data. Their decision surfaces are nonlinear and the quality of
predictions improves automatically as more reference points arrive. The catch is well known:
their accuracy is only as good as the distance metric used to decide what "near" means, and a
raw pixel or off-the-shelf-feature metric is weak. There is no learned representation tailoring
the geometry to the task.

So the prevailing wisdom splits the difference by *borrowing features*: train a deep classifier
on the base classes, then strip off the softmax and use the penultimate-layer activations as a
generic feature vector, doing nearest-neighbour matching in that space (the DeCAF recipe,
Donahue et al. 2014). It works moderately well, and it is the natural thing to try. But the
features were optimized to separate the *base* classes through a softmax — not to make
instance-to-instance comparison meaningful for *unseen* classes presented as a support set.

Several threads from the surrounding literature are load-bearing here:

- **Metric learning with soft neighbours.** Neighbourhood Components Analysis (Goldberger,
  Roweis, Hinton & Salakhutdinov 2004) makes the kNN metric *learnable* by replacing the
  discontinuous leave-one-out kNN error with a differentiable surrogate built on *stochastic*
  neighbour assignments. Its construction is detailed under Baselines below; the key transplant
  is that a softmax over distances in an embedding, summed over same-class points, is a smooth,
  trainable stand-in for nearest-neighbour classification.

- **Differentiable attention over an external memory.** Content-based attention (Bahdanau, Cho
  & Bengio 2015) reads a memory matrix by computing alignment scores between a query and each
  memory slot, softmax-normalizing them into weights, and returning the weighted sum of slots.
  Memory Networks (Weston, Chopra & Bordes 2014) and Neural Turing Machines (Graves, Wayne &
  Danihelka 2014) generalize this into "computer-like" architectures with addressable memory.
  The property that matters for us: a content-based read is invariant to the *order* of the
  memory slots — shuffling them does not change the read-out.

- **Encoding a set, not a sequence.** A support set has no natural order, yet RNN encoders are
  order-dependent and a plain bag-of-words sum is order-invariant but representationally crude
  (Vinyals, Bengio & Kudlur 2016, *Order Matters*). Their Read-Process-Write model introduces a
  "Process" block — an LSTM that takes **no inputs** and performs T steps of computation,
  each step reading the memory by content attention and folding the read-out back into its
  state. Concretely, with memory vectors m_i and a state q,

  ```
  q_t       = LSTM(q*_{t-1})           # LSTM with no input, evolves a recurrent state
  e_{i,t}   = f(m_i, q_t)              # alignment score, e.g. a dot product
  a_{i,t}   = softmax_i(e_{i,t})       # attention weights, normalized over memory index i
  r_t       = sum_i a_{i,t} m_i        # attention read-out
  q*_t      = [q_t , r_t]              # concat state with read-out -> next step
  ```

  After T steps the state q*_T is a permutation-invariant embedding of the set, and T adds
  "depth" to the attention computation. This is the canonical way, at the time, to read a set
  with iterated attention.

- **Meta-learning by treating learning itself as the task.** Memory-augmented neural networks
  (Santoro et al. 2016) train an LSTM to "learn to learn" from data presented *sequentially*,
  binding examples to their labels in memory across a sequence so that later queries can be
  answered. The empirical observation that grounds the whole training regime: a model trained
  to do something in conditions that differ from test conditions underperforms one trained in
  matched conditions — the engine of generalization in the low-data regime is making the
  training episodes look like the test episodes.

## Baselines

These are the prior methods a new one-shot classifier would be measured against and would
react to.

**k-Nearest Neighbours / kernel density estimation on fixed features.** Store the support
examples; classify a query by a (possibly distance-weighted) vote of its nearest stored
neighbours, in pixel space or in penultimate-layer features from a base-class classifier
(Donahue et al. 2014). Core algorithm: rank support points by a distance, take the k closest,
vote. Strengths: instant assimilation of new classes, nonlinear surfaces, no test-time
training. **Gap:** the metric is fixed and is not learned for the one-shot task — pixel
distance is weak, and base-class softmax features were optimized to separate the *training*
classes, so instance-to-instance comparison on *unseen* classes is only incidental to what the
features were trained for.

**Neighbourhood Components Analysis (Goldberger, Roweis, Hinton & Salakhutdinov 2004).** Learn
a *linear* transformation A (a Mahalanobis metric Q = AᵀA) so that kNN works well in the
transformed space. The leave-one-out kNN error is discontinuous in A, so NCA replaces it with
a differentiable surrogate using *soft* neighbour assignments: point i selects point j as its
neighbour with probability

```
p_ij = exp(-||A x_i - A x_j||^2) / sum_{k != i} exp(-||A x_i - A x_k||^2),    p_ii = 0,
```

a softmax over negative squared distances in the embedding. The probability that i is correctly
classified is the mass on its own class, p_i = sum_{j in C_i} p_ij with C_i = {j : c_j = c_i},
and the objective maximized is the expected number of correctly classified points,
f(A) = sum_i p_i, optimized by gradient ascent. **Gap:** the comparison is *pairwise* over the
whole training set and the map A is *linear*, so the representational power is limited; and the
objective is point-against-all-other-points, not aligned with a *multi-way, one-shot* task in
which a query is compared against a small, freshly sampled labeled support set treated as a
unit.

**Convolutional Siamese networks (Koch, Zemel & Salakhutdinov 2015).** Train a twin CNN on a
*same-or-different* verification task (pairs of images, predict whether they share a class),
then at test time use the learned features for nearest-neighbour matching against the support
set. Core idea: a powerful non-linear embedding learned through a binary pairwise objective.
**Gap:** the training objective (verify a pair) is a *proxy* for the actual task (pick the best
of N support classes for a query); the mismatch shows up as accuracy that holds up at higher
shot counts but degrades sharply in the one-shot limit.

**Memory-augmented neural networks for meta-learning (Santoro et al. 2016).** An LSTM with an
external memory is trained to absorb a *sequence* of labeled examples and answer queries about
them, learning a learning procedure. **Gap:** it treats the labeled examples as an ordered
sequence fed step by step, so the model's read of the data can depend on presentation order,
whereas a support set has no order; and the recurrent absorption is a learned black box rather
than an explicit comparison against the stored examples.

**Base-class softmax classifier + transferred features.** Train an ordinary N-way softmax
classifier on the base classes (excluding the test classes), then use the last-layer features
for cosine or softmax nearest-neighbour matching, optionally fine-tuning on the support set.
**Gap:** the features are shaped by the base-class decision boundary, not by the comparison
operation that one-shot evaluation actually performs; fine-tuning on one example per novel
class overfits massively.

## Evaluation settings

The natural yardsticks already in use, all in the N-way k-shot episodic protocol — give the
method k labeled examples from each of N classes that were not trained on, then classify a
disjoint batch of unlabeled examples into one of those N classes; random performance is 1/N.
Unless stated otherwise, training is on the complement of a held-out label subset L′ and
testing is one-shot on L′, so the test classes are never seen during training.

- **Omniglot** (Lake, Salakhutdinov, Gross & Tenenbaum 2011): 1623 handwritten characters from
  50 alphabets, each drawn by 20 different people — "the transpose of MNIST", many classes with
  few examples each, ideal for small-scale one-shot. Standard setup: train on a subset of
  characters, evaluate on held-out characters, with data augmentation by 90° rotations; images
  resized to 28×28. A common embedding CNN stacks four modules of {3×3 conv with 64 filters,
  batch normalization (Ioffe & Szegedy 2015), ReLU, 2×2 max-pool}, taking 28×28 down to a
  1×1×64 feature.
- **ImageNet / ILSVRC-2012** (Russakovsky et al. 2015): full-scale natural images. Held-out
  splits define one-shot tasks on classes excluded from training; a deep convolutional backbone
  (VGG-style, Simonyan & Zisserman 2014, or Inception, Szegedy et al. 2015) provides features.
  A smaller variant carved from it — 100 classes × 600 colour images at 84×84, split into
  train/test classes — fits in memory for rapid prototyping.
- **Penn Treebank** (Marcus, Marcinkiewicz & Santorini 1993): used to pose an analogous one-shot
  *language* task — given a query sentence with a blank and a support set of sentences each with
  a blank and a one-hot label, pick the support label that best matches the query.
- Backbone in the present setting: ResNet-12 producing a 640-dim feature vector. Protocol:
  episodic training (e.g. 500 tasks per epoch over many epochs, 5-way 5-shot); evaluation as
  mean classification accuracy over a fixed number of test episodes (e.g. 600), higher better.
  Benchmarks evaluated 5-way 5-shot include miniImageNet, a CIFAR-100-derived set, and a
  fine-grained bird set (CUB-200).

## Code framework

The component plugs into an episodic few-shot harness that already exists. A backbone maps
images to feature vectors; an outer loop samples N-way k-shot tasks, hands the support set
(images + labels) to the classifier so it can prepare whatever it needs, then asks it to score
a batch of query images into N columns and computes a loss. Everything about *how* a query is
compared to the support set — how the support set is summarized, what similarity is used, how
that becomes a label distribution — is exactly what is to be designed, so the substrate is only
the generic machinery: feature extraction, a place to stash support information, a scoring
call, and a loss.

```python
import torch
from torch import Tensor, nn


def make_backbone(use_pooling: bool = True) -> nn.Module:
    """Existing feature extractor (e.g. ResNet-12). With pooling, returns a
    (n_images, feature_dimension) tensor; without, returns feature maps."""
    ...


class FewShotClassifier(nn.Module):
    """Generic episodic few-shot classifier the harness drives.

    The harness calls process_support_set(...) once per task, then forward(...)
    on the queries. compute_features runs the backbone; the comparison rule that
    turns features into class scores is exactly what we will design."""

    def __init__(self, backbone: nn.Module, use_softmax: bool = False):
        super().__init__()
        self.backbone = backbone
        self.use_softmax = use_softmax

    def compute_features(self, images: Tensor) -> Tensor:
        return self.backbone(images)                 # (n_images, feature_dimension)

    def softmax_if_specified(self, scores: Tensor) -> Tensor:
        return scores.softmax(-1) if self.use_softmax else scores

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        # TODO: extract and store whatever summary of the support set
        #       the comparison rule we design will need.
        pass

    def forward(self, query_images: Tensor) -> Tensor:
        # TODO: the comparison rule we will design — turn query features and the
        #       stored support information into class scores of shape (n_query, n_way).
        pass

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        # TODO: the training loss matching whatever forward() returns.
        pass


# existing episodic training loop the classifier plugs into
def train_episode(model, support_images, support_labels, query_images, query_labels, optimizer):
    optimizer.zero_grad()
    model.process_support_set(support_images, support_labels)   # prepare from the support set
    scores = model(query_images)                                # score queries into n_way columns
    loss = model.compute_loss(scores, query_labels)            # loss on the query batch
    loss.backward()
    optimizer.step()
```

The single empty slot is the comparison rule: what `process_support_set` stores, what `forward`
computes to turn query features plus support information into `(n_query, n_way)` scores, and the
loss that matches it.
</content>
