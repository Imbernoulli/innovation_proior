# Context: batch active learning for deep networks (circa 2018-2019)

## Research question

Deep neural networks need a lot of labeled data, and labels are expensive. Pool-based
active learning is the lever: a large unlabeled pool `U` is available, and at each round the
algorithm chooses a *batch* of `B` examples whose labels it requests from an oracle, then
retrains the model on the enlarged labeled set and repeats. The goal is to reach a target
accuracy with as few total labels as possible — to spend the label budget on the examples
that teach the model the most.

The precise difficulty is the *batch*. Classical active-learning theory selects one point at
a time, refitting after each label. For deep nets that is doubly impossible: a single example
has negligible effect on a network with millions of parameters, and retraining a deep net to
convergence after every single query is computationally intractable. So labels must be
acquired `B` at a time. But a batch acquired by scoring each example *independently*
introduces a new failure that the one-at-a-time setting never had — the `B` points can be
redundant with each other, all carrying the same information, so the batch teaches the model
far less than `B` independent labels should.

There is a second, sharper constraint that is special to active learning and easy to miss. In
ordinary supervised learning you tune hyperparameters on a validation set for free. In active
learning, *every* change of a hyperparameter generally causes the algorithm to query a
*different* set of examples, and you have to pay for those labels. A hyperparameter sweep is
therefore a label-budget sweep — ruinously expensive. So a deployable batch acquisition rule
must "just work" at fixed hyperparameters, to a degree supervised learning never demands.

What a solution would have to achieve: a single batch-selection rule that is robust across
network architecture (a small MLP, a deep ResNet, a VGG), across batch size (from a hundred
to tens of thousands), and across dataset, **without** any hyperparameter that has to be
re-tuned per setting — because tuning it would itself cost labels.

## Background

The field at this point has split active learning into two broad strategies, and the central
pre-method fact is that *each works only in part of the space*.

**Uncertainty sampling.** Query the examples the current model is least sure about, on the
intuition that confidently-classified points teach nothing new. With a `K`-class softmax model
producing class probabilities `p = f(x; theta)`, the standard per-example scores are:
least-confidence `1 - max_i p_i`; margin `p_(1) - p_(2)` (gap between the top two class
probabilities, smaller = more uncertain); and entropy `H(p) = -sum_i p_i log p_i` (larger =
more uncertain). These are cheap and often strong with linear models and at small batch.

**Representative / diversity sampling.** Pick a batch that *covers* the unlabeled
distribution, so that fitting the model on the batch is a good surrogate for fitting it on the
whole pool, independent of any label. Concretely this is done in the network's
*penultimate-layer* feature space — the `d`-dimensional representation `z(x)` just before the
final linear classifier — selecting a geometrically spread-out set of `z`'s.

The empirical landscape that motivates everything below — observed about *existing* methods,
knowable before any new method — is that these two families are complementary and fragile.
Uncertainty methods tend to win at small batch size and with simple models (MLPs), and to
degrade badly at large batch, where they pile up near-duplicate uncertain points. Diversity /
representative methods tend to win at large batch and when the architecture has strong
inductive biases (convolutional nets on images) so the penultimate representation is
meaningful, and to degrade — sometimes below uniform random selection — on harder data or
weaker architectures where that representation is not informative. Which regime you are in is
itself a function of the (often unknown) statistics of the data, so a practitioner has no
reliable way to pick the right family in advance. Worse, deep-net softmax outputs are known to
be overconfident and poorly calibrated, which undermines uncertainty scores directly.

Two further pre-method facts about *how* deep nets learn frame the design space. First, deep
nets are trained by gradient descent, so the natural currency of "how much will this example
change the model" is the *gradient of the loss* the example induces — a large induced gradient
means a large parameter update. Second, for a softmax network with cross-entropy loss there is
a clean closed form for the last-layer gradient, which makes the last layer a cheap and
analytically convenient place to measure that induced change (the full-parameter gradient is
enormous; the last-layer gradient is not).

## Baselines

These are the prior methods a new batch rule would be measured against and would react to.

**Least-confidence / margin / entropy uncertainty sampling** (Lewis & Gale 1994; Tong & Koller
2001; Roth & Small 2006 for margin; Wang & Shang 2014). Score every pool example by a
predictive-uncertainty functional of `p = f(x; theta)` and take the top `B`. Core idea: the
model learns most from points near its decision boundary. **Limitation:** the score is a
function of one example in isolation, with no term coupling the chosen points to each other, so
in a batch it repeatedly selects examples from the *same* uncertain region — a cluster of
near-identical points where a single label would have resolved the model's uncertainty about
all of them. The redundancy grows with batch size, so the method's edge erodes as `B` grows,
and overconfident deep-net probabilities make the raw scores unreliable.

**Core-Set / representative sampling** (Sener & Savarese 2018). Frame active learning as
*core-set selection*: choose a labeled subset such that a model trained on it is competitive
with a model trained on the entire pool. They bound the population risk of the subset-trained
model by (training error) + (generalization error) + a *core-set loss* — the gap between the
average loss over the full pool and over the selected subset — and show that minimizing the
core-set loss reduces to the `k`-Center objective: pick `B` centers minimizing the largest
distance from any pool point to its nearest center, in penultimate-layer feature space. They
solve it with a greedy furthest-first traversal. **Limitation:** the criterion is purely
geometric in the representation — it has no notion of model uncertainty or label
informativeness, so it can spend the budget covering regions the model already classifies
correctly. It hinges entirely on the penultimate representation being meaningful; when it is
not (hard data, weak architecture), the chosen "representative" batch is no better, and can be
worse, than uniform random. Effectiveness also decays as the number of classes and the feature
dimension grow.

**Expected gradient length (EGL)** (Settles, Craven & Ray 2008; Huang et al. 2016; Zhang et al.
2017). Score an example by the expected magnitude of the gradient it would induce, averaging
over the unknown label under the model's own predictive distribution:
`EGL(x) = sum_y p(y | x) * || grad_theta L(x, y; theta) ||`. A large expected gradient means a
large expected model update, hence an informative example. **Limitation:** it is still a
per-example score with no batch-diversity term, so it inherits the duplicate-batch pathology;
it averages a gradient over all `K` candidate labels (more computation), and empirical studies
report it selects quite different points than entropy without a principled way to also enforce
diversity within a batch.

**Active Learning by Learning (ALBL)** (Hsu & Lin 2015). A bandit-style meta-strategy that, at
each round, chooses *which* of several base acquisition rules (e.g. a representative one and an
uncertainty one) to run, treating the choice as a sequential decision problem. **Limitation:**
it can only pick among the base rules it is given — it does not build a single criterion that
captures both properties at once, and it inherits whatever weakness the active base rule has in
the current regime.

**Determinantal point processes (`k`-DPP)** (Kulesza & Taskar 2011; survey 2012). A
probabilistic model over *sets*: draw a size-`k` subset `Y` with probability proportional to
`det(L_Y)`, where `L` is the Gram matrix of the items' feature vectors. The determinant of a
Gram matrix equals the squared product of the vectors' lengths times the squared volume they
span, so it is large exactly when the chosen vectors are both *long* (high quality) and
*mutually near-orthogonal* (diverse) — a single object that rewards quality and diversity
together, with no tradeoff coefficient. As the batch size shrinks, length dominates; as it
grows, the volume (linear-independence) term dominates. **Limitation:** sampling from a `k`-DPP
is computationally heavy — exact samplers are high-order polynomial in the batch size and
feature dimension, and MCMC samplers face slow mixing — so it does not scale to large pools and
large batches (it runs out of memory at the largest batch sizes of interest).

**`k`-means++ seeding** (Arthur & Vassilvitskii 2007). Not an active-learning method but a
classic seeding primitive: to initialize `k`-means, pick the first center uniformly at random,
then iteratively sample each next center from the ground set with probability proportional to
its *squared distance to the nearest already-chosen center* (`D^2` weighting). It comes with a
guarantee that the resulting potential is within `8(ln k + 2)` of optimal in expectation. The
`D^2` rule pulls new centers toward points that are far from those already picked, i.e. it
produces a spread-out, diverse set of seeds, and it does so cheaply (a few passes over the
data, no matrix algebra) and with no tunable knob.

## Evaluation settings

The natural yardstick already in use for deep batch active learning:

- **Datasets:** image benchmarks SVHN, CIFAR-10, MNIST, plus several non-image tabular
  datasets from the OpenML repository (chosen so that neural nets beat linear models on them).
  In the tabular regime the relevant OpenML tasks include letter recognition, spambase, and
  splice.
- **Architectures:** a two-layer ReLU MLP, an 18-layer convolutional ResNet, and an 11-layer
  VGG — deliberately spanning weak-to-strong inductive bias. Penultimate embedding dimension on
  the order of a few hundred to ~1000.
- **Protocol:** start from `M = 100` randomly labeled examples; query in batches of size `B`
  swept across a wide range (e.g. 100 / 1000 / 10000), over a fixed number of rounds, retraining
  the model from scratch each round (no warm starting), optimizing cross-entropy with Adam to a
  high training accuracy; repeat each configuration several times for error bars; no learning-rate
  schedule or data augmentation.
- **Metrics:** test accuracy as a function of the number of labeled examples (the learning
  curve), and the area under that learning curve, which captures label efficiency across the
  whole run; for the cross-setting comparison, pairwise win/loss counts (two-sided `t`-test) and
  cumulative distribution functions of errors normalized to the random baseline.

## Code framework

The acquisition rule plugs into a fixed pool-based active-learning harness that already exists:
it owns the labeled mask, retrains the model each round, and exposes the model's outputs.
Nothing about *which* examples to pick is settled — that selection rule is exactly what is to
be designed. The substrate is only the generic machinery: a `Strategy` base class that holds
the pool features `X`, labels `Y`, the boolean labeled mask `idxs_lb`, and the current trained
network, and gives the acquisition rule read access to the model through a few primitives —
softmax probabilities for pool points, the penultimate-layer embeddings, and (since deep nets
learn by gradients) the last-layer loss-gradient features. The one empty slot is `query(n)`:
given the number of points to acquire, return the indices of the unlabeled examples to label.

```python
import numpy as np


class Strategy:
    """Generic pool-based active-learning harness. Owns the pool and the current
    trained model; the acquisition rule reads the model through these primitives.
    The retraining loop and data management are fixed — only query() is to be designed."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X                       # pool features [n_pool, n_features]
        self.Y = Y                       # labels (LongTensor [n_pool])
        self.idxs_lb = idxs_lb           # boolean labeled mask [n_pool]
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)

    # --- model read-out primitives that already exist ---
    def predict_prob(self, X, Y):
        """Softmax probabilities p = f(x; theta), shape [len(X), n_classes]."""
        ...

    def get_embedding(self, X, Y, return_probs=False):
        """Penultimate-layer features z(x), shape [len(X), emb_dim]
        (optionally also the softmax probabilities)."""
        ...

    def get_grad_embedding(self, X, Y):
        """Last-layer loss-gradient features, shape [len(X), emb_dim * n_classes].
        For each example, the per-class blocks of the gradient of the (cross-entropy)
        loss w.r.t. the final linear layer's weights."""
        ...

    def query(self, n):
        # TODO: the batch acquisition rule we will design.
        #       Return n indices into self.X of currently-unlabeled examples to label.
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
```

The harness hands `query()` the model's read-outs for the unlabeled pool and asks for `n`
indices; the selection rule itself is the empty slot.
