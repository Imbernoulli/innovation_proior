# Context: batch active learning for neural networks (circa 2020-2021)

## Research question

In pool-based active learning we hold a large pool `U = {x_1, ..., x_n}` of unlabeled examples
and may pay to have the label of any one revealed. Because retraining a deep network after every
single query is prohibitively expensive, labels must be requested in *batches*: at each round we
commit to a set `S` of `B` points, pay for all their labels at once, fold them into the training
set, retrain, and repeat. The goal is to reach the lowest possible loss on the data distribution
with the fewest labels — equivalently, to make each batch as *informative* as possible for the
learner.

A good batch has to simultaneously be **uncertain** (points the model is unsure about, where a
label changes the model a lot), **diverse** (not `B` near-duplicates, since one label would then
settle the rest), and **representative** (covering the regions of input space that actually carry
weight under the pool distribution). These pull in different directions, and getting the trade-off
right is the whole problem. The precise difficulty is that the methods which perform well on deep
networks are largely heuristic and hard to reason about — it is unclear *why* they work, when
they will fail, or how to extend them past the classification setting they were built for — while
the methods that come with theory were derived for convex models and one-point-at-a-time querying,
and do not obviously survive the move to overparametrized nets with a representation that shifts
every round. Closing that gap — a batch acquisition rule that is *both* principled *and* tractable
at neural scale, and that degrades gracefully to convex models and to regression — is the goal.

## Background

**The probabilistic view of a network.** A classifier or regressor can be read as specifying a
conditional distribution `p(y | x, θ)` over labels: apply a softmax to the logits for
classification, or treat the outputs as the mean of a Gaussian for regression. Under this view the
natural loss is the negative log-likelihood `ℓ(x, y; θ) = -log p(y | x, θ)`, and fitting the model
is maximum-likelihood estimation. This is the bridge that lets one import classical statistical
machinery — written for MLE in convex models — into the neural setting.

**Fixed-design (transductive) objective.** For algorithm design it is cleaner to treat the
unlabeled pool itself as the distribution of interest and aim to minimize the pool loss
`L_U(θ) = E_{x ~ U} E_{y ~ p(·|x, θ*)} ℓ(x, y; θ̂)`, rather than the population loss directly; the
two are related by standard generalization arguments. Here `θ*` is the unknown true parameter.

**The Fisher information matrix.** The central object of classical estimation theory is the Fisher
information `I(x; θ) := E_{y ~ p(·|x, θ)} ∇²ℓ(x, y; θ)`, the expected Hessian of the per-example
log-loss. It is well known to govern the asymptotic distribution of the MLE: under regularity
conditions the estimator is asymptotically normal with covariance equal to the inverse Fisher, so
the Fisher is the precision with which a labeled example pins down the parameter. For a useful
class of models — linear regression, logistic and multiclass-logistic regression, and generalized
linear models more broadly — the Hessian of the loss does not depend on the label `y`, only on `x`
and `θ`. For multiclass logistic regression a direct computation gives, with `π` the softmax
vector, the per-example Fisher `I(x; W) = x x^T ⊗ (diag(π) - π π^T)`; for `k`-output regression
with noise covariance `Σ` it is `I(x; W) = x x^T ⊗ Σ^{-1}`. The pool Fisher is the average
`I_U(θ) = (1/|U|) Σ_{x ∈ U} I(x; θ)`.

**Optimal experimental design.** Classical design theory asks which experiments (here, which
points to label) to run to estimate `θ` best, and summarizes the resulting Fisher matrix by a
scalar. Two summaries dominate. *D-optimality* maximizes `det(I)` — the product of the eigenvalues
— which tends to push the directions that are *already* most informative. *A-optimality* minimizes
`tr(I^{-1})` — the sum of the inverse eigenvalues, which equals the average estimation variance of
the parameters — and so tends to shore up the *smallest* eigenvalues, the directions of *lowest*
information. The design literature presents these as a menu of criteria but offers little guidance
on which to use for a given downstream goal, and classically analyzes them in the single-parameter
or asymptotic regime rather than for a finite labeling budget.

**The convex MLE-error result.** A line of statistical active-learning work analyzes exactly how
the choice of which points to label feeds through to the MLE's error. For a sampling distribution
`Γ` over the pool with per-example Fisher averaged to `I_Γ(θ*)`, the expected excess
log-likelihood error of the MLE on `m` labels drawn from `Γ` is, to leading order and with a
matching lower bound, `tr(I_Γ(θ*)^{-1} I_U(θ*)) / m` (Chaudhuri, Kakade, Netrapalli & Sanghavi,
NeurIPS 2015). The same weighted-trace quantity also drops out of the Bayes risk for Bayesian
linear regression: with prior `θ* ~ N(0, λ^{-1} I)` and Gaussian noise of variance `σ²`, the MAP
estimate is ridge regression with regularizer `λσ²`, and for a labeled set `S`, writing
`Λ_S = Σ_{x ∈ S} x x^T + λσ² I` and `Σ = (1/n) Σ_i x_i x_i^T` the pool second moment, the Bayes
risk is exactly `σ² tr(Λ_S^{-1} Σ)` — a fact whose right-hand side, notably, contains no labels at
all. So two independent derivations — frequentist MLE error and Bayesian risk — produce
essentially the same weighted-trace functional, differing only in the regularizer `λ`.

**The state of deep active learning, and its pain points.** The dominant recipe on neural nets is
to retrain from scratch each round (warm-starting hurts), seed with ~100 random labels, and query
in batches. Diagnostically, the field had observed several things about *existing* methods that
frame the problem. Pure uncertainty methods (label the lowest-confidence points) pick redundant
near-duplicate batches, because a batch of similar uncertain points is wasteful — one label would
resolve the others. Pure diversity methods (cover the penultimate-layer space, e.g. coreset
selection) ignore which points the model is actually unsure about. The best-performing combined
method on deep nets at the time, BADGE, works well empirically but its behavior is "fairly
limited" in explanation, it cannot be run on regression at all, and — as becomes visible when one
deliberately tests it on a random, uninformative feature projection that mimics the convex
regime — it performs poorly when the high-norm feature directions are not the discriminative ones.
The theoretically clean methods, conversely, were derived for convex models with single-point
queries and lean on solving a semidefinite program, which does not scale to neural dimensionality.

## Baselines

**Random sampling.** Draw `B` points uniformly from `U`. No uncertainty, no diversity, no
representativeness — but a surprisingly strong baseline, especially in regression, and the
yardstick every method must beat. *Gap:* spends labels on points the model already handles.

**Confidence / least-confidence sampling** (Lewis & Gale 1994; Tong & Koller 2001 for the
margin/SVM version). Score each pool point by how unsure the model is — e.g. `max_y f(x; θ)_y`
(lowest top-class probability) or proximity to the decision boundary — and label the `B` most
uncertain. *Gap:* purely pointwise. It has no notion of a *batch*: the `B` most uncertain points
are typically clustered together (similar hard examples), so the batch is redundant and a single
label could have resolved much of it. It also requires a classification-style notion of
confidence, so it does not transfer to regression.

**Coreset / diversity sampling** (Sener & Savarese, ICLR 2018; Geifman & El-Yaniv 2017). Embed
each pool point with the network's penultimate layer and choose a batch that best *covers* that
embedding space — e.g. a `k`-center / facility-location selection so every unlabeled point is
close to some chosen point. *Gap:* it optimizes geometric coverage of the representation and is
blind to model uncertainty; it can spend its budget describing easy, already-learned regions.
Because it only uses the penultimate representation it carries no information about how a label
would move the model, and like other last-layer-geometry methods it leans on the learned
representation being good — on a poor or random feature basis its coverage is uninformative.

**BADGE — Batch Active Learning by Diverse Gradient Embeddings** (Ash, Zhang, Krishnamurthy,
Langford & Agarwal, ICLR 2020), the state of the art on deep classification. For each pool point
compute the *hallucinated* last-layer loss gradient `g_x = ∇_{θ_out} ℓ(x, ŷ; θ)` where
`ŷ = argmax_y f(x; θ)_y` is the model's own most-likely label — a `dk`-dimensional vector (`d` =
penultimate dimension, `k` = number of classes). The *length* of `g_x` is a lower bound on the
gradient norm any true label would induce, so it measures uncertainty (confident points give tiny
gradients); the *direction* of `g_x` records how that label would push the model. BADGE then
selects a batch with large Gram determinant in this gradient space — points that are
simultaneously high-magnitude (uncertain) and linearly independent (diverse) — sampling them with
`k`-means++ seeding as a cheap stand-in for a `k`-determinantal point process. *Gaps:* (i) the
explanation is heuristic — the Gram-determinant objective is motivated by DPP intuition but only
loosely connected to any error quantity, so it is unclear when it should work; (ii) each `g_x` is
a single vector — effectively a rank-one summary of the point — discarding the rest of the
per-example second-order structure; (iii) the determinant objective scores a batch purely by its
own internal spread and has no channel through which the *pool* distribution can enter — it cannot
weight directions by how much they matter under `U`; (iv) the hallucinated-label construction is
intrinsically classification-bound and has no regression analogue; and (v) empirically it stumbles
on feature bases where the highest-norm directions are not the discriminative ones.

**The classical two-phase MLE design** (Chaudhuri et al. 2015). In the convex, well-specified
setting: label a small random batch, fit a crude `θ_1`, then choose a labeling distribution that
minimizes `tr(I_Γ(θ_1)^{-1} I_U(θ_1))`, relabel, and refit. Backed by matched upper and lower
bounds showing this trace is the right thing to minimize and that, in the convex case, a *single*
extra round suffices. *Gaps:* the selection step is cast as a semidefinite program, which is
infeasible in high dimensions; the analysis assumes a convex, well-specified model with a fixed
representation, none of which holds for a deep network whose internal features move every round;
and per-example `I(x; θ)` is a `dk × dk` matrix that is far too large to form, store, or invert
for every pool point at neural scale.

## Evaluation settings

The natural yardsticks already in use for deep batch active learning:

- **Datasets / architectures.** MNIST and OpenML tabular datasets (e.g. OpenML 155, and small
  classification sets such as letter recognition, spambase, splice) with a multilayer perceptron
  (one hidden ReLU layer of ~128 units); SVHN and CIFAR-10 with both an MLP and an 18-layer
  ResNet. For regression: predicting the year of yearbook photographs (ResNet), predicting Austin
  rainfall from meteorological features (linear model), and classification datasets re-cast as
  regression onto one-hot targets.
- **Protocol.** Seed each learner with ~100 randomly labeled points; at each round select a batch,
  query labels, and *retrain from a fresh random initialization* (no warm-starting) until high
  training accuracy. Sweep batch sizes (e.g. 10, 100, 1000) since the uncertainty/diversity
  balance shifts with `B`. Repeat each experiment several times with different seeds and report
  standard error. A controlled variant fits a representation on half the data and does active
  learning in that fixed embedding, plus a *random Gaussian projection* of the inputs to mimic the
  convex regime where the learner cannot shape its own features.
- **Metrics.** Test accuracy after a fixed label budget; area under the accuracy-vs-labels
  learning curve (sample efficiency); for regression, squared error vs labels. Aggregate
  comparisons via pairwise significance counts across the many (dataset, architecture, batch-size)
  settings. Wall-clock selection time is also tracked, since per-round selection cost matters when
  many rounds are run.

## Code framework

The acquisition rule plugs into a fixed active-learning harness. The harness owns the pool, the
labeled mask, model (re)training, and the round loop; the *only* thing left open is the function
that, given the current trained model and the unlabeled pool, returns the indices of the next `B`
points to label. The substrate below is exactly the prior-art machinery that already exists:
penultimate-layer embeddings, softmax probabilities, and the last-layer loss gradient at a chosen
label — the same primitives the diversity and gradient baselines are built from. The one empty
slot is the batch acquisition rule.

```python
import numpy as np


class Strategy:
    """Fixed active-learning harness. Owns the pool, the labeled mask, and model
    (re)training; exposes the standard last-layer primitives the prior baselines use.
    A query strategy only has to choose which unlabeled points to label next."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X, self.Y = X, Y               # pool features, labels (labels hidden until queried)
        self.idxs_lb = idxs_lb              # boolean mask of already-labeled points
        self.n_pool = len(Y)
        self.clf = net                      # current trained model
        self.handler, self.args = handler, args

    # ---- primitives that already exist (used by the baselines) ----
    def predict_prob(self, X, Y):
        """Softmax probabilities, shape [len(X), n_classes]."""
        ...

    def get_embedding(self, X, Y):
        """Penultimate-layer embeddings x^L, shape [len(X), emb_dim].
        (This is what coreset/diversity selection operates on.)"""
        ...

    def get_grad_embedding(self, X, Y):
        """Last-layer loss gradient at the model's most-likely label, flattened to
        [len(X), emb_dim * n_classes] — the single hallucinated-gradient vector."""
        ...

    def get_exp_grad_embedding(self, X, Y):
        """Per-class last-layer loss gradients, each class's gradient scaled by the
        square root of that class's predicted probability, shape
        [len(X), n_classes, emb_dim * n_classes]."""
        ...

    def train(self):
        """Retrain the model from scratch on the current labeled set."""
        ...

    def query(self, n) -> np.ndarray:
        # Return n indices into the unlabeled pool to send to the oracle.
        # TODO: the batch acquisition rule we will design.
        raise NotImplementedError


# the round loop the strategy plugs into (already exists)
def active_learning_loop(strategy, n_rounds, batch_size):
    for _ in range(n_rounds):
        idxs = strategy.query(batch_size)          # choose the next batch
        strategy.idxs_lb[idxs] = True              # oracle labels them
        strategy.update(strategy.idxs_lb)
        strategy.train()                           # retrain from scratch on the larger set
```

The harness supplies the trained model, the pool, and the last-layer primitives; `query` is where
the batch acquisition rule will live.
