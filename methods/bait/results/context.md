# Context: batch active learning for neural networks (circa 2020-2021)

## Research question

In pool-based active learning we hold a large pool `U = {x_1, ..., x_n}` of unlabeled examples
and may pay to have the label of any one revealed. Because retraining a deep network after every
single query is prohibitively expensive, labels are requested in *batches*: at each round we
commit to a set `S` of `B` points, pay for all their labels at once, fold them into the training
set, retrain, and repeat. The goal is to reach the lowest possible loss on the data distribution
with the fewest labels — to make each batch as *informative* as possible for the learner.

A batch is described along several axes: how **uncertain** its points are (points the model is
unsure about, where a label changes the model a lot), how **diverse** it is (avoiding `B`
near-duplicates, since one label would then settle the rest), and how **representative** it is
(covering the regions of input space that carry weight under the pool distribution). The question
is what batch acquisition rule to use, and how it should behave across classification and
regression, on overparametrized nets whose representation shifts every round as well as on convex
models.

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
log-loss. It governs the asymptotic distribution of the MLE: under regularity conditions the
estimator is asymptotically normal with covariance equal to the inverse Fisher, so the Fisher is
the precision with which a labeled example pins down the parameter. For a useful class of models —
linear regression, logistic and multiclass-logistic regression, and generalized linear models more
broadly — the Hessian of the loss does not depend on the label `y`, only on `x` and `θ`. For
multiclass logistic regression a direct computation gives, with `π` the softmax vector, the
per-example Fisher `I(x; W) = x x^T ⊗ (diag(π) - π π^T)`; for `k`-output regression with noise
covariance `Σ` it is `I(x; W) = x x^T ⊗ Σ^{-1}`. The pool Fisher is the average
`I_U(θ) = (1/|U|) Σ_{x ∈ U} I(x; θ)`.

**Optimal experimental design.** Classical design theory asks which experiments (here, which
points to label) to run to estimate `θ` best, and summarizes the resulting Fisher matrix by a
scalar. Two summaries dominate. *D-optimality* maximizes `det(I)` — the product of the eigenvalues
— which tends to push the directions that are *already* most informative. *A-optimality* minimizes
`tr(I^{-1})` — the sum of the inverse eigenvalues, which equals the average estimation variance of
the parameters — and so tends to shore up the *smallest* eigenvalues, the directions of *lowest*
information. The design literature presents these as a menu of criteria and classically analyzes
them in the single-parameter or asymptotic regime.

**The convex MLE-error result.** A line of statistical active-learning work analyzes how the
choice of which points to label feeds through to the MLE's error. For a sampling distribution
`Γ` over the pool with per-example Fisher averaged to `I_Γ(θ*)`, the expected excess
log-likelihood error of the MLE on `m` labels drawn from `Γ` is, to leading order and with a
matching lower bound, `tr(I_Γ(θ*)^{-1} I_U(θ*)) / m` (Chaudhuri, Kakade, Netrapalli & Sanghavi,
NeurIPS 2015). The same weighted-trace quantity also drops out of the Bayes risk for Bayesian
linear regression: with prior `θ* ~ N(0, λ^{-1} I)` and Gaussian noise of variance `σ²`, the MAP
estimate is ridge regression with regularizer `λσ²`, and for a labeled set `S`, writing
`Λ_S = Σ_{x ∈ S} x x^T + λσ² I` and `Σ = (1/n) Σ_i x_i x_i^T` the pool second moment, the Bayes
risk is exactly `σ² tr(Λ_S^{-1} Σ)` — a right-hand side that contains no labels at all. So two
independent derivations — frequentist MLE error and Bayesian risk — produce essentially the same
weighted-trace functional, differing only in the regularizer `λ`.

**The state of deep active learning.** The dominant recipe on neural nets is to retrain from
scratch each round (warm-starting hurts), seed with ~100 random labels, and query in batches. Pure
uncertainty methods label the lowest-confidence points; pure diversity methods cover the
penultimate-layer space, e.g. coreset selection. The best-performing combined method on deep nets
at the time is BADGE, which scores points by hallucinated last-layer gradients and selects a
batch with large Gram determinant in that gradient space. The theoretically clean methods, by
contrast, were derived for convex models with single-point queries and solve a semidefinite
program to choose the labeling distribution.

## Baselines

**Random sampling.** Draw `B` points uniformly from `U`. No uncertainty, no diversity, no
representativeness — a strong baseline, especially in regression, and the yardstick every method
is measured against.

**Confidence / least-confidence sampling** (Lewis & Gale 1994; Tong & Koller 2001 for the
margin/SVM version). Score each pool point by how unsure the model is — e.g. `max_y f(x; θ)_y`
(lowest top-class probability) or proximity to the decision boundary — and label the `B` most
uncertain. It is pointwise, with no explicit notion of a batch, and uses a classification-style
notion of confidence.

**Coreset / diversity sampling** (Sener & Savarese, ICLR 2018; Geifman & El-Yaniv 2017). Embed
each pool point with the network's penultimate layer and choose a batch that best *covers* that
embedding space — e.g. a `k`-center / facility-location selection so every unlabeled point is
close to some chosen point. It optimizes geometric coverage of the penultimate representation.

**BADGE — Batch Active Learning by Diverse Gradient Embeddings** (Ash, Zhang, Krishnamurthy,
Langford & Agarwal, ICLR 2020), the state of the art on deep classification. For each pool point
compute the *hallucinated* last-layer loss gradient `g_x = ∇_{θ_out} ℓ(x, ŷ; θ)` where
`ŷ = argmax_y f(x; θ)_y` is the model's own most-likely label — a `dk`-dimensional vector (`d` =
penultimate dimension, `k` = number of classes). The *length* of `g_x` is a lower bound on the
gradient norm any true label would induce, so it measures uncertainty (confident points give tiny
gradients); the *direction* of `g_x` records how that label would push the model. BADGE then
selects a batch with large Gram determinant in this gradient space — points that are
simultaneously high-magnitude (uncertain) and linearly independent (diverse) — sampling them with
`k`-means++ seeding as a cheap stand-in for a `k`-determinantal point process.

**The classical two-phase MLE design** (Chaudhuri et al. 2015). In the convex, well-specified
setting: label a small random batch, fit a crude `θ_1`, then choose a labeling distribution that
minimizes `tr(I_Γ(θ_1)^{-1} I_U(θ_1))`, relabel, and refit. Backed by matched upper and lower
bounds showing this trace is the right thing to minimize and that, in the convex case, a *single*
extra round suffices. The selection step is cast as a semidefinite program, and per-example
`I(x; θ)` is a `dk × dk` matrix.

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

    def train(self):
        """Retrain the model from scratch on the current labeled set."""
        ...

    def query(self, n) -> np.ndarray:
        # Return n indices into the unlabeled pool to send to the oracle.
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
</content>
</invoke>
