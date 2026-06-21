## Research question

A learner can choose which inputs `x` to have labeled, paying for each label, and wants the
best model from the fewest labels. One inductive goal — agnostic to any particular downstream
loss or test distribution — is to query the `x` whose label most reduces the learner's
uncertainty about the unknown parameters `theta` of the model `p(y|x,theta)`. With a Bayesian
model and a posterior `p(theta|D)` over those parameters, "uncertainty" is Shannon entropy
`H[theta|D]`, and the greedy (myopic) criterion is the input whose label is expected to shrink
that entropy the most,

```
argmax_x  H[theta | D]  -  E_{y ~ p(y|x,D)} [ H[theta | y, x, D] ].          (1)
```

The full set-selection problem is NP-hard, but the myopic version is known to be near-optimal
for this kind of sequential decision problem, so (1) is the object of interest.

Criterion (1) lives in *parameter* space:

- Computing `H[theta|D]` involves the parameter posterior. For models with high-dimensional
  parameters this is an entropy of a high-dimensional distribution; for a *nonparametric* model
  — a Gaussian process, where the "parameter" is a whole latent function `f` — the parameter
  space is infinite-dimensional. Gridding the parameter space is exponential in dimension, and
  entropies can be estimated by sampling from the posterior.
- Evaluating (1) for one candidate `x` requires `E_y[H[theta|y,x,D]]`: for each possible label
  value `y` the learner imagines adding `(x,y)` to the data and recomputes the posterior. With
  `N_x` candidate inputs and `N_y` possible label values that is `O(N_x N_y)` posterior updates.

The setting, then: evaluate the information gain (1) for every pool candidate, including for
models with intractable or infinite-dimensional parameter posteriors.

## Background

By 2010 the field had two broad routes to active learning. The *decision-theoretic* route
(Roy & McCallum 2001; Kapoor et al. 2007; Zhu et al. 2003) minimizes expected future loss —
the Bayes posterior risk — directly. It requires knowing the loss function and the test
distribution in advance, it is often transductive (it uses the test inputs), and it typically
costs `O(N_x N_y)` posterior updates. The *information-theoretic* route scores queries purely
by how much they shrink uncertainty about the model, agnostic to the eventual decision task.
This is the route (1) takes, and its intellectual roots are old.

**Shannon information as an experiment-design criterion.** Lindley (1956) scored an experiment
in a Bayesian frame by the expected reduction in entropy of the unknown parameter from prior to
posterior, `E_y[ H[p(theta)] - H[p(theta|y)] ]`; this *expected information gain* is exactly the
quantity inside (1). Bernardo (1979) recast it as expected utility. MacKay (1992) brought the
criterion to neural-network learning, discussing several information-based objectives for
choosing which data to acquire; he evaluates it under a Gaussian (Laplace) approximation to the
posterior so the parameter-space entropies become computable.

**Information theory toolbox.** The relevant primitives (Cover & Thomas) are Shannon entropy
`H`, conditional entropy, expected information gain, KL divergence, and the algebraic identities
that relate them. For a binary output the entropy is the Bernoulli/binary entropy
`h(p) = -p log p - (1-p) log(1-p)` (measured in bits when the log is base 2; `h(1/2) = 1`).

**Gaussian processes for classification (Rasmussen & Williams 2005).** A flexible nonparametric
model: place a GP prior on a latent function, `f ~ GP(mu, k)`, and for binary classification use
a probit likelihood `y | x, f ~ Bernoulli(Phi(f(x)))` with `Phi` the standard-normal CDF. The
posterior over `f` is non-Gaussian and intractable; it is approximated by a Gaussian via the
Laplace approximation, Expectation Propagation (EP), Assumed Density Filtering (ADF), or sparse
methods. Under any such approximation the latent at a point is Gaussian,
`f_x ~ N(mu_{x,D}, sigma^2_{x,D})`. A diagnostic comparison (Kuss & Rasmussen 2005) found EP
substantially more accurate than the Laplace approximation for GPC, at higher cost. This is the
model with infinite-dimensional `theta` and an analytically intractable posterior.

**Behavior of predictive-uncertainty criteria.** It is documented that a criterion which scores
a query by raw predictive uncertainty alone tends, in classification, to repeatedly pick points
on or near the decision boundary or in regions of high observation noise (e.g. Huang et al.
2010).

## Baselines

**Maximum Entropy Sampling (Sebastiani & Wynn 2000).** Work directly in data space: query the
`x` of maximum predictive entropy `H[y|x,D]`. For a regression model with input-*independent*
observation noise this is provably the optimal design, because every candidate carries the same
irreducible noise contribution.

**Query by Committee (Seung, Opper & Sompolinsky 1992).** Draw several hypotheses consistent
with the data — committee members sampled from the version space — and let them vote on the
label of each candidate `x`; query the `x` with the most balanced vote, the "principle of
maximal disagreement." For the toy models studied this yields asymptotically finite information
per query and generalization error decaying exponentially in the number of labels. The vote is a
deterministic tally over the committee members' predictions.

**The Informative Vector Machine (Lawrence, Seeger & Herbrich 2003).** A GP-specific
information-theoretic method for choosing which points to include when training a GP. It uses the
parameter-space objective (1), approximating the parameter entropy in the marginal subspace
spanned by the observed points, and computing the entropy decrease after including a point
efficiently from the GP covariance matrix. It is designed for subsampling an already-labeled set,
computing the entropy change between the current approximate posterior `q_t` and the
re-approximated posterior `q_{t+1}` after inclusion; it uses ADF for `q_{t+1}`.

**Direct use of the parameter-entropy objective (MacKay 1992; Krishnapuram et al. 2004;
Lawrence et al. 2003).** Several methods plug (1) in literally, computing the entropy of a
Gaussian (or otherwise low-dimensional) approximation to the parameter posterior, at a cost of
`O(N_x N_y)` updates.

## Evaluation settings

The pool-based active-learning yardstick of the time. The learner repeatedly selects inputs from
a fixed pool of unlabeled points, has them labeled, retrains, and is scored on a held-out test
set; pool selection (rather than synthesizing arbitrary `x`) is used so that methods which cannot
generate continuous queries can be compared on the same footing.

- **Synthetic 2-D datasets** chosen to stress specific behaviors: a cluster of noisy points
  sitting on the decision boundary; a cluster of uninformative points far from the boundary; a
  checkerboard of disjoint single-class islands.
- **UCI classification datasets** — *crabs, vehicle, wine, wdbc, isolet, australia, cancer*,
  and *letter* (a multiclass set, used by pulling out hard-to-separate pairs such as E vs. F
  and D vs. P).
- **Preference-learning datasets** built from regression sets (*cpu, cart, kinematics*) by
  forming item pairs with a binary "preferred" label.
- **Metrics:** test classification accuracy as a function of the number of labels acquired —
  the learning curve — and how many labels a method needs to reach a target fraction of the
  full-pool accuracy. Extensive Monte Carlo estimation of the true information gain serves as a
  gold standard against which approximate scores are compared.

## Code framework

Pool-based active learning runs inside a fixed harness: a `Strategy` base class owns the pool
features `X`, the labels `Y`, a boolean `idxs_lb` marking which pool points are already labeled,
and the current classifier; the harness handles retraining the model after each acquisition
round. The one thing the harness does *not* provide is the rule for which unlabeled points to
query next — that is the single empty slot. The base class also exposes the predictive primitives
a query rule might use, including the deterministic softmax probabilities and a
stochastic-forward-pass primitive that returns several softmax outputs per input.

```python
import numpy as np
import torch


class Strategy:
    """Pool-based active-learning harness. Owns the pool and the classifier; the
    surrounding loop retrains the model after each round of acquisitions."""

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X                 # pool features        [n_pool, n_features]
        self.Y = Y                 # pool labels (LongTensor)
        self.idxs_lb = idxs_lb     # boolean mask: which pool points are labeled
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)

    # --- predictive primitives already available to a query rule ---

    def predict_prob(self, X, Y):
        """Deterministic softmax probabilities, shape [len(X), n_classes]."""
        ...

    def predict_prob_dropout_split(self, X, Y, n_drop):
        """n_drop stochastic forward passes, shape [n_drop, len(X), n_classes]
        -- an ensemble of predictive distributions per input."""
        ...

    # --- the empty slot: the acquisition rule ---

    def query(self, n):
        # Return n indices into self.X of currently-unlabeled points to label.
        # TODO: fill in the query rule for the unlabeled pool.
        pass


# fixed outer loop the strategy plugs into
def active_learning_loop(strategy, n_rounds, n_per_round):
    for _ in range(n_rounds):
        idxs = strategy.query(n_per_round)
        label_oracle(idxs)                        # acquire labels for those points
        strategy.update(mark_labeled(idxs))       # move them into the labeled set
        strategy.train()                          # retrain on the expanded labeled set
```

Everything except `query` is fixed by the harness. The empty slot returns pool indices for the
next labels.
