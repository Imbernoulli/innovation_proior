# Context: model-based neural architecture search under a query budget

## Research question

We have a search space `A` of neural architectures — each architecture `a` is a labeled
directed acyclic graph (a "cell": a handful of nodes/edges, each labeled with an operation
such as `3x3 conv`, `1x1 conv`, `3x3 avg-pool`, `skip`, or `none`), and the whole network is
built by stacking copies of the cell. We want the architecture with the lowest validation
error, `a* = argmin_{a in A} f(a)`, where `f(a)` is the validation error after training `a`
on a fixed dataset for a fixed number of epochs. `f` is a black box and expensive: a single
evaluation means training a neural network, hours of GPU time, with no gradient of `f` with
respect to the discrete architecture. The search space is large (tens of thousands to `10^18`
architectures) and discrete — a set of DAGs, not a region of `R^d`. So the game is: spend as
few evaluations of `f` as possible to find a near-optimal `a`. With a strict budget of `K`
evaluations (in the sample-efficient regime, `K` on the order of tens), brute-force
enumeration is impossible and the *algorithm* — how each query is chosen from the results of
the previous ones — is what produces the final architecture. The problem is to design that
query-selection strategy: a search that models what it has learned about `{f(a)}_{a in A}` and
uses the model to pick the most informative architecture to evaluate next.

## Background

Since the 2017 surge of interest in NAS, several paradigms compete: evolutionary search,
reinforcement learning, gradient-based ("one-shot") relaxations, and **Bayesian optimization
(BO)**. BO is a standard framework for global optimization of a noisy, expensive black-box
function where each query is precious, and it has a long track record in deep-learning
hyperparameter tuning. Its recipe is well established. Maintain a probabilistic *surrogate
model* of `f` from the architectures evaluated so far; at each step, choose the next
architecture by maximizing an **acquisition function** that scores how promising each candidate
is, trading off exploitation (low predicted error) against exploration (high predictive
uncertainty); evaluate `f` there; update the model; repeat. The acquisition function carries
the explore/exploit balance, and the standard choices — expected improvement (EI), probability
of improvement (PI), upper-confidence bound (UCB), and Thompson sampling (TS) — all consume two
numbers per candidate: a predicted value `fhat` and a predictive uncertainty `sigmahat`.

The classic surrogate is the **Gaussian process (GP)**. A GP is defined by a kernel
`kappa(x, x')` measuring similarity between inputs, and it returns a posterior mean and
variance in closed form. Off-the-shelf kernels (Gaussian, Matérn, Hamming) exist for Euclidean
or categorical input domains. GP inference scales cubically in the number of observations `N`,
since it inverts a dense `N x N` covariance matrix.

There are several knobs each NAS algorithm must set. (1) **Encoding.** A neural predictor or a
kernel needs the architecture turned into a feature vector. The default is the *adjacency-matrix
encoding*: fix an (arbitrary) ordering of the nodes, put a binary feature for each possible edge
`i<j`, and append a list of the operation at each node. Because the node ordering is arbitrary,
a single architecture maps to many different adjacency matrices (the isomorphism question), and
the features are inter-dependent — an edge from the input to node 2 carries information only in
combination with a path from node 2 to the output, and that edge's role is entangled with the
operation chosen at node 2. (2) **Uncertainty.** Any acquisition function needs a calibrated
uncertainty per prediction. (3) **Acquisition optimization.** Given a surrogate, one must
*maximize the acquisition over `A`* each step — but `A` has up to `10^18` elements, so one
cannot score every architecture; one constructs a manageable candidate set and scores those.
Each of these is a place where reasonable choices give different end-to-end NAS performance, and
prior systems tended to fix them implicitly and report the bundled result.

## Baselines

These are the prior methods a new model-based NAS search would be measured against and react
to.

**GP-based Bayesian optimization (Mockus 1975 for EI; Snoek et al. 2012).** The canonical BO
surrogate. Place a GP prior on `f`, condition on the evaluated `(a, f(a))` pairs to get a
posterior mean/variance, and pick the next point by maximizing EI/UCB. Inference costs
`O(N^3)`, and applying it requires a kernel defined on the input domain.

**NASBOT (Kandasamy et al., NeurIPS 2018).** Applies GP-BO to NAS by *hand-constructing* a
(pseudo-)distance between architectures — OTMANN, an optimal-transport distance over the
"masses" of each layer type — turning it into a kernel, then running GP-BO with an evolutionary
routine to optimize the acquisition. The distance has its own hyperparameters and is defined
for a particular architecture space.

**DNGO (Snoek et al., ICML 2015).** Replaces the GP surrogate with a *neural network*: train a
deep net on the data, then do Bayesian linear regression on its learned last-layer features
(adaptive basis regression). Inference becomes linear in `N` instead of cubic, enabling far
greater parallelism. It was developed for Euclidean hyperparameter spaces; the architecture is
turned into a feature vector, and its uncertainty comes from the last-layer Bayesian linear
regression.

**BOHAMIANN (Springenberg et al., NeurIPS 2016).** Uses a *Bayesian neural network* as the
surrogate, with scalable Hamiltonian Monte Carlo to sample the posterior over weights, giving
both predictions and uncertainties from one model.

**Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017).** A non-Bayesian route to predictive
uncertainty: train `M` copies of the same network from different random initializations (and
data orderings), and read off the mean and variance of their predictions. It is simple to
implement, trivially parallelizable, needs little tuning, and yields uncertainty estimates as
good as or better than Bayesian NNs — even at `M` as small as 3 or 5.

**Regularized Evolution (Real et al., AAAI 2019).** A simple NAS baseline that needs no
surrogate: keep a population of evaluated architectures; each step, sample a tournament, mutate
the best member by a single random edit (change one operation/edge), add the child, and remove
the *oldest* member (the age-based "regularization"). It does local random edits, building no
model of `f`.

**Random search (Li & Talwalkar 2019).** Draw `n` architectures uniformly at random, return the
best by validation error. Embarrassingly simple, and a notoriously competitive baseline. Every
architecture is drawn independently of previous queries.

## Evaluation settings

The natural yardsticks already in use, all designed for fair and reproducible comparison.

- **Tabular NAS benchmarks.** NAS-Bench-101 (Ying et al. 2019): a cell-based space of ~423,000
  architectures (7-node DAG, 3 ops, ≤9 edges), each with precomputed validation/test accuracy
  on CIFAR-10, so `f(a)` is a table lookup rather than a training run. NAS-Bench-201
  (Dong & Yang, ICLR 2020): a complete DAG on 4 nodes with 6 edges, each edge taking one of 5
  operations (`skip_connect, none, nor_conv_3x3, nor_conv_1x1, avg_pool_3x3`), giving
  `5^6 = 15,625` architectures, with precomputed accuracies on three datasets — CIFAR-10,
  CIFAR-100, and ImageNet16-120. An architecture is a list of 6 operation indices in `[0,4]`.
- **The DARTS search space** (Liu et al. 2019): two cells (normal + reduction), four nodes each
  with two incoming edges from eight operations, ~`10^18` architectures — not tabular, so
  evaluation means actually training (e.g. 50 epochs, record the last few epochs' mean
  validation error), and final comparison trains the best-found cell with a standard pipeline.
- **Metric and protocol.** Compare algorithms by the test error of the architecture with the
  best *validation* error found so far, as a function of the number of queries to `f` (the
  budget). On NAS-Bench-201 the validation and test errors are *not* tightly correlated, so the
  lowest-validation-error architecture is not always the lowest-test-error one. Average over
  many trials/seeds (variance is large at small budgets) and report mean ± std. A single final,
  unbudgeted test query scores the returned architecture.

## Code framework

The search plugs into a fixed harness around a budgeted black-box benchmark. The benchmark API
exposes one budgeted call, `query_val_loss(arch)`, which counts against the `K`-query budget,
plus helpers that already exist before any search strategy is designed: sampling a random valid
architecture, applying a small edit to an architecture, deduplicating architectures, and
recording validation and test losses. The harness calls `search_step(epoch)` up to `K` times
and then tests whatever `get_best_architecture()` returns. What is not settled is the policy
inside the search loop: how the evaluated data should be used to choose the next architecture.

```python
import numpy as np


# --- pre-existing search-space helpers (fixed) ---
def random_architecture():
    """Sample a random valid architecture: a list of 6 op indices in [0, 4]."""
    ...

def mutate_architecture(parent):
    """Return a copy of parent with one random edge changed to a different op."""
    ...

def architecture_id(arch):
    """Stable identifier used to avoid duplicate evaluations."""
    ...


class SearchPolicy:
    """The still-undesigned model-based rule for choosing the next query."""

    def propose(self, history):
        # TODO: design how evaluated architectures determine the next query.
        pass


class NASOptimizer:
    """Sample-efficient search under a strict K-query budget."""

    def __init__(self, api, num_epochs, seed):
        self.api = api                 # api.query_val_loss(arch) costs 1 query
        self.num_epochs = num_epochs   # K, the query budget
        self.seed = seed
        self.policy = SearchPolicy()
        self.seen = {}                 # arch_id -> (arch, measured val_loss)
        self.best_arch = None
        self.best_val_loss = np.inf

    def _record(self, arch, val_loss):
        self.seen[architecture_id(arch)] = (arch, val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss, self.best_arch = val_loss, list(arch)

    def search_step(self, epoch):
        if not self.seen:
            arch = random_architecture()
        else:
            arch = self.policy.propose(list(self.seen.values()))
        val_loss = self.api.query_val_loss(arch)
        self._record(arch, val_loss)
        return {"best_val_loss": self.best_val_loss, "queries": self.api.query_count}

    def get_best_architecture(self):
        return self.best_arch
```

The single conceptual slot is the body of `SearchPolicy.propose`: the rest is the budgeted
query loop that already exists.
