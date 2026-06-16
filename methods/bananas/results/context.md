# Context: model-based neural architecture search under a query budget

## Research question

We have a search space `A` of neural architectures — each architecture `a` is a labeled
directed acyclic graph (a "cell": a handful of nodes/edges, each labeled with an operation
such as `3x3 conv`, `1x1 conv`, `3x3 avg-pool`, `skip`, or `none`), and the whole network is
built by stacking copies of the cell. We want the architecture with the lowest validation
error, `a* = argmin_{a in A} f(a)`, where `f(a)` is the validation error after training `a`
on a fixed dataset for a fixed number of epochs. The defining difficulty is that **`f` is a
black box and brutally expensive**: a single evaluation means training a neural network,
hours of GPU time, with no gradient of `f` with respect to the discrete architecture. The
search space is large (tens of thousands to `10^18` architectures) and discrete — a set of
DAGs, not a region of `R^d`. So the entire game is: spend as few evaluations of `f` as
possible to find a near-optimal `a`. With a strict budget of `K` evaluations (in the
sample-efficient regime, `K` on the order of tens), brute-force enumeration is impossible and
the *algorithm* — how each query is chosen from the results of the previous ones — is what
separates a good final architecture from a mediocre one. The problem is to design that
query-selection strategy: a search that models what it has learned about `{f(a)}_{a in A}` and
uses the model to pick the most informative architecture to evaluate next.

## Background

Since the 2017 surge of interest in NAS, several paradigms compete: evolutionary search,
reinforcement learning, gradient-based ("one-shot") relaxations, and **Bayesian optimization
(BO)**. BO is the natural framework for global optimization of a noisy, expensive black-box
function where each query is precious — exactly the NAS setting — and it has a long track
record in deep-learning hyperparameter tuning. Its recipe is well established. Maintain a
probabilistic *surrogate model* of `f` from the architectures evaluated so far; at each step,
choose the next architecture by maximizing an **acquisition function** that scores how
promising each candidate is, trading off exploitation (low predicted error) against
exploration (high predictive uncertainty); evaluate `f` there; update the model; repeat. The
acquisition function is the heart of the explore/exploit balance, and the standard choices —
expected improvement (EI), probability of improvement (PI), upper-confidence bound (UCB), and
Thompson sampling (TS) — all consume two numbers per candidate: a predicted value `fhat` and a
predictive uncertainty `sigmahat`.

The historical workhorse surrogate is the **Gaussian process (GP)**. A GP is defined by a
kernel `kappa(x, x')` measuring similarity between inputs, and it returns a posterior mean and
variance in closed form. This works beautifully when the input domain is Euclidean or
categorical, because off-the-shelf kernels (Gaussian, Matérn, Hamming) already exist there.
Two pain points dominate when BO meets NAS. First, **the inputs are DAGs, not vectors**, so
there is no off-the-shelf kernel: applying GP-BO requires *inventing* a similarity/distance
function between architectures, which is itself a hard, hand-tuned modeling problem. Second,
**GP inference scales cubically** in the number of observations `N` — it inverts a dense
`N x N` covariance matrix — so the surrogate becomes a bottleneck as evaluations accumulate
and parallelism is hard to exploit. These two facts had limited GP-BO's ability to reach
state-of-the-art on NAS.

There are several knobs each NAS algorithm must set, and the field had, by this point, a few
diagnostic observations about them. (1) **Encoding.** A neural predictor or a kernel needs the
architecture turned into a feature vector. The default is the *adjacency-matrix encoding*: fix
an (arbitrary) ordering of the nodes, put a binary feature for each possible edge `i<j`, and
append a list of the operation at each node. This representation is awkward for a learner to
read: because the node ordering is arbitrary, a single architecture maps to *many* different
adjacency matrices (the isomorphism problem), and the features are strongly
inter-dependent — an edge from the input to node 2 carries no information unless there is also
a path from node 2 to the output, and if there is, that edge's usefulness is entangled with
the operation chosen at node 2. Highly correlated, order-dependent features are exactly what a
predictor struggles to fit. (2) **Uncertainty.** To run any acquisition function you need a
calibrated uncertainty per prediction. (3) **Acquisition optimization.** Even granting a
surrogate, you must *maximize the acquisition over `A`* each step — but `A` has up to `10^18`
elements, so you cannot score every architecture; you must construct a manageable candidate
set and score those. Each of these is a place where two reasonable choices give very
different end-to-end NAS performance, and prior systems tended to fix them implicitly and
report only the bundled result, so it was hard to tell which knob mattered.

## Baselines

These are the prior methods a new model-based NAS search would be measured against and react
to.

**GP-based Bayesian optimization (Mockus 1975 for EI; Snoek et al. 2012).** The canonical BO
surrogate. Place a GP prior on `f`, condition on the evaluated `(a, f(a))` pairs to get a
posterior mean/variance, and pick the next point by maximizing EI/UCB. *Gap in the NAS
setting:* a GP needs a kernel on the input domain, and there is no natural kernel on the space
of DAGs; and the cubic `O(N^3)` inference cost makes the surrogate expensive and hard to
parallelize as observations accumulate.

**NASBOT (Kandasamy et al., NeurIPS 2018).** Makes GP-BO usable for NAS by *hand-constructing*
a (pseudo-)distance between architectures — OTMANN, an optimal-transport distance over the
"masses" of each layer type — turning it into a kernel, then running GP-BO with an
evolutionary routine to optimize the acquisition. *Gap:* the distance function is cumbersome
to define, comes with its own hyperparameters to tune, is tailored to a particular
architecture space, and the underlying GP still pays the cubic cost and a per-iteration matrix
inversion.

**DNGO (Snoek et al., ICML 2015).** Replaces the GP surrogate with a *neural network*: train a
deep net on the data, then do Bayesian linear regression on its learned last-layer features
(adaptive basis regression). Inference becomes linear in `N` instead of cubic, enabling far
greater parallelism. The stated goal is efficiency — bringing GP-BO's cost from cubic to
linear — rather than improving sample efficiency per se. *Gap:* it was designed for Euclidean
hyperparameter spaces, so the architecture must still be turned into a feature vector somehow,
and its uncertainty comes only from the last-layer Bayesian linear regression.

**BOHAMIANN (Springenberg et al., NeurIPS 2016).** Uses a *Bayesian neural network* as the
surrogate, with scalable Hamiltonian Monte Carlo to sample the posterior over weights, giving
both predictions and uncertainties from one model. *Gap:* Bayesian NNs are comparatively hard
to implement and slow (they require long sampling chains), and the quality of their
uncertainty is sensitive to the prior and to the degree of posterior approximation.

**Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017).** A non-Bayesian route to predictive
uncertainty: train `M` copies of the same network from different random initializations (and
data orderings), and read off the mean and variance of their predictions. It is simple to
implement, trivially parallelizable, needs little tuning, and yields uncertainty estimates as
good as or better than Bayesian NNs — repeatedly, even at `M` as small as 3 or 5. *Gap for NAS
search:* it is an uncertainty-estimation primitive, not a discrete architecture-search rule by
itself.

**Regularized Evolution (Real et al., AAAI 2019).** A strong, simple NAS baseline that needs no
surrogate at all: keep a population of evaluated architectures; each step, sample a tournament,
mutate the best member by a single random edit (change one operation/edge), add the child,
and remove the *oldest* member (the age-based "regularization"). *Gap as a search strategy:*
it never builds a model of `f`, so it cannot extrapolate to unseen architectures or exploit
the structure it has already measured — it only does local random edits, which is
sample-inefficient under a tight query budget.

**Random search (Li & Talwalkar 2019).** Draw `n` architectures uniformly at random, return the
best by validation error. Embarrassingly simple, and a notoriously competitive baseline. *Gap:*
purely uninformed — it learns nothing from previous queries, so every architecture is a fresh
gamble and the budget is spent without compounding knowledge.

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
  budget). On NAS-Bench-201 the validation and test errors are *not* tightly correlated, so an
  algorithm can overfit the validation signal — the lowest-validation-error architecture is not
  always the lowest-test-error one. Average over many trials/seeds (variance is large at small
  budgets) and report mean ± std. A single final, unbudgeted test query scores the returned
  architecture.

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
