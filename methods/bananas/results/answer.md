# BANANAS, distilled

BANANAS (Bayesian Optimization with Neural Architectures for Neural Architecture Search) is a
model-based NAS search: a Bayesian-optimization loop whose surrogate is an **ensemble of small
feedforward neural networks** trained on a novel **path encoding** of architectures, whose
**uncertainty** comes from ensemble disagreement, whose **acquisition function** is
**independent Thompson sampling (ITS)**, and whose **acquisition optimization** generates
candidates by **mutating the best-found architectures**. Each component addresses a concrete
failure mode in applying Bayesian optimization to discrete neural architectures.

## Problem it solves

Find `a* = argmin_{a in A} f(a)`, the architecture with the lowest validation error, where `A`
is a discrete space of labeled-DAG cells and `f(a)` (validation error after training) is an
expensive, gradient-free black box. The objective is to spend as few evaluations of `f` as
possible — sample efficiency under a strict query budget.

## Key ideas

1. **Neural surrogate instead of a GP.** A GP needs a kernel on DAGs (which does not exist
   off-the-shelf, forcing a hand-built distance like NASBOT's OTMANN) and costs `O(N^3)`. A
   neural predictor consumes a feature vector, learns its own similarity, and scales linearly —
   so no distance function and no cubic cost.

2. **Path encoding.** Represent an architecture by the set of input→output operation-paths it
   contains: one binary feature per possible path (e.g. `input→conv_1x1→pool_3x3→output`), set
   to 1 if present. Versus the adjacency matrix, path features are near-independent (each is a
   complete computational route, not a fragment whose meaning depends on others) and
   node-order-free (one architecture → one encoding, no isomorphism ambiguity). It is not
   one-to-one, so it deliberately treats architectures with the same operation-path set as
   functionally close.

3. **Truncation makes it scale.** The full path encoding has `sum_{i=0}^n r^i >= r^n` features
   (exponential in the number of nodes `n`), worse than the adjacency matrix's `O(n^2)`. But
   under sparse random sampling the long paths are exponentially rare, so truncating to the
   short/frequent paths keeps a **linear** number of features with provably vanishing
   probability of missing an observed path (Theorem below). In the 4-node, 5-op code path, the
   nonempty path vector has `5 + 25 + 125 = 155` coordinates and the short-path cutoff is `30`.

4. **Ensemble uncertainty (Deep Ensembles), not a Bayesian NN.** Train `M = 5` copies of the
   predictor from different random initializations/data orders. For any architecture,
   `fhat = (1/M) sum_m f_m`, `sigmahat = sqrt( sum_m (f_m - fhat)^2 / (M-1) )`. Simple,
   parallelizable, well-calibrated, and as good as a BNN at a fraction of the cost — important
   because the surrogate is retrained every BO iteration (and cheap relative to evaluating one
   architecture).

5. **Independent Thompson sampling acquisition.** From `(fhat, sigmahat)` per candidate, draw an
   *independent* sample `ftilde_a ~ N(fhat(a), sigmahat(a)^2)` for each architecture and select
   the minimizer. It is stochastic (so it slots directly into batch/parallel BO — take the `k`
   best draws) and decorrelated across candidates (unlike plain TS, which uses one shared
   ensemble member for all).

6. **Mutation-based acquisition optimization.** The space is too large to score exhaustively, so
   build a pool of ~100 candidates by **mutating the best-so-far** architectures with single
   edits. These stay edit-close to evaluated architectures (where the predictor is reliable) and
   cluster near the current optimum.

7. **MAPE loss.** Train the predictor with mean absolute percentage error against a lower-error
   floor `y_LB`, so low-error (high-accuracy) architectures — the regime the search cares
   about — get higher weight than a plain MAE would give.

## Acquisition functions (minimizing validation error; `y_min` = best error so far)

With a normal approximation `N(fhat, sigmahat^2)` to the predictive density and
`gamma = (y_min - fhat)/sigmahat`:

```
phi_EI(a)  = E[ max(0, y_min - f(a)) ] = integral_{-inf}^{y_min} (y_min - y) N(fhat,sigmahat^2) dy
           = sigmahat * (gamma * Phi(gamma) + phi(gamma))
phi_PI(a)  = P(f(a) < y_min)           = integral_{-inf}^{y_min}             N(fhat,sigmahat^2) dy = Phi(gamma)
phi_UCB(a) = fhat - beta * sigmahat                              (beta = 0.5)
phi_TS(a)  = f_mtilde(a),       mtilde ~ Unif(1, M)             (one ensemble member, all candidates)
phi_ITS(a) = ftilde_a,          ftilde_a ~ N(fhat(a), sigmahat(a)^2)   (independent per candidate)
```

Over the candidate pool, select the architecture that maximizes improvement for EI/PI
(`argmax phi_EI`, `argmax phi_PI`) and minimizes the predicted error for UCB/TS/ITS
(`argmin phi_UCB`, `argmin phi_TS`, `argmin phi_ITS`). The canonical implementation writes all
of them in a single sort by negating the improvement-style scores so one `argmin` selects in
every case.

## Algorithm

```
Input: search space A, dataset D, parameters t0 (warm start), T (budget),
       M (ensemble size), c (candidate-pool size), x (#archs to mutate),
       acquisition phi, expensive evaluator f(a) (= val error after training).

1. Draw t0 architectures uniformly at random from A; evaluate f on each.
2. For t = t0 to T:
   i.   Train an ensemble of M feedforward predictors on {(a_j, f(a_j))},
        using the (truncated) path encoding and the MAPE loss.
   ii.  Generate c candidates by mutating the x architectures a with the
        lowest f(a) found so far (single-edit mutations).
   iii. For each candidate a, compute the acquisition phi(a) from the
        ensemble mean fhat(a) and std sigmahat(a).
   iv.  Evaluate f on the selected candidates: largest raw EI/PI, or smallest
        UCB/TS/ITS. In code, negate EI/PI so one ascending sort handles all
        acquisition types. Use k=1 in a strict serial budget, k>1 for parallel BO.
Output: a* = argmin_{t} f(a_t).
```

## Path-encoding scaling theorem

Let `G_{n,k,r}` be a random architecture: `n` nodes each labeled with one of `r` operations;
each forward edge `(i,j)` (for `i<j`) present independently with probability `2k/(n(n-1))` (so
the expected edge count is `k`); rejected and resampled if there is no path from node 1 to node
`n`. Let `P` be the set of all possible paths and "path" mean a 1→`n` path.

**Theorem.** For integers `r, c > 0` there exists `N` such that for all `n > N`, with
`k = n + c`, there is a set `P'` of `n` paths with
`P( G_{n,n+c,r} contains some p in P \ P' ) <= 1/n^2`.

So the encoding can be truncated to a linear `O(n)` number of features (the short/frequent
paths) with vanishing probability of omitting a path that appears in the sampled architecture.

*Proof.* Let `L = floor(log_r n)` and choose `P'` as the `n` shortest paths, so it contains all
paths of length at most `L - 1`.

- **Few short paths.** The number of paths of length at most `L - 1` is
  `sum_{ell=0}^{L-1} r^ell = (r^L - 1)/(r - 1) <= (n - 1)/(r - 1) < n`,
  so fewer than `n` possible operation-label paths have length below `L`.
- **Expected length-`ell` path count.** A length-`ell` path chooses `ell - 1` intermediate
  nodes from `n - 2` and needs all `ell` edges present:
  `a_{n,k,ell} = C(n-2, ell-1) (2k/(n(n-1)))^ell`. At `ell = 1`,
  `a_{n,k,1} = 2k/(n(n-1)) >= 1/n` for large `n`.
- **Upper bound.** Using `C(m, j) <= (em/j)^j` and `k/(n-1) <= 2` (since `k = n + c`),
  `a_{n,k,ell} <= (4/n) (4e/(ell-1))^{ell-1}`.
- **Tail collapse.** For `ell >= L`, `(4e/(ell-1)) <= 4e/(L-1) < 1/2` (large `n`, up to harmless
  constant shifts), so the tail sums geometrically:
  `sum_{ell = L}^{n-1} a_{n,k,ell} <= 2 (4e/(L-1))^{L-1}`.
  The numerator `(4e)^L` is only polynomial in `n`, while
  `(L-1)^{L-1}` beats any fixed polynomial; the identity
  `(log n)^{log n} = n^{loglog n}` gives the same fact. Therefore
  `sum_{ell = L}^{n-1} a_{n,k,ell} < 1/n^3` for large enough `n`.
  By Markov, the probability a long path exists (no-rejection model `G'`) is `< 1/n^3`.
- **Condition on acceptance.** The chance `G'` has *any* 1→`n` path is `>= a_{n,k,1} >= 1/n`, so
  `P(exists long path in G_{n,k,r}) = P(long path in G') / P(any path in G') <= (1/n^3)/(1/n) = 1/n^2`.
  ∎

(Caveats: stated for freshly sampled, not mutated, architectures; and most-frequent paths are
not necessarily the most-informative paths.)

## Working code

A faithful instantiation of the BANANAS loop: truncated path encoding -> ensemble of feedforward
MAPE predictors -> ITS acquisition -> mutation-based candidate generation. The structure matches
the canonical implementation: `MetaNeuralnet` predictor, `acq_fn` with `explore_type='its'`, and
mutation-based `get_candidates`.

```python
import numpy as np
from tensorflow import keras
import tensorflow as tf

NUM_OPS = 5
LONGEST_PATH = 3


# ---- truncated path encoding ----
def encode_paths(arch):
    """One binary feature per input->output operation-path; full length
    NUM_OPS + NUM_OPS**2 + NUM_OPS**3."""
    o = arch
    L = sum(NUM_OPS ** i for i in range(1, LONGEST_PATH + 1))
    v = np.zeros(L, dtype=np.float32)
    v[o[3]] = 1.0                                                 # length-1 path
    off = NUM_OPS
    v[off + o[0] * NUM_OPS + o[4]] = 1.0                          # length-2 path
    v[off + o[1] * NUM_OPS + o[5]] = 1.0                          # length-2 path
    off = NUM_OPS + NUM_OPS ** 2
    v[off + o[0] * NUM_OPS ** 2 + o[2] * NUM_OPS + o[5]] = 1.0    # length-3 path
    return v


def path_encoding(arch, cutoff=30):
    """Keep the cutoff shortest/most-frequent paths."""
    full = encode_paths(arch)
    return full[:cutoff] if cutoff else full


# ---- feedforward predictor with MAPE loss ----
def mape_loss(y_true, y_pred):
    y_lb = 4.5                                  # lower bound on best val error
    return tf.abs((y_pred - y_lb) / (y_true - y_lb) - 1.0)


class Predictor:
    def fit(self, X, y, num_layers=10, width=20, epochs=150, lr=0.01):
        net = keras.models.Sequential(
            [keras.layers.Dense(width, activation='relu') for _ in range(num_layers)]
            + [keras.layers.Dense(1)])
        net.compile(optimizer=keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.99),
                    loss=mape_loss)
        net.fit(X, y, batch_size=32, epochs=epochs, verbose=0)
        self.net = net
        return self

    def predict(self, X):
        return np.squeeze(self.net.predict(X))


# ---- independent Thompson sampling ----
def acq_its(ensemble_preds):
    """ensemble_preds: (M, num_candidates). One independent N(fhat, sigmahat^2)
    draw per candidate; lower (predicted error) is better."""
    preds = np.array(ensemble_preds)
    fhat = preds.mean(axis=0)
    sigmahat = preds.std(axis=0, ddof=1)
    return np.random.normal(fhat, sigmahat)


class NASOptimizer:
    """BANANAS predictor-guided NAS."""

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed
        self.warm_start = min(10, num_epochs)
        self.ensemble_size = 5
        self.num_candidates = 100
        self.num_arches_to_mutate = 1
        self.patience_factor = 5
        self.seen = {}
        self.best_arch, self.best_val_loss = None, np.inf

    def _record(self, arch, val_loss):
        self.seen[tuple(arch)] = val_loss
        if val_loss < self.best_val_loss:
            self.best_val_loss, self.best_arch = val_loss, list(arch)

    def _fit_ensemble(self):
        X = np.stack([path_encoding(list(a)) for a in self.seen])
        y = np.array([self.seen[a] for a in self.seen], dtype=np.float32)
        return [Predictor().fit(X, y) for _ in range(self.ensemble_size)]

    def _propose_next(self):
        ensemble = self._fit_ensemble()
        best = sorted(self.seen, key=lambda a: self.seen[a])[
            : self.num_arches_to_mutate * self.patience_factor
        ]
        cands = []
        while len(cands) < self.num_candidates:
            parent = list(best[np.random.randint(len(best))])
            child = mutate_architecture(parent)               # single-edit mutation
            if tuple(child) not in self.seen:
                cands.append(child)
        Xc = np.stack([path_encoding(a) for a in cands])
        preds = [p.predict(Xc) for p in ensemble]
        return cands[int(np.argmin(acq_its(preds)))]          # min predicted error

    def search_step(self, epoch):
        if epoch < self.warm_start or len(self.seen) < 2:
            arch = random_architecture()
            while tuple(arch) in self.seen:
                arch = random_architecture()
        else:
            arch = self._propose_next()
        val_loss = self.api.query_val_loss(arch)
        self._record(arch, val_loss)
        return {"best_val_loss": self.best_val_loss, "queries": self.api.query_count}

    def get_best_architecture(self):
        return self.best_arch
```
