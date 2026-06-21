# Context: learning a directed causal graph from observational nonlinear data (circa 2018-2019)

## Research question

We observe a random vector `X = (X_1, ..., X_d)` and we want the *directed acyclic graph* `G`
that encodes which variable is a direct cause of which — not a correlation graph, not an
undirected skeleton, but oriented edges `i -> j` carrying a causal reading. The data is purely
*observational*: no interventions, no experiments, just samples from the joint distribution
`P_X`.

Two features of the setting shape any approach. The space of DAGs on `d` nodes is
*super-exponential* in `d` (faster than `d!`, since the orderings themselves are only part of the
count), and the defining property — acyclicity — is a global, discrete constraint on the edge
set: an algorithm that proposes an edge must, in principle, re-check that no cycle was created.
Separately, from `P_X` alone the causal direction is generally *not* recoverable: many distinct
DAGs induce exactly the same set of conditional independences (the same *Markov equivalence
class*), so observational data pins down that class — a partially directed graph (CPDAG) — unless
one adds parametric assumptions on the data-generating mechanism that break the symmetry between
cause and effect.

The question we take up: given observational samples, under a stated assumption that makes `G`
identifiable, recover the *fully directed* graph for *nonlinear* mechanisms at graph sizes (tens
of nodes) of practical interest.

## Background

**Causal graphical models and the Markov factorization.** A causal graphical model is a pair
`(P_X, G)` with `G` a DAG; `P_X` is Markov to `G`, so its density factorizes over the graph,
`p(x) = prod_{j=1}^d p_j(x_j | x_{pi_j})`, where `pi_j` is the parent set of node `j`. The edges
carry causal meaning, so the model answers interventional queries (what happens to `X_j` if we
externally set `X_i`). Structure learning is recovering `G` from a sample `{x^(1), ..., x^(n)}`.
There are two broad families: *constraint/independence-based* methods (PC, run conditional-
independence tests and orient by collider rules; Spirtes et al. 2000) and *score-based* methods
(define a score `S(G)` — typically a regularized maximum likelihood — and search for
`argmax_{G in DAG} S(G)`; Koller & Friedman 2009).

**Additive noise models make nonlinear mechanisms identifiable.** The identifiability result the
field rests on for continuous nonlinear data is the *additive noise model* (ANM): assume
`X_j := f_j(X_{pi_j}) + N_j` with the `N_j` mutually independent and the `f_j` nonlinear,
satisfying mild regularity conditions (Hoyer, Janzing, Mooij, Peters & Scholkopf 2008; Peters,
Mooij, Janzing & Scholkopf 2014). Under these assumptions `G` is identifiable from `P_X` — even
when the noise is Gaussian, *because the nonlinearity itself breaks the cause/effect symmetry*
(the special restricted case where a linear-Gaussian model would be unidentifiable but a
nonlinear-Gaussian one is not). Intuitively, regressing effect on cause leaves a residual
independent of the cause, while regressing cause on effect does not. This is what lets a method
aim at the *directed* graph rather than only its equivalence class. A practical estimator built
directly on this is RESIT (regress each variable on candidate parents, test the residuals for
independence). A property of maximum-likelihood scoring relevant to any score-based method: the
unpenalized log-likelihood *never decreases when an edge is added*, so such methods include an
explicit sparsity control.

**The combinatorial constraint and the spectral view of cycles.** A standard fact about a
nonnegative (or binary) adjacency matrix `B`: `(B^k)_{jj}` counts the closed walks of length `k`
through node `j`, so `tr(B^k)` counts all length-`k` cycles, and `B` is acyclic *iff* `tr(B^k)=0`
for every `k = 1, 2, ...`. This characterizes acyclicity, taken as a finite series
`sum_{k=1}^d tr(B^k)=0`.

## Baselines

These are the prior structure-learning methods a new approach would be measured against.

**GES / greedy CPDAG search (Chickering 2003).** Greedily searches the space of CPDAGs to optimize
the Bayesian information criterion, usually under a linear-Gaussian model, and returns a CPDAG.

**GSF (Huang et al. 2018).** The GES search engine with a generalized (kernel-based) score that
admits nonlinear relationships, again over CPDAGs.

**RESIT (Peters et al. 2014).** Built directly on ANM identifiability: greedily regress and test
residual independence to find an ordering, then prune.

**CAM — Causal Additive Models (Buhlmann, Peters & Ernest 2014).** Score-based, assuming an
*additive* mechanism `f_j(x_{pi_j}) = sum_{i in pi_j} f_{ij}(x_i)` plus Gaussian noise. It
*decouples* the search for a topological order of the variables (greedy restricted maximum
likelihood) from per-node feature/edge selection (sparse additive regression). For larger graphs
it first runs a *preliminary neighborhood selection* (PNS) — fit a flexible regressor of each
variable on all others and keep only high-importance candidates — and after fitting it *prunes*
each node's parents by a significance test, dropping parents whose covariate test exceeds a small
p-value.

**NOTEARS (Zheng, Aragam, Ravikumar & Xing 2018).** Casts structure learning as a *continuous
constrained optimization* by encoding the graph as a single real weighted matrix and replacing the
combinatorial acyclicity constraint with one smooth equality. It builds on the spectral fact above
using the matrix exponential, which reweights the length-`k` closed-walk counts by `1/k!`: for a
*binary* `B`, `B` is a DAG iff `tr e^B = d`. To handle real, possibly negative weights it replaces
`B` by the (entrywise nonnegative) Hadamard square `W ∘ W`, giving the smooth constraint

```
h(W) = tr e^{W ∘ W} - d = 0,     with gradient   ∇h(W) = (e^{W ∘ W})^T ∘ 2W.
```

The graph is read off `W`: the matrix `W` *is* the coefficient matrix of a **linear** structural
model `X_j = u_j^T X + N_j`, so `W_{ij} != 0` exactly means edge `i -> j`, and `|W_{ij}|` is the
edge's weight feeding the constraint. With a least-squares (Gaussian) score and an `L1` sparsity
penalty, the program

```
max_W  -(1/(2n)) || X - X W ||_F^2  -  λ || W ||_1     s.t.   tr e^{W ∘ W} - d = 0
```

is solved by an off-the-shelf augmented-Lagrangian method, and every edge is updated at each step
from the gradient.

**DAG-GNN (Yu et al. 2019).** Pushes the continuous-constraint paradigm to nonlinear mechanisms via
a graph-neural-network decoder trained on an evidence lower bound; the per-variable mechanisms are
tied together through parameter sharing in the GNN.

**MADE (Germain et al. 2015).** Not a causal method but a building block: a masked autoencoder that
multiplies each weight matrix by a fixed binary mask so the network respects the *autoregressive
property* — output `j` depends only on inputs before `j` in a chosen ordering. The autoregressive
property is exactly acyclicity for a *fixed* ordering. MADE fixes the masking (hence the ordering)
a priori.

## Evaluation settings

The natural yardsticks already in use for observational structure learning:

- **Synthetic ANM data.** Sample a ground-truth DAG from a random-graph scheme — *Erdos-Renyi*
  (ER, edges added independently, Poisson degree) or *scale-free* (SF, Barabasi-Albert preferential
  attachment, power-law degree with a few high-degree hubs) — at a chosen expected edge count
  (e.g. `d` or `4d` edges). Then generate data per `X_j := f_j(X_{pi_j}) + N_j` with the `f_j`
  drawn from a Gaussian process (unit-bandwidth RBF kernel) and Gaussian noise — the identifiable
  nonlinear-Gaussian ANM regime. Graph sizes of order 10-100 nodes, sample sizes of order
  hundreds to a few thousand. Variant generators stress-test misspecification: linear-Gaussian
  (favors a linear method), additive nonlinear `sum_i f_{ij}(x_i)` (favors an additive method),
  and post-nonlinear `g_j(f_j(x_{pi_j}) + N_j)` with non-Gaussian noise (violates the additive
  assumption). Noise families beyond Gaussian (exponential, Laplace) are also used.
- **Real / pseudo-real data.** A protein-signaling flow-cytometry dataset (Sachs et al. 2005, 11
  nodes, ~853 observational samples) with an accepted ground-truth network; and pseudo-real gene-
  expression data from the SynTReN simulator of transcriptional regulatory networks (~20 nodes).
- **Metrics on the recovered graph.** *Structural Hamming distance* (SHD): number of missing,
  extra, or reversed edges. *Structural interventional distance* (SID, Peters & Buhlmann 2015):
  number of intervention pairs `(i,j)` whose interventional distribution would be mis-estimated
  using the predicted graph's parent-adjustment set — directly tied to causal-inference quality.
  For methods that return only a CPDAG, SID is reported as a lower/upper bound over the
  equivalence class, and SHD-C compares the CPDAGs. F1, precision, and recall on the directed edge
  set are also natural.
- **Protocol.** Compare against random-graph baselines; hold out part of the data for early
  stopping and for selecting hyperparameters by held-out score (no ground truth available in
  practice). Identifiability guarantees hold in the population/exact-optimization limit, so
  finite-sample, non-convex behavior is checked empirically.

## Code framework

The method plugs into a standard score-based-learning harness: load observational data `X`, fit a
differentiable model of the conditionals, optimize a score subject to an acyclicity constraint by an
off-the-shelf constrained solver, then read a DAG out of the fitted representation. The available
building blocks are PyTorch with autograd and RMSprop; a matrix exponential
(`scipy.linalg.expm` / `torch.matrix_exp`); the smooth acyclicity constraint `tr e^{M} - d` for a
*nonnegative* matrix `M`; the augmented-Lagrangian update scheme for a single equality-constrained
problem; and CAM-style preliminary neighbor selection and significance-test pruning as reusable
pre/post-processing.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGraphModel(nn.Module):
    """Differentiable score model over variables with an unknown DAG."""

    def __init__(self, d):
        super().__init__()
        self.d = d
        # TODO: the conditional architecture and its learnable graph representation.
        pass

    def log_likelihood(self, x):
        """(batch, d) per-variable conditional log-likelihoods."""
        # TODO: forward each variable's conditional and score the data.
        pass

    def graph_weights(self):
        """Current nonnegative edge-strength matrix for the acyclicity solver."""
        # TODO: fill from the representation we design.
        pass


# --- existing smooth acyclicity constraint (nonnegative input) -------------
def acyclicity(M):                       # tr e^{M} - d, zero iff M is a DAG
    return torch.trace(torch.matrix_exp(M)) - M.shape[0]


# --- existing augmented-Lagrangian outer loop ------------------------------
def fit(model, X, mu0=1e-3, lam0=0.0, eta=10.0, gamma=0.9, h_tol=1e-8):
    mu, lam = mu0, lam0
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    h_prev = np.inf
    while True:
        # inner: maximize score - lam*h - (mu/2) h^2 by minibatch RMSprop
        #        until a held-out criterion stops improving
        # TODO: inner subproblem loop, then read h at the approx solution
        h = None                          # placeholder for h(model) at convergence
        if h is not None and h <= h_tol:
            break
        lam = lam + mu * h
        if h > gamma * h_prev:
            mu = mu * eta
        h_prev = h
    return model


# --- existing CAM-style pre/post-processing (reused as-is) ------------------
def preliminary_neighbor_selection(X):    # variable-importance candidate parents
    pass

def prune_with_significance_test(model, X):  # drop parents failing a p-value test
    pass
```
