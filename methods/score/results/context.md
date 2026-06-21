# Context: recovering a nonlinear causal DAG from the score of the data distribution (circa 2021-2022)

## Research question

We observe samples of a random vector `X = (X_1, ..., X_d)` and we want the *directed acyclic
graph* `G` of who causes whom — oriented edges, not an undirected skeleton, from purely
observational data. Under a nonlinear additive-noise model `X_j := f_j(X_{pa(j)}) + N_j` with
independent noise and nonlinear `f_j`, `G` is identifiable from `P_X`. The space of DAGs is
super-exponential in `d`, and the irreducibly causal part is the *topological order*, since a
correct order reduces the problem to per-node regression plus variable selection. How can the
topological order be recovered from the observational distribution?

## Background

**Additive noise models and identifiability.** For `X_j = f_j(X_{pa(j)}) + N_j` with mutually
independent `N_j` and nonlinear `f_j`, the DAG is identifiable from the observational distribution
(Hoyer, Janzing, Mooij, Peters & Scholkopf 2008; Peters, Mooij, Janzing & Scholkopf 2014) — even
under Gaussian noise, because the nonlinearity itself breaks the cause/effect symmetry. The
practical consequence is that the *order* is identifiable: a correct topological order makes the
SEM triangular, and pruning the resulting super-DAG is a downstream variable-selection step
(efficiency, not correctness).

**The score function.** The score is `s(x) = ∇_x log p(x)`, the gradient field of the log-density.
For a Gaussian-noise ANM the log-density is a sum of Gaussian noise terms,
`log p(x) = sum_j -(x_j - f_j(pa(j)))^2 / (2 σ_j^2) + const`. Differentiating, `s_j(x)` picks up the
term where `x_j` is its own noise argument *and* the terms where `x_j` appears inside its children's
mechanisms. The second derivative along `x_j` — the `j`-th diagonal entry of the Hessian of
`log p` — is the load-bearing quantity: the Gaussian-noise term contributes the *constant* `-1/σ_j^2`
(because `f_j` cannot depend on `x_j`), so the only `x`-dependence in that diagonal Hessian entry
comes from `x_j`'s children. This is what ties the score's Hessian to the graph.

**Stein identities for score estimation.** The score and its Jacobian can be estimated
non-parametrically from samples without ever forming the density, via Stein's identities with a
reproducing kernel. With a Gaussian kernel `K` (Gram matrix `K_{ij} = exp(-||x_i-x_j||^2/(2s^2))/s`,
bandwidth `s`), the first-order Stein estimate of the score at the sample points is
`G = (K + η_G I)^{-1} ∇K`, and a second-order identity yields the diagonal of the score's Jacobian
(the diagonal Hessian of `log p`), `-G^2 + (K + η_H I)^{-1} ∇^2 K`. Both are ridge-regularized
linear solves, `O(n^3)` per evaluation.

## Baselines

**CAM (Buhlmann, Peters & Ernest 2014).** Decouples a topological-order search (restricted maximum
likelihood, scored by residual variances) from per-node edge selection (sparse regression), with
preliminary neighbor selection and significance pruning. The order is found by a greedy search that
repeatedly fits regressions of every remaining variable on the current prefix — `O(d^2)` fits on
growing predictor sets — and it assumes additive mechanisms.

**RESIT (Peters et al. 2014).** Regresses each variable on candidates and tests residual
independence to find an order, using kernel independence tests.

**GraN-DAG / NOTEARS-MLP (Lachapelle et al. 2020; Zheng et al. 2020).** Continuous-constraint
neural methods that learn the graph by gradient descent under the smooth acyclicity constraint
`tr e^{·} - d = 0`.

## Evaluation settings

Synthetic nonlinear-ANM data: a ground-truth DAG from a random-graph scheme (Erdos-Renyi or
scale-free / Barabasi-Albert) at a chosen edge density, data generated as `X_j = f_j(pa(j)) + N_j`
with `f_j` drawn from a Gaussian process and various noise families; graph sizes of order 10-50
nodes, sample sizes hundreds to a few thousand. Metrics on the directed edge set: SHD (missing +
extra + reversed edges), SID, and F1/precision/recall.

## Code framework

The method plugs into a structure-learning harness: load observational data `X`, recover a
topological order, then prune edges. The available building blocks are numpy/scipy linear algebra
(kernel Gram matrices, regularized solves), and CAM-style nonlinear regression pruning as a reusable
post-processing step. What is open is the order-recovery stage.

```python
import numpy as np


def stein_hessian_diagonal(X, eta_G, eta_H, s=None):
    """Estimate the diagonal of the Hessian of log p_X at the sample points,
    via first- and second-order Stein identities with a Gaussian kernel.

    Returns an (n, d) matrix whose (k, j) entry estimates d^2 log p / d x_j^2
    at sample k.
    """
    # TODO: Gaussian Gram matrix K with bandwidth s; kernel gradient nablaK and
    #       second derivative nabla2K; score G = (K + eta_G I)^{-1} nablaK;
    #       diagonal Hessian = -G^2 + (K + eta_H I)^{-1} nabla2K.
    pass


def compute_topological_order(X, eta_G=1e-3, eta_H=1e-3):
    """Recover a topological order from the diagonal Hessian estimates."""
    # TODO: use the diagonal Hessian to determine an ordering of variables.
    pass


def prune_edges(X, order):
    """CAM-style nonlinear edge selection along a known order (reused as-is)."""
    pass


def run_causal_discovery(X):
    order = compute_topological_order(X)
    return prune_edges(X, order)
```
