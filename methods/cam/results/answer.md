# CAM (Causal Additive Models), distilled

CAM recovers a directed causal DAG from purely observational data under a nonlinear additive-noise
model `X_j = sum_{k in pa(j)} f_{j,k}(X_k) + eps_j`, `eps_j ~ N(0, sigma_j^2)` independent. Its
organizing idea is to **decouple two problems that other methods fuse**: estimate the topological
*order* with **unregularized** maximum likelihood, and select *edges* with **regularized** sparse
regression. The order is identifiable without any penalty because nonlinearity makes the DAG
identifiable; edge selection is then a standard variable-selection cleanup.

## Problem it solves

Constraint-based (PC/FCI) and score-based (GES) methods recover only the Markov equivalence class
for linear-Gaussian SEMs — many edges stay undirected — because forward and backward linear-
Gaussian models are observationally identical. CAM targets the *nonlinear* additive regime, where
the full DAG (hence every arrow, hence intervention distributions) is identifiable from the
observational distribution, and does so with a procedure that scales to many variables and is
consistent in low- and high-dimensional (`p >> n`) regimes, with explicit positive-gap conditions
for noise-distribution misspecification.

## Key ideas

1. **Identifiability via nonlinearity.** For a bivariate ANM `X_2 = f(X_1) + N_2`, a backward ANM
   exists only if `(f, p_{X_1}, p_{N_2})` solves a third-order differential equation in the
   log-densities `xi = log p_{X_1}`, `nu = log p_{N_2}`:
   `xi''' = xi''(-nu''' f'/nu'' + f''/f') - 2 nu'' f'' f' + nu' f''' + nu' nu''' f'' f'/nu'' - nu'(f'')^2/f'`.
   Generically this necessary equation fails, so the direction is identifiable. Special case:
   Gaussian `X_1, N_2` with a backward model forces `f` linear — so **nonlinear `f` + Gaussian
   noise is identifiable**. This lifts to the multivariate DAG and to the set of true orderings
   `Pi^0`.

2. **Likelihood collapses to a sum of log residual scales.** Under additive-Gaussian errors,
   profiling out the functions, the expected negative log-likelihood is
   `E[-log p_theta(X)] = sum_j log(sigma_j) + C`, `C = (p/2) log(2 pi) + p/2`, with
   `sigma_j^2 = E[(X_j - sum_k f_{j,k}(X_k))^2]` the best additive-regression residual variance.
   So a structure is scored by residual variances — ordinary regressions, no independence tests.

3. **The true order is the strict minimizer of the UNPENALIZED score.** Define the identifiability
   gap `xi_p = min_{pi not in Pi^0} p^{-1} sum_j (log sigma_j^{pi,0} - log sigma_j^0) >= 0`. A
   wrong order scoring below the truth would give a negative KL divergence — impossible — so wrong
   orders score higher or equal. If `xi_p = 0`, the best projected triangular additive-Gaussian
   distribution under a wrong order has KL zero from the truth; closedness gives an attained
   projection, and ANM identifiability rules out that different minimal additive-noise DAG. Hence
   `xi_p > 0` for nonlinear, three-times differentiable additive mechanisms under the closedness
   condition: **no sparsity penalty is needed to find the order.** (For linear-Gaussian, `xi_p = 0`,
   all orders tie, and unpenalized order search correctly fails.)

4. **A correct order already gives consistent causal effects.** If `pi-hat in Pi^0`, the
   fully-connected `D^{pi-hat}` is a super-DAG of `D^0`, and by do-calculus
   `p_{D^0}(x_j | do(X_k=x)) = p_{D^{pi-hat}}(x_j | do(X_k=x))`. So pruning is for statistical
   efficiency and a lower SHD, not for correctness — which is why edge selection can be done
   afterward, with regularization, as a separate step.

## Algorithm (three modular stages)

- **PNS — preliminary neighborhood selection** (for high-dim): additively regress each `X_j` on all
  others by boosting (gamboost); over 100 boosting iterations, keep the variables selected most
  often, capped at 10 candidate parents, and require selection frequency `> 0.02` (at least 3
  selections out of 100). Under the population additive-influence condition `pa(j) subseteq A_j`
  and screening `A_j subseteq A_hat_j`, true parents survive; this sparsifies the search to
  feasibility for thousands of nodes.
- **IncEdge — greedy order search.** Start empty; maintain a `p x p` gain matrix
  `gain[k,j]` as the increase in the SEMGAM score `-log(var(residuals))` from adding `k -> j`
  (equivalent to decreasing `log(sigma_j)`). Each step add the largest finite edge; the score
  decomposes over nodes so only column `j` is recomputed; forbid cycle-creating edges
  (reachability -> `-inf`). Without PNS or a parent cap, completing the build fixes a unique
  `pi-hat`.
  Fits use penalized regression splines (`gam`, mgcv) with ~10 basis functions; per-node score is
  `-log(var(residuals))`. Handles ~30 nodes without PNS.
- **Prune — regularized edge selection.** For each node, fit the additive model on its parents and
  keep a parent only if its smooth term is significant (gam p-value <= 0.001, raise for small `n`);
  equivalently use a Group-Lasso with sparsity-smoothness penalty. Screening keeps true edges with
  probability -> 1.

## Consistency

- **Theorem 1 (low-dim).** Under smoothness/tail/moment conditions (A1)-(A4) and `xi_p > 0`,
  `P[pi-hat in Pi^0] -> 1`.
- **Theorem 2 (misspecified).** Same conclusion with independent non-Gaussian errors if the
  residual-variance gap is assumed positive: either `(A1)-(A4)` with `xi_p > 0`, or `(A1)-(A3)`
  with the finite-basis gap `liminf_n xi_p^{a_n} > 0`. The Gaussian likelihood is a working score;
  the positive gap is the condition that makes misspecification harmless for order recovery.
- **Theorem 3 (high-dim, `p >> n`).** With PNS + restricted MLE, `(A1)`, `(A4)`, and `(B1)-(B4)`, if
  `max( sqrt(log(p)/n), max_{j,k} E[(f^0_{j,k}(X_k) - f^0_{n;j,k}(X_k))^2] ) = o(xi_p)`, or the
  same condition holds with `xi_p^{a_n}`, then `P[pi-hat in Pi^0] -> 1`. Lemma 0: under
  additive-influence `(B1)`, `pa(j) subseteq A_j`, so PNS is valid when `A_j subseteq A_hat_j` is
  screened in.

## Design choices and why

- **Gaussian likelihood as the score**, even under non-Gaussian noise: collapses to `sum log
  sigma_j`, one regression per candidate edge, far cheaper than HSIC independence tests; Theorem 2
  shows the order stays consistent under misspecification when the residual-variance gap condition
  holds.
- **Decouple unpenalized order from penalized edges:** order is identifiable via `xi_p > 0` without
  sparsity; over-inclusion in the super-DAG is harmless for causal effects; only edge existence
  benefits from a penalty. This is the core simplification.
- **Greedy decomposable search** instead of `p!` permutations or DAG-space search; recompute only
  the edited column; forbid cycles by reachability.
- **Small basis count (~10):** enough nonlinearity to preserve identifiability (a linear fit kills
  `xi_p`), few enough to keep residual-variance estimates stable — an identifiability-vs-variance
  tradeoff distinct from the usual prediction bias-variance one.
- **PNS first in high-dim:** restricts candidate parents for tractability and stability when
  `p >> n`, with the true parents provably retained.
- **Prune p-value 0.001:** conservative removal; since a super-DAG already gives consistent effects,
  a strict threshold is affordable; raise it for small `n`.

## Implementation skeleton (canonical structure)

```python
import numpy as np


def gam_fit(y, X_parents, num_basis=10):
    """Library GAM fit: y ~ sum_k s(X_parent_k, k=num_basis)."""
    raise NotImplementedError


def residual_variance(y, X_parents, num_basis=10):
    """sigma_j^2 for the current structure: residual variance after additive regression."""
    if X_parents.shape[1] == 0:
        return float(np.var(y, ddof=1))
    fitted, _ = gam_fit(y, X_parents, num_basis)
    return float(np.var(y - fitted, ddof=1))


def semgam_score(y, X_parents, num_basis=10):
    """Canonical SEMGAM node score: -log(var(residuals))."""
    return -np.log(residual_variance(y, X_parents, num_basis))


def gamboost_selection_frequency(y, X_others):
    """Library additive boosting role: fraction of boosting iterations selecting each term."""
    raise NotImplementedError


def preliminary_neighborhood_selection(X, max_neighbors=10, min_fraction_selected=0.02):
    """PNS (Step 1): keep, per node, the few candidate parents an additive boosting fit
    selects most often. With 100 boosting steps, min_fraction_selected=0.02 means
    picked at least 3 times because the implementation uses a strict '>' cutoff.
    Population condition: pa(j) subset A_j, with A_j screened into A_hat_j."""
    n, p = X.shape
    candidates = []
    for j in range(p):
        others = [k for k in range(p) if k != j]
        freq = gamboost_selection_frequency(X[:, j], X[:, others])
        selected = [i for i in np.argsort(freq)[::-1] if freq[i] > min_fraction_selected]
        candidates.append({others[i] for i in selected[:max_neighbors]})
    return candidates


def order_search_incedge(X, candidates, num_basis=10, max_num_parents=None):
    """IncEdge (Step 2): greedily add the edge with the largest decrease in
    log residual variance, implemented as the largest increase in -log(var(residuals)).
    Recompute only the edited column; forbid cycles by reachability."""
    n, p = X.shape
    if max_num_parents is None:
        max_num_parents = min(p - 1, round(n / 20))

    parents = [[] for _ in range(p)]
    node_score = np.array([semgam_score(X[:, j], X[:, []], num_basis) for j in range(p)])

    NEG_INF = -np.inf
    gain = np.full((p, p), NEG_INF)                      # gain[k, j]: score increase from k->j
    for j in range(p):
        for k in candidates[j]:
            new_score = semgam_score(X[:, j], X[:, [k]], num_basis)
            gain[k, j] = new_score - node_score[j]

    reach = np.eye(p, dtype=bool)                        # reach[a, b]: a ->...-> b
    while np.isfinite(gain).any():
        k, j = np.unravel_index(np.argmax(gain), gain.shape)
        if not np.isfinite(gain[k, j]):
            break
        parents[j].append(k)
        node_score[j] += gain[k, j]
        gain[k, j] = NEG_INF

        reach[k, j] = True                              # update reachability, forbid cycles
        desc_j, anc_k = np.where(reach[j])[0], np.where(reach[:, k])[0]
        reach[np.ix_(anc_k, desc_j)] = True
        gain[reach.T] = NEG_INF

        if len(parents[j]) >= max_num_parents:
            gain[:, j] = NEG_INF
            continue

        for kk in candidates[j]:                        # only column j changed
            if kk in parents[j] or kk == j or reach[j, kk]:
                continue
            new_score = semgam_score(X[:, j], X[:, parents[j] + [kk]], num_basis)
            gain[kk, j] = new_score - node_score[j]
    return parents


def prune(X, parents, cutoff_pval=0.001, num_basis=10):
    """Prune (Step 3): keep a parent only if its gam smooth term is significant."""
    p = len(parents)
    pruned = [[] for _ in range(p)]
    for j in range(p):
        pa = parents[j]
        if not pa:
            continue
        _, pvals = gam_fit(X[:, j], X[:, pa], num_basis)
        pruned[j] = [pa[i] for i in range(len(pa)) if pvals[i] <= cutoff_pval]
    return pruned


def run_causal_discovery(X, num_basis=10, do_pns=True, cutoff_pval=0.001):
    """CAM: identifiable nonlinear-ANM DAG via PNS + unregularized order search + pruning.
    Returns B with B[k, j] = 1 meaning edge k -> j (parent -> child)."""
    n, p = X.shape
    max_num_parents = min(p - 1, round(n / 20))
    candidates = (preliminary_neighborhood_selection(X) if do_pns
                  else [set(range(p)) - {j} for j in range(p)])
    parents = order_search_incedge(X, candidates, num_basis, max_num_parents)
    parents = prune(X, parents, cutoff_pval, num_basis)
    B = np.zeros((p, p))
    for j in range(p):
        for k in parents[j]:
            B[k, j] = 1.0
    return B
```

The canonical implementation uses `SEMGAM` as `-log(var(residuals))` from an `mgcv::gam` fit,
`numBasisFcts = 10`, `maxNumParents = min(p-1, round(n/20))`, optional PNS via `selGamBoost`
with `atMostThatManyNeighbors = 10` and `atLeastThatMuchSelected = 0.02`, and optional pruning via
`selGam` with `cutOffPVal = 0.001`. Output is a `p x p` adjacency matrix with `B[k,j] = 1`
meaning `k -> j`.
