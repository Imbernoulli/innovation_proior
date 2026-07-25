The problem is to recover a directed acyclic causal graph from a purely observational sample of continuous variables X_1,...,X_p, when the true structural equations are nonlinear additive-noise models of the form X_j = sum_{k in pa(j)} f_{j,k}(X_k) + eps_j. The difficulty is threefold. First, identifiability: for linear-Gaussian structural equation models, forward and backward models are observationally identical, so no method can orient every edge from observational data alone. Constraint-based methods such as PC and score-based methods such as GES therefore return only a Markov equivalence class, leaving many edges undirected and making intervention distributions ambiguous. Second, computation: the space of DAGs grows super-exponentially in p, so direct search is infeasible. Third, statistics: in high dimensions with p comparable to or larger than n, every regression step must be regularized or screened, yet the regularization must not destroy the signal that identifies directions.

Existing approaches fall short in complementary ways. LiNGAM exploits non-Gaussian noise and the Darmois-Skitovic theorem to identify a linear DAG, but it is misspecified as soon as the mechanisms are nonlinear. The practical ANM algorithm of Mooij, Peters, and colleagues regresses each variable on all others and declares sinks by kernel independence testing on the residuals; it is sound in principle, but each step pays for an HSIC independence test against many regressors, and the procedure has no finite-sample high-dimensional theory. What is needed is a statistic that is cheap enough to evaluate thousands of times, yet still carries the directional information that disappears in the linear-Gaussian case.

I propose CAM, Causal Additive Models. CAM separates the problem into two stages that other methods fuse: estimate the topological order by an unregularized maximum-likelihood score, and only afterward select edges by regularized sparse additive regression. The key fact is that, for a nonlinear additive-noise model, the order is identifiable without any sparsity penalty. In the bivariate case X_2 = f(X_1) + N_2, a backward additive model X_1 = g(X_2) + N_1' with independent residual can exist only if f, the density of X_1, and the density of N_2 satisfy a specific third-order differential equation; generically this equation fails, so the direction is identifiable. The practically important corollary is that nonlinear f with Gaussian noise is identifiable, whereas the linear-Gaussian case is the degenerate non-identifiable exception. Under multivariate restricted-ANM conditions, this lifts to the whole DAG and to the set of true topological orderings.

The score that turns this into an algorithm is remarkably simple. Modeling the additive mechanisms with Gaussian errors and profiling out the unknown functions, the expected negative log-likelihood collapses to sum_j log(sigma_j) plus a constant, where sigma_j^2 is the residual variance from the best additive regression of X_j on its parents. Thus a structure is scored by ordinary additive-regression residual variances, with no independence test anywhere. Because the true ordering reproduces the true mechanisms, and any wrong ordering would imply a different minimal additive-noise DAG generating the same distribution which identifiability forbids, the true ordering is the strict minimizer of this unpenalized score, provided the model is genuinely nonlinear. In the linear-Gaussian limit the score correctly ties across all orderings, honestly reflecting that the order is not identifiable there. Once a correct order is known, the fully-connected DAG respecting that order is a super-DAG of the true graph, and by do-calculus it yields the same intervention distributions; edge selection is therefore an efficiency and readability step, not a correctness step, and that is exactly where regularization belongs.

CAM implements this in three modular stages. Preliminary neighborhood selection handles high dimensions by additively regressing each variable on all the others with boosting, running on the order of a hundred boosting iterations and keeping, per node, the candidates a term is selected more than two percent of the time (at least three selections out of a hundred), capped at ten neighbors; under an additive-influence condition this screen keeps all true parents while shrinking the search enough to scale to thousands of variables. IncEdge then greedily builds a fully-connected acyclic graph by repeatedly adding the edge that gives the largest decrease in log residual variance; the score decomposes over nodes, so only the column corresponding to the node that received the new edge is recomputed, and cycles are forbidden by maintaining a reachability matrix. Finally, pruning refits each node on its current parents and keeps a parent only if its smooth term in the additive fit is statistically significant, at a strict p-value cutoff of 0.001 that can be relaxed for small samples — since a correct order already gives a super-DAG with consistent causal effects, this conservative removal only costs a few spurious edges, never correctness. The spline basis count is kept small, around ten basis functions, preserving the nonlinearity that drives identifiability while keeping the residual-variance estimates stable.

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
    selects often. With 100 boosting steps, min_fraction_selected=0.02 means
    picked at least 3 times because the implementation uses a strict '>' cutoff;
    if more than max_neighbors pass, keep those strictly above the next frequency.
    Population condition: pa(j) subset A_j, with A_j screened into A_hat_j."""
    n, p = X.shape
    candidates = []
    for j in range(p):
        others = [k for k in range(p) if k != j]
        freq = gamboost_selection_frequency(X[:, j], X[:, others])
        above = [i for i, f in enumerate(freq) if f > min_fraction_selected]
        if len(above) > max_neighbors:
            cutoff = np.sort(freq)[::-1][max_neighbors]
            selected = [i for i in above if freq[i] > cutoff]
        else:
            selected = above
        candidates.append({others[i] for i in selected})
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
        pruned[j] = [pa[i] for i in range(len(pa)) if pvals[i] < cutoff_pval]
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
