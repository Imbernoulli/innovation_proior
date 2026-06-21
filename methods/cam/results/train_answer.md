The problem is to recover a directed acyclic causal graph from a purely observational sample of continuous variables X_1,...,X_p, when the true structural equations are nonlinear additive-noise models of the form X_j = sum_{k in pa(j)} f_{j,k}(X_k) + eps_j. The difficulty is threefold. First, identifiability: for linear-Gaussian structural equation models, forward and backward models are observationally identical, so no method can orient every edge from observational data alone. Constraint-based methods such as PC and score-based methods such as GES therefore return only a Markov equivalence class, leaving many edges undirected and making intervention distributions ambiguous. Second, computation: the space of DAGs grows super-exponentially in p, so direct search is infeasible. Third, statistics: in high dimensions with p comparable to or larger than n, every regression step must be regularized or screened, yet the regularization must not destroy the signal that identifies directions.

Existing approaches fall short in complementary ways. LiNGAM exploits non-Gaussian noise and the Darmois-Skitovic theorem to identify a linear DAG, but it is misspecified as soon as the mechanisms are nonlinear. The practical ANM algorithm of Mooij, Peters, and colleagues regresses each variable on all others and declares sinks by kernel independence testing on the residuals; it is sound in principle, but each step pays for an HSIC independence test against many regressors, and the procedure has no finite-sample high-dimensional theory. What is needed is a statistic that is cheap enough to evaluate thousands of times, yet still carries the directional information that disappears in the linear-Gaussian case.

I propose CAM, Causal Additive Models. CAM separates the problem into two stages that other methods fuse: estimate the topological order by an unregularized maximum-likelihood score, and only afterward select edges by regularized sparse additive regression. The key fact is that, for a nonlinear additive-noise model, the order is identifiable without any sparsity penalty. In the bivariate case X_2 = f(X_1) + N_2, a backward additive model X_1 = g(X_2) + N_1' with independent residual can exist only if f, the density of X_1, and the density of N_2 satisfy a specific third-order differential equation; generically this equation fails, so the direction is identifiable. The practically important corollary is that nonlinear f with Gaussian noise is identifiable, whereas the linear-Gaussian case is the degenerate non-identifiable exception. Under multivariate restricted-ANM conditions, this lifts to the whole DAG and to the set of true topological orderings.

The score that turns this into an algorithm is remarkably simple. Modeling the additive mechanisms with Gaussian errors and profiling out the unknown functions, the expected negative log-likelihood collapses to sum_j log(sigma_j) plus a constant, where sigma_j^2 is the residual variance from the best additive regression of X_j on its parents. Thus a structure is scored by ordinary additive-regression residual variances, with no independence test anywhere. Because the true ordering reproduces the true mechanisms, and any wrong ordering would imply a different minimal additive-noise DAG generating the same distribution which identifiability forbids, the true ordering is the strict minimizer of this unpenalized score, provided the model is genuinely nonlinear. In the linear-Gaussian limit the score correctly ties across all orderings, honestly reflecting that the order is not identifiable there. Once a correct order is known, the fully-connected DAG respecting that order is a super-DAG of the true graph, and by do-calculus it yields the same intervention distributions; edge selection is therefore an efficiency and readability step, not a correctness step, and that is exactly where regularization belongs.

CAM implements this in three modular stages. Preliminary neighborhood selection handles high dimensions by first additively regressing each variable on all others and retaining a small set of candidate parents whose univariate additive fit explains a meaningful fraction of the variance; under an additive-influence condition this screen keeps all true parents while shrinking the search enough to scale to thousands of variables. IncEdge then greedily builds a fully-connected acyclic graph by repeatedly adding the edge that gives the largest decrease in log residual variance; the score decomposes over nodes, so only the column corresponding to the node that received the new edge is recomputed, and cycles are forbidden by maintaining a reachability matrix. Finally, pruning fits each node on its current parents and removes parents that do not meaningfully reduce the residual variance, using standard sparse additive regression. The spline basis count is kept small, around ten basis functions, preserving the nonlinearity that drives identifiability while keeping the residual-variance estimates stable.

```python
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import RidgeCV


def fit_additive_model(y, X_parents, num_basis=10):
    """Fit y ~ sum_k f_k(X_parents[:, k]) with penalized regression splines."""
    if X_parents.shape[1] == 0:
        return np.zeros_like(y)
    blocks = []
    for k in range(X_parents.shape[1]):
        spl = SplineTransformer(
            n_knots=num_basis, degree=3, include_bias=False,
            extrapolation="constant"
        )
        blocks.append(spl.fit_transform(X_parents[:, [k]]))
    X_design = np.concatenate(blocks, axis=1)
    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    model.fit(X_design, y)
    return model.predict(X_design)


def residual_variance(y, X_parents, num_basis=10):
    if X_parents.shape[1] == 0:
        return float(np.var(y, ddof=1))
    fitted = fit_additive_model(y, X_parents, num_basis)
    return float(np.var(y - fitted, ddof=1))


def semgam_score(y, X_parents, num_basis=10):
    return -np.log(residual_variance(y, X_parents, num_basis) + 1e-12)


def preliminary_neighborhood_selection(X, max_neighbors=10, min_fraction=0.02):
    """Keep, per node, a small set of candidate parents by univariate screen."""
    n, p = X.shape
    candidates = []
    for j in range(p):
        others = [k for k in range(p) if k != j]
        base = residual_variance(X[:, j], X[:, []])
        scores = np.array([
            base - residual_variance(X[:, j], X[:, [k]]) for k in others
        ])
        above = np.where(scores / base > min_fraction)[0]
        if len(above) > max_neighbors:
            order = np.argsort(scores[above])[::-1][:max_neighbors]
            selected = above[order]
        else:
            selected = above
        candidates.append({others[i] for i in selected})
    return candidates


def order_search_incedge(X, candidates, num_basis=10, max_num_parents=None):
    """Greedily add the edge with the largest SEMGAM score gain."""
    n, p = X.shape
    if max_num_parents is None:
        max_num_parents = min(p - 1, max(1, round(n / 20)))

    parents = [[] for _ in range(p)]
    node_score = np.array([
        semgam_score(X[:, j], X[:, []], num_basis) for j in range(p)
    ])
    NEG_INF = -np.inf
    gain = np.full((p, p), NEG_INF)
    for j in range(p):
        for k in candidates[j]:
            gain[k, j] = (
                semgam_score(X[:, j], X[:, [k]], num_basis) - node_score[j]
            )

    reach = np.eye(p, dtype=bool)
    while np.isfinite(gain).any():
        k, j = np.unravel_index(np.argmax(gain), gain.shape)
        if not np.isfinite(gain[k, j]):
            break
        parents[j].append(k)
        node_score[j] += gain[k, j]
        gain[k, j] = NEG_INF

        reach[k, j] = True
        desc_j = np.where(reach[j])[0]
        anc_k = np.where(reach[:, k])[0]
        reach[np.ix_(anc_k, desc_j)] = True
        gain[reach.T] = NEG_INF

        if len(parents[j]) >= max_num_parents:
            gain[:, j] = NEG_INF
            continue

        for kk in candidates[j]:
            if kk in parents[j] or kk == j or reach[j, kk]:
                continue
            new_score = semgam_score(
                X[:, j], X[:, parents[j] + [kk]], num_basis
            )
            gain[kk, j] = new_score - node_score[j]
    return parents


def prune(X, parents, rel_threshold=0.05, num_basis=10):
    """Keep a parent only if removing it noticeably increases residual variance."""
    p = len(parents)
    pruned = [[] for _ in range(p)]
    for j in range(p):
        pa = parents[j]
        if len(pa) == 0:
            continue
        full_var = residual_variance(X[:, j], X[:, pa], num_basis)
        for k in pa:
            other = [pp for pp in pa if pp != k]
            if len(other) == 0:
                reduced = np.var(X[:, j], ddof=1)
            else:
                reduced = residual_variance(X[:, j], X[:, other], num_basis)
            if (reduced - full_var) / reduced > rel_threshold:
                pruned[j].append(k)
    return pruned


def run_causal_discovery(X, num_basis=10, do_pns=True, rel_threshold=0.05):
    """CAM: nonlinear-ANM DAG via PNS + unregularized order search + pruning."""
    n, p = X.shape
    max_num_parents = min(p - 1, max(1, round(n / 20)))
    candidates = (
        preliminary_neighborhood_selection(X) if do_pns
        else [set(range(p)) - {j} for j in range(p)]
    )
    parents = order_search_incedge(X, candidates, num_basis, max_num_parents)
    parents = prune(X, parents, rel_threshold, num_basis)
    B = np.zeros((p, p))
    for j in range(p):
        for k in parents[j]:
            B[k, j] = 1.0
    return B
```
