NOTEARS-MLP landed exactly on the trade I predicted: precision jumped (mean 0.43–0.50, with 0.64 on ER20-Gauss seed 42) and SHD collapsed on the dense scenarios (SF20-GP from GraN-DAG's 92 to 35.7, ER20-Gauss from 76 to 51). But on SF20-GP the recall *cratered* to 0.13 (mean F1 0.153 — below GraN-DAG and even below DirectLiNGAM), with all three seeds in the 0.08–0.19 recall range. That is the linear-candidate bottleneck made visible: the linear NOTEARS skeleton plus the $0.15$ correlation screen never *proposes* the smoothly-nonlinear GP parents, so the gradient-boosted refinement — which can only prune what stage one offered — has nothing to recover them from. Every method so far either generated edges densely and over-connected (GraN-DAG) or generated them linearly and under-connected (NOTEARS-MLP). What I need is a method that gets the *nonlinear order right first* — without a linear skeleton or a free net — and then does disciplined edge selection. If I separate "get the order" from "select the edges," I can use the right tool for each.

I propose **CAM** (causal additive models): decouple the two problems and treat the order as the irreducibly causal part. The justification is best made at the population level. A DAG has a topological order; pick a correct one and the model is triangular — each variable regresses only on those earlier. If somebody handed me a correct order, discovery would dissolve into per-node nonlinear regression plus ordinary variable selection, a solved problem. And a correct order is enough for the causal answer *even with extra spurious edges*: the fully-connected DAG of a correct order is a super-DAG of the truth, and under modularity the intervention distributions it implies match the truth's, so edge pruning is an efficiency-and-readability step, not a correctness step. That tells me where penalties belong: the order search needs *no* regularization (identifiability supplies the gap that makes the true order the strict optimum), and the regularization goes into the *edge selection* afterward — exactly the split NOTEARS-MLP got wrong by entangling a linear skeleton with the edge selection.

What score recovers the order is the likelihood. Model the additive SEM with Gaussian errors; profiling out the functions, the expected negative log-likelihood collapses to $\sum_j \log \sigma_j$, where $\sigma_j^2$ is the residual variance of the best nonlinear regression of $X_j$ on its candidate parents. So a structure is scored purely by residual variances from ordinary nonlinear regressions — no independence tests, no kernel HSIC. The true order is the strict minimizer of this *unpenalized* score, because identifiability forces a positive gap: a wrong order scoring below the truth would imply a negative KL divergence, which is impossible, so wrong orders score higher or equal, and the closedness of the nonlinear additive class makes the gap strict. The linear-Gaussian tie ($\text{gap}=0$) reappears exactly where the order is not identifiable anyway. This is why a residual-variance order search beats DirectLiNGAM's linear residual test and NOTEARS-MLP's linear skeleton: it reads the nonlinear asymmetry directly off the regression residuals.

The fill I run is a **CAM-inspired heuristic**, not the full machinery, but it preserves the two ideas that matter — decouple order from edges, and generate candidates nonlinearly and order-respecting. It runs in three stages.

Stage one is the order, built greedily by residual variance with one specific rule. The **first** variable is the one with the *smallest marginal variance* — a root cause's variance is just its noise variance, so roots tend to have low total variance. Then at each subsequent step, for every not-yet-placed variable I fit a GradientBoostingRegressor of that variable on the *current ordering* (all already-placed variables) and append the one whose residual variance is *smallest*. This realizes the residual-variance score with boosted trees, and greedily appending the lowest-residual-variance node is a forward surrogate for the likelihood order search. The deviations from canonical CAM are deliberate: boosted trees instead of penalized splines, no preliminary neighbor selection (the regression at step $k$ uses *all* $k$ placed variables as predictors), and the root chosen by marginal-variance heuristic rather than likelihood gain.

Stage two is preliminary edge selection along the order. For each node at position `pos`, fit a GradientBoostingRegressor of it on *all earlier* variables and keep an edge from a candidate parent only if its **feature importance** exceeds $0.05$ — a tree-based surrogate for CAM's GAM significance test. Because it operates on the *full* set of predecessors (which, for a correct order, contains all true parents), it does not suffer NOTEARS-MLP's linear-candidate bottleneck: the candidate pool is the whole upstream set, fitted nonlinearly. This is the single most important structural difference from rung three — candidate generation is *nonlinear and order-respecting*, not linear.

Stage three is the regularized cleanup: a *partial-residual independence test*. For any node with more than one parent, for each parent $p$ I regress both $X_j$ and $X_p$ on the *other* parents (boosted trees again), take the two residuals, and if their absolute correlation is below $0.05$ I remove $p$. The logic is that if, after conditioning on the other parents, $X_p$ carries no leftover dependence with $X_j$, then $p$ is not a genuine parent given the others. This is the surrogate for CAM's significance-based pruning, using residual correlation as the cheap dependence proxy. It is the place the threshold lives — exactly where I argued regularization belongs — and it should give CAM the precision NOTEARS-MLP bought *without* paying the recall NOTEARS-MLP lost, because the edges were generated nonlinearly in the first place. The output obeys the harness convention $B[\text{child},\text{parent}] = 1$, i.e. $B[i,j] \ne 0$ means $j \to i$.

The same-named-vs-paper gap, plainly: this `cam` is a *heuristic* CAM — gradient-boosted residual-variance ordering (root by min-marginal-variance), feature-importance edge selection over all predecessors, and partial-residual-correlation pruning — not the canonical CAM with GAM splines, `gamboost` PNS, IncEdge likelihood-gain order search, and GAM-significance pruning. The decisive claim is that it fixes the recall collapse *without* surrendering precision: on SF20-GP the order-respecting candidate generation should recover the GP parents the linear skeleton missed, leaping recall and F1; on ER20-Gauss it should keep the high precision and restore recall; on ER12-LowSample, where the boosted ordering on 150 samples is the riskiest stage, the gain should be real but smaller, because the order search is variance-limited at small $n$. And that names CAM's own residual weakness for the next rung: the *order search itself* is greedy and residual-variance-driven, which a stronger, score-matching order recovery could improve.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    import os
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score

    n_samples, n_vars = X.shape
    seed = int(os.environ.get("SEED", "42"))

    # --- Step 1: Estimate causal ordering via greedy score-based search ---
    # For each variable, compute residual variance after nonlinear regression
    # on candidate parents. Variables with lower residual variance given
    # earlier variables are placed later in the ordering.

    remaining = list(range(n_vars))
    ordering = []

    while remaining:
        if len(ordering) == 0:
            # First variable: pick the one with highest marginal variance
            # (root causes tend to have variance = noise variance only)
            scores = []
            for j in remaining:
                scores.append(np.var(X[:, j]))
            # Pick the one with smallest variance (likely a root)
            best_idx = np.argmin(scores)
            ordering.append(remaining.pop(best_idx))
        else:
            # For each remaining var, fit nonlinear regression on current ordering
            best_score = np.inf
            best_var = None
            best_var_idx = None
            parents_X = X[:, ordering]
            for idx, j in enumerate(remaining):
                y = X[:, j]
                # Use gradient boosting as nonlinear regressor
                gbr = GradientBoostingRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    random_state=seed, subsample=0.8
                )
                gbr.fit(parents_X, y)
                residuals = y - gbr.predict(parents_X)
                resid_var = np.var(residuals)
                if resid_var < best_score:
                    best_score = resid_var
                    best_var = j
                    best_var_idx = idx
            ordering.append(remaining.pop(best_var_idx))

    # --- Step 2: Preliminary adjacency via nonlinear regression along ordering ---
    B = np.zeros((n_vars, n_vars))
    for pos in range(1, len(ordering)):
        j = ordering[pos]
        candidate_parents = ordering[:pos]
        y = X[:, j]
        pa_X = X[:, candidate_parents]

        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, subsample=0.8
        )
        gbr.fit(pa_X, y)
        importances = gbr.feature_importances_

        # Keep edges with importance above threshold
        threshold = 0.05
        for k, p in enumerate(candidate_parents):
            if importances[k] > threshold:
                B[j, p] = 1.0  # B[child, parent] = 1 means parent -> child

    # --- Step 3: Prune spurious edges via partial residual independence test ---
    for j in range(n_vars):
        parents = list(np.where(B[j, :] != 0)[0])
        if len(parents) <= 1:
            continue
        to_remove = []
        for p in parents:
            other_parents = [pp for pp in parents if pp != p]
            if len(other_parents) == 0:
                continue
            # Regress j on other parents
            gbr_j = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=seed
            )
            gbr_j.fit(X[:, other_parents], X[:, j])
            resid_j = X[:, j] - gbr_j.predict(X[:, other_parents])
            # Regress p on other parents
            gbr_p = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=seed
            )
            gbr_p.fit(X[:, other_parents], X[:, p])
            resid_p = X[:, p] - gbr_p.predict(X[:, other_parents])
            # Check correlation of partial residuals
            corr = np.abs(np.corrcoef(resid_j, resid_p)[0, 1])
            if corr < 0.05:
                to_remove.append(p)
        for p in to_remove:
            B[j, p] = 0.0

    return B
```
