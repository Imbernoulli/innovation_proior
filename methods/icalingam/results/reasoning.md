I have a pile of observational vectors `x = (x_1, ..., x_m)` and I want the causal graph behind them — who causes whom, with what strength — and I have no experiments, no time stamps, no ordering handed to me. Let me be honest about why this is hard before I reach for any tool. I'll commit to the model that everyone in the linear tradition uses: the process is recursive, each variable is a linear function of the variables that come before it causally plus its own noise, `x_i = sum_{k(j) < k(i)} b_ij x_j + e_i`, with the disturbances `e_i` independent of each other, no hidden common cause, and the dependency graph acyclic. Stack the equations and it's `x = Bx + e`, where `B` collects the `b_ij`. Acyclicity has a clean algebraic face: there is some ordering of the variables under which `B` is *strictly* lower triangular — lower triangular with zeros on the diagonal, because a variable can't be its own parent and can only depend on earlier ones. I just don't know that ordering; in my data the rows of `x` are in some arbitrary order, so `B` is "permutable to strict lower triangularity" but not actually triangular as I see it.

The wall is direction, and I should feel exactly how solid it is before I try to climb it. The standard move is to assume the `e_i` are Gaussian and work from the covariance matrix. So let me push that to its conclusion. If every `e_i` is Gaussian and the map is linear, then `x` is jointly Gaussian, and a multivariate Gaussian is *completely* described by its mean and covariance — there is nothing else to observe, no higher-order structure, nothing. Now take the smallest case, two variables. Model one: `x_1 = e_1`, `x_2 = 0.8 x_1 + e_2`, and pick `var(e_1) = 1`, `var(e_2) = 0.36`. Then `var(x_1) = 1`, `var(x_2) = 0.8^2 · 1 + 0.36 = 0.64 + 0.36 = 1`, and `cov(x_1, x_2) = 0.8 · var(x_1) = 0.8`. Model two, the *reverse* arrow: `x_1 = 0.8 x_2 + e_1`, `x_2 = e_2`, with `var(e_2) = 1`, `var(e_1) = 0.36`. Then `var(x_2) = 1`, `var(x_1) = 0.8^2 + 0.36 = 1`, `cov = 0.8` again. Same means, same variances, same covariance — and since both are Gaussian, the *same joint density*. The two `B` matrices are completely different causal claims (`b_21 = 0.8` versus `b_12 = 0.8`), yet no statistic of the data can tell them apart. This isn't a weakness of any particular algorithm; it's information-theoretic. Anything that reads only second-order statistics — the whole constraint-based and Gaussian-SEM apparatus — recovers at best a Markov equivalence class: the skeleton and the orientations a collider forces, with the rest of the edges left undirected and the coefficients unidentified. If I stay Gaussian, I am provably stuck on direction for any edge a collider doesn't pin down.

So the covariance is a dead end *by assumption*. The assumption is Gaussianity. What if I drop it? There's a hint that this is the right lever: for two variables, people have shown the cause-effect direction *is* recoverable when the noise is non-Gaussian — the asymmetry shows up in third moments, in the regression residuals being non-normal in a direction-dependent way (Dodge and Rousson 2001; Shimizu and Kano 2006). That's a proof of concept that non-Gaussianity carries directional information the covariance throws away. But it's two variables, and it doesn't hand me coefficients or an ordering for `m` variables. I need to find the structure that turns "non-Gaussianity breaks the symmetry" into a general procedure.

Let me stare at my own model written the other way. From `x = Bx + e`, solve for `x`: `(I - B) x = e`, so `x = (I - B)^{-1} e`. Call `A = (I - B)^{-1}`. Then `x = A e`. Now read what this says: my observed vector is a fixed linear mixing `A` of the disturbance vector `e`, and the components of `e` are — by my own model assumption — mutually independent. If I also insist they're non-Gaussian, then `x = A e` is *exactly* the linear independent component analysis model: observed data is an invertible linear mixture of independent non-Gaussian sources. That's not a metaphor; it's the same equations. The disturbances `e_i` are the independent components; `A` is the mixing matrix; `W = A^{-1} = I - B` is the separating matrix.

Now the old Gaussian obstruction has a precise opposite. There's a theorem (Comon 1994) that the ICA mixing matrix is identifiable from the data when the sources are independent and at most one source is Gaussian — identifiable up to permutation, scaling, and sign of its columns, with no rotational ambiguity. In this structural model I am making every disturbance non-Gaussian, so the condition is satisfied. Contrast that with Gaussian sources, where any orthogonal rotation `A R` produces the same distribution (rotating independent Gaussians gives independent Gaussians of the same covariance), which is precisely the rotational freedom that left me unable to tell direction. So the non-Gaussianity I was tempted to assume away for analytic convenience is the exact thing that collapses the rotational ambiguity and makes `A` — and therefore `W = I - B`, and therefore `B` — recoverable. The two-variable direction result generalizes because the underlying engine is ICA identifiability. I should estimate `W` by ICA and read `B` off it.

But "up to permutation, scaling, and sign" is doing a lot of work in that theorem, and those three indeterminacies are exactly what stand between me and `B`. ICA hands me some matrix `W_ica` that, in the population limit, equals `P D W` — the true separating matrix `W = I - B` with its rows permuted by an unknown permutation matrix `P` and each row scaled and sign-flipped by an unknown diagonal `D` (`W_ica = P D W = P D A^{-1}`). In a typical ICA application you don't care: the order, scale, and sign of recovered audio sources is meaningless, so people fix scale by forcing unit-variance components and ignore the rest. I cannot ignore them. The rows of `W` correspond to the disturbance variables `e_i`, the columns to the observed variables `x_i`; a random row permutation means I've lost the correspondence between which recovered component is the noise of which observed variable. And the scaling means the rows aren't yet in the units of `I - B`. I have to *undo* the permutation and *fix* the scaling, and nothing in ICA does that for me. ICA gives me the raw material; the causal content is in resolving its indeterminacies. So the two sub-problems are: find the right row permutation `P^{-1}`, and find the right row scaling.

Take the scaling first because it's easier and it tells me what "correct" even means. In the SEM convention each variable's own coefficient is one — the equation `x_i = (stuff) + e_i` has coefficient `+1` on `e_i` — so `W = I - B` has an all-ones diagonal: `W_ii = 1 - b_ii = 1 - 0 = 1`. That's a fixed, known anchor. If I had the rows in the right order, every diagonal entry of the true `W` would be exactly one, and to remove ICA's per-row scaling I'd just divide each row by its own diagonal entry. That pins the scale *and* the sign (dividing by the signed diagonal element fixes the sign too). So the scaling indeterminacy is solved the instant I know the row order, by normalizing the diagonal to ones. Which throws all the weight onto the permutation.

The permutation is the subtle one, so let me think hard about what distinguishes the correct row order from the `m! - 1` wrong ones. The correct `W = I - B`: since `B` is permutable to strict lower triangular, `W = I - B` is permutable (by the *same* row-and-column permutation) to lower triangular with a *nonzero* diagonal — the ones from `I`. So the correct, properly-ordered `W` is lower triangular with no zeros on its diagonal. ICA scrambled the rows. The property I need is stronger than a heuristic: among all ways to permute the rows back, exactly one yields a matrix with no zeros on the diagonal, and that one is correct.

Write the correct separating matrix (aligned, disturbance-`i` paired with variable-`i`) as `W = P_d M P_d^T`, where `M` is lower triangular with a nonzero diagonal and `P_d` is the permutation representing the true causal order. ICA returns the rows in random order: `W_ica = P_ica W = P_ica P_d M P_d^T`. Set `P_1 = P_ica P_d` (the combined row permutation acting on `M`) and `P_2 = P_d` (the column permutation), so `W_ica = P_1 M P_2^T`. I want to know: for which choices of `P_1, P_2` does `P_1 M P_2^T` have a fully nonzero diagonal? The support pattern gives the obstruction. Let `K` be the lower-triangular matrix of all ones, with every slot that could be nonzero in `M` filled. The number of potentially nonzero diagonal entries after row and column permutations is `tr(P_1 K P_2^T)`, and by cyclicity this is `tr(K P_2^T P_1)`. So only the combined column permutation `Q = P_2^T P_1` matters for the pattern. Now suppose `Q` is not the identity. Then some original column `i` must move to a position `j < i`; if no column moved left, every moved column would move right, and the finite set of column positions could not be filled. In the permuted `K`, the diagonal entry in new column `j` comes from old entry `K[j, i]`, and `j < i` lies strictly above the lower-triangular support, so that entry is zero. Because `K` is the densest possible lower-triangular support, the same diagonal slot is zero for any lower-triangular `M`. Thus a zero-free diagonal forces `Q = I`, so `P_2^T P_1 = I` and `P_1 = P_2`. Conversely, if `P_1 = P_2`, then `P_1 M P_1^T` just applies the same relabeling to rows and columns; its diagonal entries are the original diagonal entries of `M` in a different order, and those are all nonzero. So a permuted lower-triangular matrix with nonzero diagonal has a fully nonzero diagonal if and only if its row and column permutations are equal. Translating back: the only row permutation `P_ica^{-1}` of `W_ica` that yields a zero-free diagonal is the one that re-pairs each disturbance with its own variable — the correct one. The DAG structure is what makes the permutation unique, and it gives me a concrete, decidable criterion: find the row order that makes the diagonal nonzero.

I should double-check that I actually *need* this DAG constraint and I'm not overcomplicating. What goes wrong without acyclicity? Let me look at the two-variable mixing directly. Parameterize `x = A e` with `A = (1/(1 - b_12 b_21)) [[1, b_12],[b_21, 1]] diag(sigma_1, sigma_2)` acting on normalized sources `s = (e_1/sigma_1, e_2/sigma_2)`. Now try a *different* parameter set: `b_12' = 1/b_21`, `b_21' = 1/b_12`, `sigma_1' = sigma_2/b_21`, `sigma_2' = sigma_1/b_12`, together with swapping and sign-flipping the sources `s_1' = -s_2`, `s_2' = -s_1`. Grind through the matrix product — pull the `1/(1 - b_12 b_21)` out front, multiply the permuted, rescaled mixing matrix by the swapped sources — and after the dust settles the two parameterizations produce *identical* data: same mixing of the same independent sources, same distribution, with wildly different `b` values. So without the acyclicity constraint there genuinely are multiple `B`'s giving the same data, and the permutation problem is real, not a technicality. But notice: if the original system *is* a DAG (say `b_12 = 0`), then the primed system needs `b_21' = 1/b_12 = infinity` — it's not a valid finite model. The DAG constraint is exactly what kills the spurious alternative. Good: the uniqueness lemma above and this example are two sides of the same coin, and they tell me to lean on acyclicity hard.

Now, finite data. ICA on a finite sample never returns exact zeros — every entry of `W_ica` is some small nonzero number, so *every* row permutation gives a diagonal that's technically nonzero. The clean criterion "the diagonal has no zeros" degenerates. I need a soft version: among permutations, prefer the one whose diagonal entries are as *far from zero* as possible — large in absolute value — since the correct alignment puts the structurally-large unit-ish entries on the diagonal and the structurally-small (true-zero) entries off it. A natural cost: penalize small diagonal magnitudes heavily. Take `sum_i 1/|W_ii|` after permutation and minimize it; a near-zero diagonal entry blows that term up, so minimizing it pushes large entries onto the diagonal.

I shouldn't just assert that cost — let me see if it falls out of something principled, because a maximum-likelihood reading would be reassuring. Model each disturbance value with a generalized-Gaussian density, `log p(e_it) = -|e_it|^alpha / beta + Z`. For a candidate row placement, that row will later be divided by its diagonal entry `W_ii`, so the disturbance value entering the density is scaled as `e_it / W_ii`. Substituting that into the log likelihood gives, up to constants, `-sum_i (1/(beta |W_ii|^alpha)) sum_t |e_it|^alpha`. To maximize over the row correspondence, I therefore want large diagonal magnitudes, with a sharper penalty when `alpha` is larger. If I make the simplifying assumption that all independent components have the same density, the row-correspondence objective becomes `min over row-perms sum_i 1/|W_ii|^alpha`. Fixing `alpha = 1` keeps the same qualitative behavior and gives `sum_i 1/|W_ii|`. So the heavy penalty on small diagonal entries isn't ad hoc; it's the ML objective for finding the correct correspondence under this simple density model.

Now the computational shape. Naively, searching over all `m!` row permutations for the one minimizing `sum_i 1/|W_ii|` is hopeless beyond a handful of variables. But look at the structure of the objective. If row `r` is placed at diagonal position `c`, the contribution is `C_{r,c} = 1/|W_ica[r,c]|`; equivalently, if `phi(c)` is the row chosen for diagonal column `c`, the total is `sum_c C_{phi(c),c}` over a permutation `phi`. That is precisely the linear assignment problem: match rows to columns minimizing the summed assignment cost. It's not combinatorially hopeless at all; it's solved in `O(m^3)` by the Hungarian algorithm (Kuhn 1955), and there's a solver sitting on the shelf. So I build the cost matrix `C` with `C_{r,c} = 1 / |W_ica[r, c]|`, run linear assignment to get the row-to-diagonal matching, and apply it to permute the rows of `W_ica`. The implementation detail matters: the assignment solver returns, for each original row, the column it is matched to, and I scatter row `r` into new row `c`; then the matched entries land exactly on the diagonal.

With the rows now correctly ordered, the scaling is the easy step I set up earlier: take the diagonal of the permuted `W`, call it `D = diag(PW_ica)`, and divide each row by its own diagonal entry, `W_estimate = PW_ica / D` (broadcasting `D` down the rows). Now `W_estimate` has all ones on the diagonal — it's an estimate of `I - B` in the right units, with scaling and sign resolved. Then `B = I - W_estimate`. I have all the connection strengths.

Except I'm not done, because for visualization and for a clean directed graph I want an explicit causal *order* `k(i)` under which `B` is strictly lower triangular — and on finite data `B` won't contain exact zeros, so the upper triangle won't be exactly zero under any permutation. If the estimates *were* exact, finding the order is trivial: in `B` the row for variable `i` holds its incoming coefficients (`B[i,j]` is the effect of `j` on `i`), so an all-zero row is a variable with no remaining parents — a source. A strict-lower-triangular structure always has at least one such all-zero row, so I can peel it off — record that variable, delete its row and column, and repeat until the matrix is empty; peeling sources one at a time builds a valid topological order, and if at some point no all-zero row exists, the remaining variables form a cycle and the matrix isn't a DAG. That's the exact-case test.

For finite data I need an approximate version, and I'll lean again on "true zeros are small." How many zeros does a strict-lower-triangular `m x m` matrix have? The diagonal (`m` entries) plus the entire upper triangle (`m(m-1)/2` entries), which is `m + m(m-1)/2 = m(m+1)/2`. So in the true `B`, the `m(m+1)/2` structural zeros are the diagonal and all entries above the causal-order diagonal. My estimate corrupts them to small nonzeros, but they should still be among the smallest entries. The implemented approximation sorts all `m^2` entries by absolute value, sets the `m(m+1)/2` smallest to exact zero, and then walks through the remaining entries from smallest upward; at each step it zeroes the next entry and runs the peel-off test, stopping at the first valid order. If in the true matrix all structural zeros are smaller than all true nonzeros, the first valid peel gives the right order; in general it's a greedy approximation, but it's cheap and the magnitude ordering is informative. That gives me the causal order `k(i)`.

Let me make the peel-off concrete since it carries the acyclicity test. Keep a list of original indices. Repeatedly: find a row whose entries are all zero — in the working matrix, a row with `sum of |entries| == 0`. Under `B[i,j] =` effect of `j` on `i`, that row is a variable with no remaining parents. If no such row exists, the current zero pattern cannot be peeled as an acyclic order, so this candidate zeroing fails and I need to zero more. Otherwise take the first such row, append its original index to the order, and delete that row and the corresponding column from the working matrix. When the matrix is empty, if I've collected all `m` indices I have a valid exogenous-first order; if I broke out early, return failure.

So I have an order `k(i)` and a coefficient matrix. But the raw `B = I - W_estimate` is fully dense — every off-diagonal entry is some nonzero number, including all the ones that should be structural zeros. I haven't actually *pruned* anything; I've only found an ordering that makes the upper triangle "as zero as possible." For a usable directed graph I need to decide which edges are real. The honest, statistically grounded way to get the final coefficients is to *re-estimate* them by regression along the discovered order, with sparsity. Walk the variables in causal order; for each variable `target` at position `i`, its only possible parents are the variables earlier in the order, `predecessors = order[:i]`. Regress `x[:, target]` on `x[:, predecessors]`. Plain least squares would give dense coefficients again, so use a sparse regression that can drive genuinely-absent edges to exactly zero. The adaptive lasso is the right tool: first fit ordinary least squares to get pilot coefficients, form per-predictor weights `w_j = |coef_j|^gamma` (with `gamma = 1`), rescale the predictors by these weights, fit a lasso with the penalty level chosen by an information criterion (BIC via least-angle-regression), and multiply the resulting coefficients back by the weights. The weighting gives the consistent-selection property — strong pilot coefficients get a light penalty (kept), weak ones a heavy penalty (zeroed) — so spurious weak edges are pruned while true edges survive with good estimates. The first variable in the order has no predecessors, so it stays exogenous with an all-zero row. Fill the recovered, signed coefficients into the adjacency matrix at `B[target, predecessors]`. That's the final directed weighted DAG, and I return it with the convention that a nonzero `B[i, j]` means the edge `j -> i`.

One honest caveat I should keep in front of me about the whole pipeline: the ICA step is a non-convex optimization. Maximizing non-Gaussianity (the negentropy contrast — `J_G(w) = (E{G(w^T z)} - E{G(nu)})^2` after whitening, with `G(u) = (1/a) log cosh(a u)` and its `tanh` derivative driving the fixed-point Newton iteration `w+ = E{z g(w^T z)} - E{g'(w^T z)} w`, components decorrelated symmetrically) has multiple local optima, and a run started from a poor initialization can land in the wrong one, giving an unreliable `W` and hence a wrong graph. There's no convexity guarantee here; the stability depends on the initialization and step size, and the right fix at the algorithmic-parameter level is genuinely fiddly. I'll fix the random seed for reproducibility and give FastICA a generous iteration budget, but I should be clear-eyed that this is the soft spot of the approach — the part of the method that rests on an optimization that can fail, as opposed to the permutation/scaling logic, which is exact given a good `W`.

Now let me assemble exactly the pipeline into the function the harness wants — input `X` of shape `(n_samples, n_variables)`, output the adjacency matrix:

```python
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression, LassoLarsIC
from sklearn.utils import check_array


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """Recover a directed weighted DAG from observational non-Gaussian data.

    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B, B[i, j] != 0 read as the edge j -> i.
    """
    X = check_array(X)
    seed = int(os.environ.get("SEED", "42"))

    # 1. ICA: x = A e is the linear ICA model; estimate the separating matrix
    #    W_ica = P D (I - B). Rows are in arbitrary order / scale / sign.
    ica = FastICA(max_iter=1000, random_state=seed)
    ica.fit(X)
    W_ica = ica.components_

    # 2. Undo the row permutation. The correct order is the one putting large
    #    entries on the diagonal; min sum_i 1/|W_ii| is a linear assignment
    #    problem (cost C_ij = 1/|W_ica[i,j]|), solved by the Hungarian method.
    _, col_index = linear_sum_assignment(1 / np.abs(W_ica))
    PW_ica = np.zeros_like(W_ica)
    PW_ica[col_index] = W_ica

    # 3. Fix the scaling/sign: divide each row by its diagonal so the diagonal
    #    is all ones (the SEM unit-self-coefficient convention) => W ~ I - B.
    D = np.diag(PW_ica)[:, np.newaxis]
    W_estimate = PW_ica / D
    B_estimate = np.eye(len(W_estimate)) - W_estimate    # B = I - W

    # 4. Find a causal order that makes B strictly lower triangular.
    def _search_causal_order(matrix):
        # exact case: repeatedly peel off an all-zero row (no remaining parents)
        causal_order = []
        row_num = matrix.shape[0]
        original_index = np.arange(row_num)
        while 0 < len(matrix):
            row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break                                    # cyclic => not a DAG
            target_index = row_index_list[0]
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)
            mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
            matrix = matrix[mask][:, mask]
        if len(causal_order) != row_num:
            return None
        return causal_order

    def _estimate_causal_order(matrix):
        # finite data: start by zeroing the m(m+1)/2 smallest |entries| (the
        # structural zeros of a strict-lower-triangular matrix), then zero one
        # additional entry at a time and test for a valid peel-off order.
        matrix = matrix.copy()
        pos_list = np.argsort(np.abs(matrix), axis=None)
        pos_list = np.vstack(np.unravel_index(pos_list, matrix.shape)).T
        initial_zero_num = int(matrix.shape[0] * (matrix.shape[0] + 1) / 2)
        for i, j in pos_list[:initial_zero_num]:
            matrix[i, j] = 0
        causal_order = None
        for i, j in pos_list[initial_zero_num:]:
            matrix[i, j] = 0
            causal_order = _search_causal_order(matrix)
            if causal_order is not None:
                break
        return causal_order

    causal_order = _estimate_causal_order(B_estimate)

    # 5. Prune + re-estimate signed coefficients along the order by adaptive
    #    lasso: each variable regressed on its predecessors, weak edges zeroed.
    def _predict_adaptive_lasso(data, predictors, target, gamma=1.0):
        lr = LinearRegression()
        lr.fit(data[:, predictors], data[:, target])
        weight = np.power(np.abs(lr.coef_), gamma)
        reg = LassoLarsIC(criterion="bic")
        reg.fit(data[:, predictors] * weight, data[:, target])
        return reg.coef_ * weight

    B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]                    # only earlier vars
        B[target, predictors] = _predict_adaptive_lasso(X, predictors, target)
    return B                                              # B[i,j]!=0 : j -> i
```

The causal chain, start to finish: I wanted full directed structure plus coefficients from observation alone, and the covariance — the only thing a Gaussian model exposes — provably can't tell `x_1 -> x_2` from `x_1 <- x_2`, as the matched two-variable example shows, so any second-order method stalls at the Markov equivalence class. Dropping Gaussianity is the lever: rewriting `x = Bx + e` as `x = (I - B)^{-1} e = A e` reveals the exact linear ICA model with `W = A^{-1} = I - B`, and ICA's non-Gaussian identifiability (no rotational ambiguity, unlike the Gaussian case that caused the symmetry) recovers `A` up to permutation, scaling, and sign. Those three indeterminacies are all that stand between ICA's output and `B`: the scaling/sign I kill by normalizing the diagonal of the correctly-ordered `W` to ones (the unit-self-coefficient convention), and the permutation I pin down because acyclicity makes the correct `W = I - B` lower-triangular with a nonzero diagonal — uniquely the row order whose diagonal is zero-free, which on finite data becomes "maximize the diagonal magnitudes," a heavy `1/|W_ii|` penalty that I showed is the maximum-likelihood objective and that is exactly a linear assignment problem solvable in `O(m^3)`. From the normalized `W` I read `B = I - W`; I extract a causal order by zeroing the `m(m+1)/2` smallest entries (the count of structural zeros in a strict-lower-triangular matrix) and peeling off all-zero rows, growing the zero set until the peel succeeds; and I get the final sparse, signed edges by regressing each variable on its causal predecessors with adaptive lasso. The one fragile joint is the ICA optimization itself, which is non-convex and can settle in a local optimum — the price of routing causal discovery through non-Gaussianity.
