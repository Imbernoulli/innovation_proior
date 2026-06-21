The continuous-optimization rung told me in numbers exactly what it costs to ignore non-Gaussianity. NOTEARS-linear was strong on the easy graphs but unstable on the dense ER30 (F1 0.919, SHD 16, with seed 456 collapsing to F1 0.835 / SHD 35), and it bottomed out on the hub-heavy SF100: F1 0.716, SHD 136, and a glaring precision/recall asymmetry — precision 0.897 but recall only **0.597**. That asymmetry is the diagnosis. On the scale-free graph the least-squares program is *missing* roughly forty percent of the true edges while keeping the ones it does pick mostly right — the signature of a method that cannot orient edges into and out of high-degree hubs from second-order structure alone, so it threshold-prunes ambiguous hub edges rather than risk a reversal. This is not an estimator weakness I can optimize away: a least-squares score reads only the covariance, and under linear-Gaussian assumptions the covariance determines the joint law completely, so two reversed models — $x_1=e_1,\ x_2=0.8x_1+e_2$ versus $x_1=0.8x_2+e_1,\ x_2=e_2$ — can share identical variances and covariance and hence the *same density*. Anything reading only second-order structure recovers at best the Markov equivalence class, and the missing-edge recall on SF100 is the visible cost of being stuck there. The covariance is a dead end *by assumption*, and the assumption is Gaussianity. The fix is not a better optimizer; it is to switch the engine to one whose entire mechanism is the non-Gaussian fingerprint.

I propose **ICA-LiNGAM**. The move that unlocks it is to write the recursive linear model $x = Bx + e$ the other way: $(I-B)x = e$, so $x = (I-B)^{-1}e = A\,e$ with $A=(I-B)^{-1}$. Read literally, the observed vector $x$ is a fixed invertible linear mixture $A$ of the disturbance vector $e$, whose components are — by the model's own assumption — mutually independent and (here) non-Gaussian. That is *exactly* the linear independent component analysis model, not a metaphor but the same equations: the disturbances $e_i$ are the independent components, $A$ is the mixing matrix, and $W = A^{-1} = I-B$ is the separating matrix. And ICA carries precisely the identifiability the Gaussian world denied us: when the sources are independent and at most one is Gaussian, the mixing matrix is identifiable up to permutation, scaling, and sign of its columns, with **no rotational ambiguity**. Contrast the Gaussian case, where any orthogonal rotation $AR$ of independent Gaussians produces the same distribution — that rotational freedom is exactly what left the previous rung unable to orient. Since every disturbance here is non-Gaussian, the ambiguity collapses, and $A$, hence $W=I-B$, hence $B$, is recoverable. So I estimate $W$ by ICA — a fixed random seed and a generous 1000-iteration FastICA budget — and then spend the rest of the method *resolving the three indeterminacies*, because that is where the causal content lives.

ICA returns some $W_{\text{ica}}$ that, in the population limit, equals $PDW$: the true separating matrix with rows permuted by an unknown permutation $P$ and each row scaled and sign-flipped by an unknown diagonal $D$. In a typical ICA application nobody cares about order, scale, or sign of recovered sources, but here the rows of $W$ correspond to specific disturbances and the columns to specific observed variables, so I must *undo* the permutation and *fix* the scaling. Scaling is the easy one and it tells me what "correct" means: in the SEM convention each variable's own coefficient is one, so $W=I-B$ has an all-ones diagonal ($W_{ii}=1-b_{ii}=1$) — a fixed, known anchor. If the rows were in the right order, dividing each row by its own diagonal entry pins the scale *and* the sign at once. So scaling is solved the instant I know the row order, and all the weight falls onto the permutation. The permutation is decidable because of the DAG structure: since $B$ is permutable to strict lower triangular, the correctly ordered $W=I-B$ is lower triangular with a *nonzero* diagonal (the ones from $I$). Among all ways to permute the rows back, exactly one yields a fully nonzero diagonal, and it is the correct one. The proof is clean — writing $W_{\text{ica}}=P_1 M P_2^\top$ with $M$ lower-triangular and $K$ the all-ones lower-triangular densest support, $\mathrm{tr}(P_1 K P_2^\top)=\mathrm{tr}(K\,P_2^\top P_1)$ depends only on $Q=P_2^\top P_1$; if $Q\neq I$ some column moves strictly left, landing a diagonal slot above the lower-triangular support where the entry is structurally zero, while $Q=I$ means the same relabeling on rows and columns and a fully nonzero diagonal — so zero-free diagonal $\iff P_1=P_2$, uniquely correct.

Finite data degrades that exact criterion: ICA never returns exact zeros, so *every* row permutation gives a technically nonzero diagonal. I need a soft version that prefers the permutation whose diagonal entries are as far from zero as possible, since the correct alignment puts the structurally large unit-ish entries on the diagonal and the structurally small (true-zero) entries off it. The natural cost penalizes small diagonal magnitudes — $\sum_i 1/\lvert W_{ii}\rvert$ minimized — and this is not ad hoc: modeling each disturbance with a generalized-Gaussian density $\log p(e) = -\lvert e\rvert^\alpha/\beta + Z$ and noting that a candidate row is later divided by its diagonal, the log-likelihood becomes $-\sum_i (1/(\beta\lvert W_{ii}\rvert^\alpha))\sum_t\lvert e_{it}\rvert^\alpha$, which at $\alpha=1$ is exactly $\min_{\text{perm}}\sum_i 1/\lvert W_{ii}\rvert$. Searching all $d!$ permutations is hopeless, but the objective is a *linear assignment problem*: placing row $r$ at diagonal column $c$ contributes $C_{r,c}=1/\lvert W_{\text{ica}}[r,c]\rvert$, and minimizing the summed assignment cost is $O(d^3)$ by the Hungarian algorithm — cheap even at the 100 nodes of SF100. So I build $C_{r,c}=1/\lvert W_{\text{ica}}[r,c]\rvert$, run `linear_sum_assignment`, and scatter each row to its matched position so the matched entries land on the diagonal; then divide each row by its diagonal entry to get a unit diagonal, yielding $W_{\text{estimate}}\approx I-B$ in the right units and $B = I - W$.

Two steps remain. First, an explicit causal order: $B=I-W$ has no exact zeros on finite data, so no permutation makes the upper triangle exactly zero, but a strict-lower-triangular $d\times d$ matrix has $d(d+1)/2$ structural zeros (the diagonal plus the whole upper triangle), so I sort all $d^2$ entries by absolute value, set the $d(d+1)/2$ smallest to exact zero, then walk the remaining entries from smallest upward, zeroing one more at a time and after each running a *peel test* — find an all-zero row (a source with no remaining parents), record it, delete its row and column, and repeat; the first walk that peels all $d$ variables gives a valid topological order. Second, the actual edges, because $B=I-W$ is fully dense — the order pruned nothing. The statistically honest way is to *re-estimate* the edges by sparse regression along the discovered order, and the harness already supplies it: handing the order to `causal-learn`'s `_BaseLiNGAM._estimate_adjacency_matrix`, which for each target regresses it on its causal predecessors by **adaptive lasso** — OLS pilot coefficients, per-predictor weights $\lvert\text{coef}\rvert^{\gamma}$ with $\gamma=1$, a BIC-selected lasso on the reweighted predictors, then unweighting — giving consistent selection that keeps true edges and zeroes spurious weak ones. The first variable in the order has no predecessors and stays exogenous with an all-zero row. This is the same sparse re-estimation NOTEARS approximated with a single hard threshold $\omega=0.3$; doing it as a per-node BIC-selected lasso is strictly more careful, which is part of why I expect the missing-edge problem to shrink. The one honest soft spot: the ICA step is a non-convex optimization (maximizing a negentropy contrast via fixed-point iteration), so a poor initialization can land in the wrong basin, giving an unreliable $W$ and hence a wrong graph — there is no convexity guarantee, unlike the permutation/scaling logic, which is exact given a good $W$.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    import os
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import FastICA
    from sklearn.utils import check_array
    from causallearn.search.FCMBased.lingam.base import _BaseLiNGAM

    X = check_array(X)
    seed = int(os.environ.get("SEED", "42"))

    # Core ICALiNGAM steps (from causallearn.search.FCMBased.lingam.ica_lingam)
    ica = FastICA(max_iter=1000, random_state=seed)
    ica.fit(X)
    W_ica = ica.components_

    _, col_index = linear_sum_assignment(1 / np.abs(W_ica))
    PW_ica = np.zeros_like(W_ica)
    PW_ica[col_index] = W_ica

    D = np.diag(PW_ica)[:, np.newaxis]
    W_estimate = PW_ica / D
    B_estimate = np.eye(len(W_estimate)) - W_estimate

    def _search_causal_order(matrix: np.ndarray):
        causal_order = []
        row_num = matrix.shape[0]
        original_index = np.arange(row_num)
        while 0 < len(matrix):
            row_index_list = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
            if len(row_index_list) == 0:
                break
            target_index = row_index_list[0]
            causal_order.append(original_index[target_index])
            original_index = np.delete(original_index, target_index, axis=0)
            mask = np.delete(np.arange(len(matrix)), target_index, axis=0)
            matrix = matrix[mask][:, mask]
        if len(causal_order) != row_num:
            return None
        return causal_order

    def _estimate_causal_order(matrix: np.ndarray):
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

    class _LocalICALiNGAM(_BaseLiNGAM):
        def fit(self, X):
            return self

    model = _LocalICALiNGAM(random_state=seed)
    model._causal_order = causal_order
    model._estimate_adjacency_matrix(X)
    return model.adjacency_matrix_
```
