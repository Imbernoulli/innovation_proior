**Problem (from step 1).** The continuous-optimization rung refuses to use non-Gaussianity, so it reads
only second-order structure and bottoms out at the Markov equivalence class — visible as missed,
unoriented hub edges on SF100 (recall 0.597) and one-seed instability on the dense ER30. The fix is to
switch the engine to one whose entire mechanism *is* the non-Gaussian fingerprint.

**Key idea.** Solve `x = Bx + e` for `x`: `x = (I − B)^{-1} e = A e`. Independent non-Gaussian `e` plus
invertible `A` is *exactly* the linear ICA model, with separating matrix `W = A^{-1} = I − B`. By ICA
identifiability (at most one Gaussian source ⇒ identifiable up to permutation, scaling, sign, with **no**
rotational ambiguity — unlike the Gaussian case whose rotational freedom hides direction), the full DAG
is recoverable. Estimate `W` by ICA, then resolve the three indeterminacies.

**Why it works.** The non-Gaussianity collapses the rotational ambiguity that blinds covariance-only
methods. The three indeterminacies are killed structurally: the **permutation** because the correct
`W = I − B` is the unique row order with a zero-free diagonal — on finite data, minimize
`Σᵢ 1/|Wᵢᵢ|`, a linear assignment problem (`linear_sum_assignment`, `O(d³)`); the **scaling/sign** by
dividing each row by its diagonal so `W` has a unit diagonal, giving `B = I − W`; the **causal order** by
zeroing the `d(d+1)/2` smallest entries (the structural-zero count of a strict-lower-triangular matrix)
then peeling all-zero rows; and the **edge estimates** by `_BaseLiNGAM._estimate_adjacency_matrix`, which
regresses each variable on its predecessors via adaptive lasso (BIC-selected, consistent selection).

**Hyperparameters / caveat.** `FastICA(max_iter=1000, random_state=SEED)`. The ICA step is non-convex
(negentropy contrast), so it can settle in a local optimum depending on initialization — the soft spot;
the permutation/scaling logic is exact given a good `W`. A fixed seed and generous budget mitigate but do
not remove this.

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
