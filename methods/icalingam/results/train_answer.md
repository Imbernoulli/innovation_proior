We are given observational continuous vectors x = (x_1, ..., x_m) and asked to recover the directed weighted causal graph that generated them, including both the edge directions and their signed strengths. The assumed data-generating process is a linear structural equation model x = Bx + e that is acyclic, so B can be permuted to strict lower triangular form, with mutually independent noise terms e_i and no hidden confounders. The difficulty is direction: without experiments, time order, or a known variable ordering, many orientations of the same skeleton can produce the same observed distribution. In the two-variable Gaussian case, for example, x_1 = e_1, x_2 = 0.8x_1 + e_2 with variances 1 and 0.36 yields the same covariance as the reversed model x_1 = 0.8x_2 + e_1, x_2 = e_2. Because a multivariate Gaussian is fully described by its mean and covariance, any method that reads only second-order statistics, including PC, GES, and Gaussian SEM, can recover at best a Markov equivalence class with some edges left undirected and coefficients unidentified.

The lever that breaks this symmetry is non-Gaussianity. When the disturbances e_i are independent and non-Gaussian, the structural model becomes identifiable in full. The key observation is that solving x = Bx + e for x gives x = (I - B)^{-1}e = A e, which is exactly the linear independent component analysis (ICA) model: the observed vector is an invertible linear mixture of independent non-Gaussian sources. The separating matrix is W = A^{-1} = I - B. ICA identifiability tells us that A is recoverable up to permutation, scaling, and sign of its columns, with no rotational ambiguity, unlike the Gaussian case where any orthogonal rotation leaves the distribution unchanged. The causal problem is therefore reduced to estimating W by ICA and then resolving those three indeterminacies.

The method is ICA-LiNGAM (ICA-based Linear Non-Gaussian Acyclic Model). It first runs FastICA on the data to obtain an estimate W_ica of the separating matrix. Because ICA returns the rows in arbitrary order with arbitrary scale and sign, the next step is to find the correct row permutation. The true W = I - B is lower triangular with a nonzero diagonal under the causal ordering. It can be shown that among all row permutations of a permutable lower-triangular matrix with nonzero diagonal, only the correct paired row-and-column permutation yields a fully nonzero diagonal. On finite data there are no exact zeros, so ICA-LiNGAM chooses the permutation that places large-magnitude entries on the diagonal by minimizing sum_i 1/|W_ii|. This objective arises naturally from a maximum-likelihood argument under a generalized-Gaussian disturbance model, and it is exactly a linear assignment problem solvable in O(m^3) by the Hungarian algorithm.

Once the rows are permuted, scaling and sign are fixed by dividing each row by its diagonal entry, enforcing the SEM convention that each variable's own coefficient is one. This gives W with unit diagonal, and B is obtained as I - W. The causal order is then extracted by zeroing the m(m+1)/2 smallest absolute entries of B, the number of structural zeros in a strict-lower-triangular matrix, and repeatedly testing whether the remaining zero pattern can be peeled by removing all-zero rows. Finally, the edge coefficients are re-estimated along the discovered order by regressing each variable only on its predecessors and using adaptive lasso for consistent variable selection; this drives absent edges to exactly zero while producing accurate signed coefficients for the true parents.

The main caveat is the ICA optimization itself: maximizing a non-Gaussianity contrast such as negentropy is non-convex, so FastICA can converge to a local optimum depending on initialization. The permutation, scaling, and order-recovery logic is exact given a correct W, but the overall pipeline is only as reliable as the ICA estimate. Fixing the random seed and allowing a generous iteration budget improves stability but does not remove the non-convexity.

```python
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.linear_model import LinearRegression, LassoLarsIC
from sklearn.utils import check_array


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """ICA-LiNGAM: recover a directed weighted DAG from observational data.

    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables);
            B[i, j] != 0 means the directed edge j -> i.
    """
    X = check_array(X)
    seed = int(os.environ.get("SEED", "42"))
    n_vars = X.shape[1]

    # 1. Estimate the ICA separating matrix W_ica = P D (I - B).
    ica = FastICA(max_iter=1000, random_state=seed)
    ica.fit(X)
    W_ica = ica.components_

    # 2. Resolve row permutation via linear assignment: prefer large diagonals.
    _, col_index = linear_sum_assignment(1.0 / np.abs(W_ica))
    PW_ica = np.zeros_like(W_ica)
    PW_ica[col_index] = W_ica

    # 3. Resolve scaling/sign: unit diagonal => W ~ I - B, so B = I - W.
    D = np.diag(PW_ica)[:, np.newaxis]
    W_estimate = PW_ica / D
    B_estimate = np.eye(n_vars) - W_estimate

    # 4. Extract a causal order by peeling all-zero rows from a thresholded matrix.
    def _search_causal_order(matrix):
        causal_order = []
        original_index = np.arange(matrix.shape[0])
        while len(matrix) > 0:
            zero_rows = np.where(np.sum(np.abs(matrix), axis=1) == 0)[0]
            if len(zero_rows) == 0:
                return None
            target = zero_rows[0]
            causal_order.append(original_index[target])
            original_index = np.delete(original_index, target, axis=0)
            mask = np.delete(np.arange(len(matrix)), target, axis=0)
            matrix = matrix[mask][:, mask]
        return causal_order

    def _estimate_causal_order(matrix):
        matrix = matrix.copy()
        flat_order = np.argsort(np.abs(matrix), axis=None)
        positions = np.vstack(np.unravel_index(flat_order, matrix.shape)).T
        init_zeros = n_vars * (n_vars + 1) // 2
        for i, j in positions[:init_zeros]:
            matrix[i, j] = 0.0
        for i, j in positions[init_zeros:]:
            matrix[i, j] = 0.0
            order = _search_causal_order(matrix)
            if order is not None:
                return order
        return None

    causal_order = _estimate_causal_order(B_estimate)

    # 5. Re-estimate sparse coefficients along the order with adaptive lasso.
    def _adaptive_lasso(data, predictors, target, gamma=1.0):
        lr = LinearRegression().fit(data[:, predictors], data[:, target])
        weight = np.power(np.abs(lr.coef_), gamma)
        reg = LassoLarsIC(criterion="bic")
        reg.fit(data[:, predictors] * weight, data[:, target])
        return reg.coef_ * weight

    B = np.zeros((n_vars, n_vars), dtype=float)
    for i in range(1, len(causal_order)):
        target = causal_order[i]
        predictors = causal_order[:i]
        B[target, predictors] = _adaptive_lasso(X, predictors, target)
    return B
```
