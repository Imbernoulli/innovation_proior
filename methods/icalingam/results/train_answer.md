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
    """ICA-based LiNGAM.

    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables);
            B[i, j] != 0 means the directed edge j -> i.
    """
    X = check_array(X)
    seed = int(os.environ.get("SEED", "42"))

    # 1. x = A e is linear ICA; estimate W_ica = A^{-1} (rows scrambled in
    #    order, scale, sign). W (correctly aligned/scaled) equals I - B.
    ica = FastICA(max_iter=1000, random_state=seed)
    ica.fit(X)
    W_ica = ica.components_

    # 2. Undo the row permutation: minimize sum_i 1/|W_ii| (large entries on
    #    the diagonal) = linear assignment with cost C_ij = 1/|W_ica[i, j]|.
    _, col_index = linear_sum_assignment(1 / np.abs(W_ica))
    PW_ica = np.zeros_like(W_ica)
    PW_ica[col_index] = W_ica

    # 3. Fix scaling/sign: unit diagonal (SEM convention) => W ~ I - B, B = I - W.
    D = np.diag(PW_ica)[:, np.newaxis]
    W_estimate = PW_ica / D
    B_estimate = np.eye(len(W_estimate)) - W_estimate

    # 4. Causal order making B strictly lower triangular.
    def _search_causal_order(matrix):
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

    def _estimate_causal_order(matrix):
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

    # 5. Prune + re-estimate signed coefficients along the order (adaptive lasso).
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
        predictors = causal_order[:i]
        B[target, predictors] = _predict_adaptive_lasso(X, predictors, target)
    return B
```
