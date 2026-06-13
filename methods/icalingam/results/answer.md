# ICA-LiNGAM, distilled

ICA-LiNGAM recovers the full directed weighted DAG of a Linear Non-Gaussian Acyclic Model
(LiNGAM) from purely observational continuous data, with no time order or variable ordering
given. Its key move is to recognize that a linear acyclic structural model with independent
non-Gaussian noise is *exactly* a linear independent component analysis (ICA) model, and that
non-Gaussianity makes the model fully identifiable — direction and coefficients — where the
Gaussian / covariance-only methods (PC, GES, Gaussian SEM) recover only the Markov equivalence
class.

## Problem it solves

Given a recursive linear model `x = Bx + e` with mutually independent, non-Gaussian disturbances
`e_i`, no latent confounders, and `B` permutable to strict lower triangularity (acyclicity),
recover `B` — the directed edges and their signed strengths — from a sample of `x` alone. With
Gaussian noise this is impossible beyond the equivalence class: `x_1=e_1, x_2=0.8x_1+e_2` with
`var(e_1)=1, var(e_2)=0.36` gives `var(x_1)=var(x_2)=1` and `cov(x_1,x_2)=0.8`; the reverse
`x_1=0.8x_2+e_1, x_2=e_2` with `var(e_2)=1, var(e_1)=0.36` gives the same values, so no
covariance-based method can orient the edge.

## Key idea

Solve `x = Bx + e` for `x`: `x = (I - B)^{-1} e = A e` with `A = (I - B)^{-1}`. Independent
non-Gaussian `e` plus invertible `A` is the linear ICA model, with separating matrix
`W = A^{-1} = I - B`. By ICA identifiability (Comon 1994), an invertible mixing matrix with
independent sources and at most one Gaussian source is identifiable up to **permutation, scaling,
and sign** — and, crucially, with *no rotational ambiguity*, unlike the Gaussian case whose
rotational freedom is exactly what hides causal direction. In this model every disturbance is
non-Gaussian, so the ICA condition is satisfied. The method
estimates `W` by ICA and then resolves the three indeterminacies to read off `B`:

- **Permutation.** The correct `W = I - B` is permutable to lower triangular with an all-ones
  (hence nonzero) diagonal. Among all row permutations of the ICA output, exactly one yields a
  zero-free diagonal, and it is the correct one (proof below). On finite data there are no exact
  zeros, so pick the permutation that makes diagonal magnitudes large by minimizing
  `sum_i 1/|W_ii|` — a **linear assignment problem** (Hungarian / `linear_sum_assignment`,
  `O(m^3)`), and the maximum-likelihood objective for the correspondence under a simple
  generalized-Gaussian disturbance model.
- **Scaling/sign.** Divide each row of the permuted `W` by its diagonal entry, forcing a unit
  diagonal (the SEM convention that each variable's own coefficient is 1). Then `B = I - W`.
- **Causal order.** Permute rows and columns of `B` to strict lower triangularity: start by
  zeroing the `m(m+1)/2` smallest entries (the number of structural zeros in a
  strict-lower-triangular `m x m` matrix), then zero one additional entry at a time and run the
  all-zero-row peel test until a topological order is found.
- **Prune/estimate edges.** Re-estimate signed coefficients along the order by regressing each
  variable on its predecessors with adaptive lasso (consistent selection drives absent edges to
  exactly zero).

## Identifiability of the row permutation (the load-bearing lemma)

Write the aligned `W = P_d M P_d^T` with `M` lower triangular, nonzero diagonal, `P_d` the causal
order. ICA returns rows scrambled: `W_ica = P_1 M P_2^T` with `P_1 = P_ica P_d`, `P_2 = P_d`. Let
`K` be the all-ones lower-triangular matrix (the densest possible support pattern). The number of
potentially nonzero diagonal entries of `P_1 K P_2^T` is
`tr(P_1 K P_2^T) = tr(K P_2^T P_1)`, so only `Q = P_2^T P_1` matters. If `Q != I`, some column
`i` moves to position `j < i`; otherwise every moved column would move right and the finite set of
positions could not be filled. The diagonal entry in new column `j` comes from `K[j, i]`, which lies
above the lower-triangular support and is zero. Because `K` is the densest support, the same entry
is zero for any lower-triangular `M`. If `Q = I` (`P_1 = P_2`), the same permutation is applied to
rows and columns, so the diagonal of `P_1 M P_1^T` is just the nonzero diagonal of `M` reordered.
Hence a fully nonzero diagonal `<=>` `P_1 = P_2`, so the zero-free-diagonal row permutation is
unique and correct. (Without
acyclicity the alternative is real: the two-variable off-DAG reparameterization
`b_12'=1/b_21, b_21'=1/b_12, ...` with swapped/sign-flipped sources gives identical data — but it
requires infinite coefficients when the truth is a DAG, which is why the DAG constraint pins the
permutation.)

## ML justification of the assignment cost

Model disturbances by a generalized Gaussian `log p(e_it) = -|e_it|^alpha / beta + Z`. Because each
candidate row is later divided by its diagonal `W_ii`, the value entering the density is
`e_it / W_ii`, so the log-likelihood contribution is
`-sum_i (1/(beta |W_ii|^alpha)) sum_t |e_it|^alpha` plus constants. With identical component
densities and `alpha=1`, maximizing over the row correspondence becomes
`min_{perm} sum_i 1/|W_ii|` — the assignment cost, derived rather than assumed.

## Algorithm

```
1. ICA on X  ->  W_ica = P D (I - B)            # rows arbitrary order/scale/sign
2. col_index = argmin_perm sum_i 1/|W_ica[perm(i), i]|   (linear assignment)
   PW = permute rows of W_ica by col_index
3. W = PW / diag(PW)                            # unit diagonal => W ~ I - B
   B = I - W
4. causal_order: zero the m(m+1)/2 smallest |B_ij|; then zero one additional
                 entry at a time and peel all-zero rows until an order is found
5. for each variable in order, regress on its predecessors via adaptive lasso -> signed B_ij
return B   # B[i, j] != 0 means edge j -> i
```

## Caveat

The ICA step maximizes a non-convex non-Gaussianity contrast (negentropy), so FastICA and
gradient-based ICA can converge to local optima depending on initialization and step size; the
permutation/scaling logic is exact given a good `W`, but the estimate is only as reliable as the
ICA run. Fixing the seed and giving a generous iteration budget mitigates but does not remove
this.

## Working code

A compact implementation of the estimator as a single `run_causal_discovery(X)` function:

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
