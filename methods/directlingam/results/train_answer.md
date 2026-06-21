We are given only observational samples of a vector x = (x_1, ..., x_p) and asked to recover the directed causal DAG and connection strengths behind it. The data-generating process is a linear acyclic structural equation model x = Bx + e, where B can be permuted to strict lower triangularity under some causal order and the disturbances e_i are mutually independent, non-Gaussian, and have nonzero variance. Covariance-based methods cannot solve this problem because the covariance matrix is direction-blind: for two correlated variables, the models x_1 -> x_2 and x_2 -> x_1 can produce exactly the same 2x2 covariance when the noise is Gaussian. PC and GES therefore return only a Markov equivalence class and cannot orient edges in many places, including the two-variable case. Even ICA-LiNGAM, which leverages non-Gaussianity through an ICA unmixing step, inherits the non-convex iterative search of ICA with its local optima, initialization sensitivity, step-size choices, and convergence criteria, and its permutation steps are scale-dependent so that normalizing variables to unit variance can change the recovered order.

The way forward is to keep the non-Gaussianity, which makes the full DAG identifiable in principle, but to recover the causal order directly without any global non-convex optimization. The proposed method is DirectLiNGAM. The key observation is that any acyclic DAG with no latent confounders must contain at least one exogenous variable x_j that equals its own independent non-Gaussian disturbance e_j. DirectLiNGAM finds this source, appends it to the causal order, regresses it out of all remaining variables, and recurses on the residuals. After p-1 such peeling rounds only one variable remains, which goes last. Once the order is known, the connection strengths are obtained by a sparse triangular regression of each variable on its predecessors.

The correctness of the source-detection step rests on a non-Gaussian independence characterization. For a candidate variable x_j, form the least-squares residuals r_i^{(j)} = x_i - (cov(x_i, x_j)/var(x_j)) x_j for every other variable i. If x_j is exogenous, the regression coefficient equals the mixing coefficient a_{ij}, and the residual becomes precisely the bundle of all other independent sources, so x_j is independent of every residual. Conversely, if x_j is not exogenous, it has at least one parent x_i with nonzero covariance, and the residual r_i^{(j)} shares the non-Gaussian source e_j with x_j with nonzero weight in both linear combinations; by the Darmois-Skitovitch theorem the two must be dependent. Thus x_j is exogenous if and only if it is independent of all its least-squares residuals. Peeling the exogenous variable out by least squares preserves the LiNGAM structure on the residuals with the same relative causal order, so the recursion is sound.

Because least squares always makes residuals uncorrelated with the regressor, the exogeneity test must measure genuine independence, not uncorrelatedness. DirectLiNGAM scores each candidate by the total dependence between it and its residuals. The practical implementation below uses the pairwise likelihood-ratio criterion, which compares the two possible directions for each pair (i, j). The likelihood ratio between i -> j and j -> i can be rewritten so that all joint entropies cancel under the unit-determinant linear map from (x_i, x_j) to (x_i, residual), leaving only one-dimensional differential entropies. These are estimated by a fixed maximum-entropy approximation using a log-cosh non-Gaussianity term and a skew term. For each candidate i, the method accumulates only the evidence against i being the source, summing the squared negative pairwise differences, and picks the candidate with the least counter-evidence. Standardizing every variable and every residual makes the ordering scale-invariant.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC
from sklearn.utils import check_array


def _predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    lr = LinearRegression()
    lr.fit(X[:, predictors], X[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X[:, predictors] * weight, X[:, target])
    return reg.coef_ * weight


def _residual(xi, xj):
    return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj


def _entropy(u):
    k1 = 79.047
    k2 = 7.4129
    gamma = 0.37457
    return (1 + np.log(2 * np.pi)) / 2 \
        - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 \
        - k2 * (np.mean(u * np.exp((-u ** 2) / 2))) ** 2


def _diff_mutual_info(xi_std, xj_std, ri_j, rj_i):
    return (_entropy(xj_std) + _entropy(ri_j / np.std(ri_j))) \
        - (_entropy(xi_std) + _entropy(rj_i / np.std(rj_i)))


def _search_causal_order(X, U):
    if len(U) == 1:
        return int(U[0])
    M_list = []
    for i in U:
        M = 0.0
        for j in U:
            if i == j:
                continue
            xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
            xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            ri_j = _residual(xi_std, xj_std)
            rj_i = _residual(xj_std, xi_std)
            M += np.min([0.0, _diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
        M_list.append(-1.0 * M)
    return int(U[np.argmax(M_list)])


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """X: (n_samples, n_variables) -> adjacency B with B[i, j] != 0 meaning j -> i."""
    X = check_array(X)
    n_features = X.shape[1]
    U = np.arange(n_features)
    K = []
    X_work = np.copy(X)
    for _ in range(n_features):
        m = _search_causal_order(X_work, U)
        for i in U:
            if i != m:
                X_work[:, i] = _residual(X_work[:, i], X_work[:, m])
        K.append(int(m))
        U = U[U != m]

    B = np.zeros((n_features, n_features), dtype=float)
    for rank in range(1, len(K)):
        target = K[rank]
        predictors = K[:rank]
        B[target, predictors] = _predict_adaptive_lasso(X, predictors, target)
    return B
```
