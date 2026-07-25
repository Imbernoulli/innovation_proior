We are given only observational samples of a vector x = (x_1, ..., x_p) and asked to recover the directed causal DAG and connection strengths behind it. The data-generating process is a linear acyclic structural equation model x = Bx + e, where B can be permuted to strict lower triangularity under some causal order and the disturbances e_i are mutually independent, non-Gaussian, and have nonzero variance. Covariance-based methods cannot solve this problem because the covariance matrix is direction-blind: for two correlated variables, the models x_1 -> x_2 and x_2 -> x_1 can produce exactly the same 2x2 covariance when the noise is Gaussian. PC and GES therefore return only a Markov equivalence class and cannot orient edges in many places, including the two-variable case. Even ICA-LiNGAM, which leverages non-Gaussianity through an ICA unmixing step, inherits the non-convex iterative search of ICA with its local optima, initialization sensitivity, step-size choices, and convergence criteria, and its permutation steps are scale-dependent so that normalizing variables to unit variance can change the recovered order.

The way forward is to keep the non-Gaussianity, which makes the full DAG identifiable in principle, but to recover the causal order directly without any global non-convex optimization. The proposed method is DirectLiNGAM. The key observation is that any acyclic DAG with no latent confounders must contain at least one exogenous variable x_j that equals its own independent non-Gaussian disturbance e_j. DirectLiNGAM finds this source, appends it to the causal order, regresses it out of all remaining variables, and recurses on the residuals. After p-1 such peeling rounds only one variable remains, which goes last. Once the order is known, the connection strengths are obtained by a sparse triangular regression of each variable on its predecessors.

The correctness of the source-detection step rests on a non-Gaussian independence characterization. For a candidate variable x_j, form the least-squares residuals r_i^{(j)} = x_i - (cov(x_i, x_j)/var(x_j)) x_j for every other variable i. If x_j is exogenous, the regression coefficient equals the mixing coefficient a_{ij}, and the residual becomes precisely the bundle of all other independent sources, so x_j is independent of every residual. Conversely, if x_j is not exogenous, it has at least one parent x_i with nonzero covariance, and the residual r_i^{(j)} shares the non-Gaussian source e_j with x_j with nonzero weight in both linear combinations; by the Darmois-Skitovitch theorem the two must be dependent. Thus x_j is exogenous if and only if it is independent of all its least-squares residuals. Peeling the exogenous variable out by least squares preserves the LiNGAM structure on the residuals with the same relative causal order, so the recursion is sound.

Because least squares always makes residuals uncorrelated with the regressor, the exogeneity test must measure genuine independence, not uncorrelatedness. DirectLiNGAM scores each candidate by the total dependence between it and its residuals. The class below, `DirectLiNGAM`, uses the pairwise likelihood-ratio criterion, which compares the two possible directions for each pair (i, j). The likelihood ratio between i -> j and j -> i can be rewritten so that all joint entropies cancel under the unit-determinant linear map from (x_i, x_j) to (x_i, residual), leaving only one-dimensional differential entropies. These are estimated by a fixed maximum-entropy approximation using a log-cosh non-Gaussianity term and a skew term. For each candidate i, the method accumulates only the evidence against i being the source, summing the squared negative pairwise differences, and picks the candidate with the least counter-evidence. Standardizing every variable and every residual makes the ordering scale-invariant. `fit` drives the p-1 peeling rounds and calls `_search_causal_order` and `_residual` at each step; once the order `causal_order_` is fixed, `_estimate_adjacency_matrix` fills in the strengths via `_predict_adaptive_lasso`, and `adjacency_matrix_` exposes the result.

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


class DirectLiNGAM:
    """DirectLiNGAM with the pairwise-likelihood-ratio ('pwling') independence
    measure. Estimates a causal order by repeatedly finding the exogenous
    variable and regressing it out, then the adjacency matrix along that order.
    Convention: adjacency_matrix_[i, j] != 0 means x_j -> x_i."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        self._causal_order = None
        self._adjacency_matrix = None

    # --- independence machinery (pwling) ---
    def _residual(self, xi, xj):
        """xi regressed on xj; residual is uncorrelated with xj by construction."""
        return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj

    def _entropy(self, u):
        """Maximum-entropy approximation of differential entropy of standardized u."""
        k1, k2, gamma = 79.047, 7.4129, 0.37457
        return (1 + np.log(2 * np.pi)) / 2 \
            - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 \
            - k2 * (np.mean(u * np.exp((-u ** 2) / 2))) ** 2

    def _diff_mutual_info(self, xi_std, xj_std, ri_j, rj_i):
        """Entropy difference between the two causal directions for the pair.
        The bracket difference equals I(x_j, r_i^{(j)}) - I(x_i, r_j^{(i)})
        after the common joint entropy cancels. Residuals are standardized."""
        return (self._entropy(xj_std) + self._entropy(ri_j / np.std(ri_j))) \
            - (self._entropy(xi_std) + self._entropy(rj_i / np.std(rj_i)))

    def _search_causal_order(self, X, U):
        """Index in U most consistent with being exogenous (least evidence against)."""
        if len(U) == 1:
            return U[0]
        M_list = []
        for i in U:
            M = 0.0
            for j in U:
                if i == j:
                    continue
                xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
                xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
                ri_j = self._residual(xi_std, xj_std)
                rj_i = self._residual(xj_std, xi_std)
                M += np.min([0.0, self._diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)
        return U[np.argmax(M_list)]

    # --- driver ---
    def fit(self, X):
        X = check_array(X)
        n_features = X.shape[1]
        U = np.arange(n_features)
        K = []
        X_ = np.copy(X)
        for _ in range(n_features):
            m = self._search_causal_order(X_, U)
            for i in U:
                if i != m:
                    X_[:, i] = self._residual(X_[:, i], X_[:, m])
            K.append(int(m))
            U = U[U != m]
        self._causal_order = K
        self._estimate_adjacency_matrix(X)
        return self

    def _estimate_adjacency_matrix(self, X):
        """Strengths along the known order: adaptive lasso on predecessors."""
        B = np.zeros([X.shape[1], X.shape[1]], dtype="float64")
        for i in range(1, len(self._causal_order)):
            target = self._causal_order[i]
            predictors = self._causal_order[:i]
            B[target, predictors] = _predict_adaptive_lasso(X, predictors, target)
        self._adjacency_matrix = B
        return self

    @property
    def causal_order_(self):
        return self._causal_order

    @property
    def adjacency_matrix_(self):
        return self._adjacency_matrix


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """X: (n_samples, n_variables) -> adjacency B, B[i, j] != 0 means j -> i."""
    return DirectLiNGAM().fit(X).adjacency_matrix_
```
