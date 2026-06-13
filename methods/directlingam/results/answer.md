# DirectLiNGAM, distilled

DirectLiNGAM recovers the full directed DAG of a linear non-Gaussian acyclic model (LiNGAM)
from purely observational data, **directly** — without ICA's non-convex iterative search. It
estimates the causal order one variable at a time by repeatedly finding the exogenous variable
(the source with no parents) and regressing it out of the rest, then reads off the connection
strengths by a triangular regression along the recovered order. It uses no
initial guess, step size, or convergence criterion, and it halts after fixed peeling rounds. With
an exact independence measure, the ordering argument is population-exact; the `pwling` path
below uses a one-dimensional entropy approximation to make that comparison cheap.

## Problem it solves

Estimate the adjacency matrix `B` (structure + strengths) of `x = Bx + e`, with `B` permutable
to strictly lower triangular and the disturbances `e_i` mutually independent, non-Gaussian,
nonzero-variance, no latent confounders — from a sample of `x` alone. Covariance-based methods
(Gaussian SEM, PC, GES) bottom out at the Markov equivalence class and cannot direct the
two-variable edge; non-Gaussianity is what makes the directions identifiable.

## Key idea

1. **Exogeneity ⇔ independence of residuals (Lemma 1).** A variable `x_j` is exogenous
   (`x_j = e_j`, no parents) **iff** it is independent of every least-squares residual
   `r_i^{(j)} = x_i - (cov(x_i,x_j)/var(x_j)) x_j`, `i != j`. Forward: if exogenous, the
   regression coefficient equals the mixing coefficient `a_{ij}` and the residual is the bundle
   of other independent sources, hence independent of `x_j = e_j`. Converse: if `x_j` has a
   parent, some residual and `x_j` share the non-Gaussian source `e_j` with nonzero
   coefficients in both, so by the Darmois–Skitovitch theorem they are **dependent**.
   Non-Gaussianity is essential to the converse.
2. **Residuals are again a LiNGAM (Lemma 2), with the same order (Corollary 3).** Removing the
   exogenous variable's effect by LS zeros its column of `A` and leaves a lower-triangular,
   unit-diagonal mixing matrix for the residual vector, and the relative causal order of the
   remaining variables is unchanged. So: find source → append to order → regress it out →
   recurse. After `p-1` rounds the order is complete; append the last variable.
3. **Strengths by triangular regression.** With the order `K` known, estimate `B` by regressing
   each variable on its predecessors in `K`; the code below uses adaptive lasso for this stage.
4. **Independence, not correlation.** LS residuals are *always* uncorrelated with the regressor,
   so uncorrelatedness is useless as the exogeneity test; a genuine independence measure is
   required. Use mutual information.

## The independence statistic

Pick the variable minimizing the total dependence with its residuals,
`T(x_j) = sum_{i != j} I(x_j, r_i^{(j)})`. Two estimators:

- **kernel:** nonparametric MI from Gaussian-kernel Gram-matrix determinants
  (Bach & Jordan 2002), with incomplete-Cholesky low-rank approximation; `(kappa, sigma) =
  (2e-3, 0.5)` for `n > 1000`, `(2e-2, 1.0)` otherwise.
- **pwling:** the pairwise likelihood-ratio /
  entropy-difference measure. For standardized `x, y`, the LR between `x->y` and `y->x`
  (`R > 0` favors `x->y`) is asymptotically, with `G = log p` and sample averages converging to
  `-H`, `R -> -H(x) - H(d̂/σ_d) + H(y) + H(ê/σ_e)` (`d̂ = y - ρx`, `ê = x - ρy`). The
  mutual-information difference is the same comparison with opposite orientation,
  `I(x, d) - I(y, e) = H(x) + H(d̂/σ_d) - H(y) - H(ê/σ_e) = -R`, equivalently
  `R = I(y, e) - I(x, d)`. Thus `x->y`, i.e. `R > 0`, is the direction with the smaller
  regressor-residual MI. The joint entropies cancel under the unit-determinant map
  `(x,y) -> (x, residual)`, and the residual-variance terms cancel because
  `σ_d = σ_e = sqrt(1 - ρ^2)`, so only **one-dimensional** entropies are needed. Differential
  entropy of a standardized variable uses the maximum-entropy approximation:

  ```
  Ĥ(u) = H(ν) - k1 [E{log cosh u} - γ]^2 - k2 [E{u exp(-u^2/2)}]^2,
  H(ν) = (1 + log 2π)/2,  k1 ≈ 79.047,  k2 ≈ 7.4129,  γ ≈ 0.37457,
  ```

  two non-Gaussianity penalties — log-cosh (heavy-tail/sparsity) and `u exp(-u^2/2)`
  (asymmetry/skew); `γ = E{log cosh}` under a standard Gaussian, so the first term vanishes for
  a Gaussian. A first-order expansion gives the nonlinear correlation `R̃ = ρ E{x tanh(y) -
  tanh(x) y}`, and a cumulant analysis shows `R̃ ∝ kurt(x)(ρ^2 - ρ^4)` under `x->y` — the
  criterion reads exactly the third/fourth-order structure covariance discards.

Per candidate `i`, aggregate only the evidence *against* `i` being the source:
`M(i) = sum_{j != i} min(0, diff_MI(i, j))^2`, with
`diff_MI(i, j) = [H(x_j) + H(r_i^{(j)})] - [H(x_i) + H(r_j^{(i)})]` on standardized variables
and residuals (`r_i^{(j)}` = `x_i` regressed on `x_j`). By the joint-entropy cancellation this
is `I(x_j, r_i^{(j)}) - I(x_i, r_j^{(i)})`, the likelihood-ratio sign for `i -> j`, which is
`> 0` exactly when `i`-as-cause is the more plausible direction; so `min(0, diff_MI(i, j))^2`
keeps only the negative votes (pairs where
`j` looks upstream of `i`). A true source collects ~0 counter-evidence; select `argmin_i M(i)`.

## Why this design

- **Direct vs ICA:** no non-convex optimization, no local minima, no init/step/stop knobs;
  fixed `p-1` peeling rounds plus the final remaining variable. The population-exact part is the
  residual-independence characterization and recursion; the kernel score gives the original
  exact-independence route with cost `O(n p^3 M^2 + p^4 M^3)`, while `pwling` is the cheaper
  entropy/LR approximation used in the code below.
- **Scale-invariant:** standardizing each variable and residual makes the order independent of
  variable scales — fixing ICA-LiNGAM's permutation steps, which can reverse under
  unit-variance normalization.
- **Fixed contrasts (pwling):** ICA theory says the recovered direction is insensitive to the
  exact log-pdf as long as its shape is roughly right; a fixed log-cosh + skew contrast needs
  no kernel/bandwidth and is robust at small samples.

## Algorithm

```
Input: data matrix X (p variables), index set U = {1..p}; K = []  (causal order)
repeat p-1 times:
    m = variable in U with least dependence/counter-evidence
    K.append(m)
    for i in U, i != m:  x_i <- residual(x_i on x_m)   # peel out source; residuals are a LiNGAM
    U <- U \ {m}
K.append(last remaining variable)
Construct strictly-lower-triangular B along K; estimate b_{ij} by regression (LS / adaptive lasso)
Output: B   with B[i, j] != 0 meaning x_j -> x_i
```

## Working code (`pwling`)

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

The compact implementation above follows the `pwling` path. The same outer loop can use
the kernel MI score instead, and prior-knowledge constraints can narrow the candidate set before
each ordering step. The adjacency stage here uses adaptive lasso to prune redundant predecessors
after the order is known.
