We want to build a predictor as an additive expansion of simple base learners, F(x) = sum_m beta_m h(x; a_m), where each h is a small regression tree. Such expansions are powerful because weak but interpretable trees can be combined into an accurate and robust model. The natural way to fit them is forward stagewise: add one term at a time while holding earlier terms fixed, solving (beta_m, a_m) = argmin sum_i L(y_i, F_{m-1}(x_i) + beta h(x_i; a)). Unfortunately that subproblem has a clean solution only for half squared error, where it reduces to fitting the current residual, and for exponential loss, where it gives AdaBoost. For absolute error, Huber loss, binomial deviance, and multiclass deviance there is no convenient closed form, and the few existing algorithms are one-off derivations rather than a single general mechanism.

Existing ideas leave a clear gap. Forward stagewise least squares is welded to squared error, which is non-robust to outliers and a poor classification criterion. AdaBoost reweights examples toward hard cases, but its exponential loss is brittle on noisy labels and it gives no recipe for regression or for arbitrary differentiable losses. Newton boosting such as LogitBoost derives special machinery for each loss and divides by estimated second derivatives that can vanish on confident points, producing numerical instability. A single bagged or boosted tree also does not sequentially descend a chosen loss. What is needed is one procedure that turns the stagewise subproblem for any smooth loss into the single operation a tree library already does well: a least-squares fit.

The method is gradient boosting. Instead of parameterizing the model and optimizing in parameter space, treat the function values at the training points as the variables and perform steepest descent in that space. At each round compute the negative gradient of the loss with respect to the current prediction at every training point, y_tilde_i = -dL(y_i, F(x_i))/dF(x_i) evaluated at F = F_{m-1}. This vector is the best local descent direction, but it is defined only at the data. To generalize it, fit a regression tree to these pseudo-responses by least squares, which chooses the tree whose values are most aligned with the negative gradient. The least-squares fit gives the direction; the actual step length is then set by a one-dimensional line search on the real loss, so the magnitude is honest even though the direction was chosen with a squared-error surrogate. For squared error the pseudo-response is exactly the residual, so the procedure reduces to the classic "fit the residuals" loop.

A tree with J terminal nodes is a sum of indicator functions over disjoint regions, so once the regions are fixed the update can be improved for free by giving each leaf its own constant. The global step rho_m is replaced by J independent one-dimensional optimizations, gamma_jm = argmin_gamma sum_{x_i in R_jm} L(y_i, F_{m-1}(x_i) + gamma), one per leaf. The least-squares fit now only discovers the partition, while the leaf-specific update sets the values on the true loss. For squared error the leaf value is the mean residual in the leaf. For binomial deviance the optimal constant is approximated by a single Newton step, sum(y - p)/sum(p(1-p)) where p = sigmoid(F). Putting the first-order pseudo-response into the tree fit and deferring curvature to the per-leaf sum avoids the instability of dividing by vanishing per-point p(1-p) in the regression target itself.

Regularization is provided jointly by the number of trees M and a shrinkage factor, or learning rate, v in (0, 1). Rather than taking the full leaf update each round, we add only v times it. Each tree is a noisy greedy estimate of the descent direction; small steps average many such estimates before the model moves far, which reduces variance and changes the path that later pseudo-responses see. M and v are chosen on held-out data, with smaller v requiring more trees. Tree size J controls the interaction order captured by the ensemble: a J-leaf tree can represent interactions of order at most J-1, so depth-3 trees are usually sufficient for real problems dominated by low-order interactions.

The implementation below follows the standard stagewise loop. It initializes the ensemble to the best constant under the loss, then repeatedly computes pseudo-residuals, fits a depth-limited regression tree to them by least squares, overwrites each leaf value with the loss-specific update, and accumulates the result shrunk by the learning rate. The task argument selects squared error for regression or binomial deviance for classification.

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _safe_divide(numerator, denominator):
    if abs(denominator) <= 1e-150:
        return 0.0
    return numerator / denominator


class GradientBoostingMachine:
    """Gradient boosting: fit trees to negative gradients, set leaves by the
    loss-specific update, and shrink each step by the learning rate.
    task="regression" uses squared error; task="classification" uses binomial deviance.
    """

    def __init__(self, task="regression", n_rounds=200, max_depth=3, learning_rate=0.1):
        self.task = task
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.trees = []
        self.init_ = 0.0

    def _init_prediction(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        p = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        return float(np.log(p / (1.0 - p)))

    def _pseudo_residuals(self, y, F):
        if self.task == "regression":
            return y - F
        return y - _sigmoid(F)

    def _leaf_value(self, resid_leaf, y_leaf):
        if self.task == "regression":
            return float(np.mean(resid_leaf))
        p = y_leaf - resid_leaf
        num = float(np.sum(resid_leaf))
        den = float(np.sum(p * (1.0 - p)))
        return _safe_divide(num, den)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.init_ = self._init_prediction(y)
        F = np.full(len(y), self.init_)
        for _ in range(self.n_rounds):
            resid = self._pseudo_residuals(y, F)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, criterion="friedman_mse")
            tree.fit(X, resid)
            leaves = tree.apply(X)
            update = np.zeros(len(y))
            for leaf in np.unique(leaves):
                idx = np.where(leaves == leaf)[0]
                gamma = self._leaf_value(resid[idx], y[idx])
                tree.tree_.value[leaf, 0, 0] = gamma
                update[idx] = gamma
            F += self.learning_rate * update
            self.trees.append(tree)
        return self

    def decision_function(self, X):
        F = np.full(X.shape[0], self.init_)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict(self, X):
        F = self.decision_function(X)
        if self.task == "regression":
            return F
        return (_sigmoid(F) >= 0.5).astype(int)

    def predict_proba(self, X):
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p, p])
```
