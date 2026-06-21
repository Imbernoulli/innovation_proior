# Support-Vector Machine From Maximum Margin

The method chooses the separating hyperplane with largest geometric margin, because VC theory bounds the capacity of margin-restricted hyperplanes by a radius-to-margin term rather than by the full feature dimension. Duality then makes the solution depend only on inner products, so any Mercer kernel can supply a nonlinear feature space without explicitly constructing it.

## Hard-Margin Form

For a separating hyperplane `D(x) = w . x + b`, normalize the scale so the closest points satisfy `y_i D(x_i) = 1`. The boundary-to-nearest-point margin is then `1 / ||w||`, while the full distance between the two supporting hyperplanes is `2 / ||w||`; maximizing either is equivalent to:

```text
minimize    1/2 ||w||^2
subject to  y_i (w . x_i + b) >= 1,  i = 1,...,l
```

The Lagrangian conditions give:

```text
w = sum_i alpha_i y_i x_i
sum_i alpha_i y_i = 0
alpha_i >= 0
```

Substitution yields the dual:

```text
maximize    sum_i alpha_i
            - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)
subject to  sum_i alpha_i y_i = 0
            alpha_i >= 0
```

Points with `alpha_i > 0` lie on the supporting boundary in the hard-margin case; points strictly outside the margin have `alpha_i = 0`. In degenerate cases a boundary point can have zero coefficient, so the sparse expansion uses the nonzero-coefficient support vectors:

```text
f(x) = sign(sum_i alpha_i y_i K(x_i, x) + b)
```

where the sum is over support vectors.

## Soft-Margin Form

For overlapping or noisy data, introduce `xi_i >= 0`:

```text
minimize    1/2 ||w||^2 + C sum_i xi_i
subject to  y_i (w . x_i + b) >= 1 - xi_i
            xi_i >= 0
```

The dual objective is unchanged, and the slack penalty becomes the box constraint:

```text
maximize    sum_i alpha_i
            - 1/2 sum_i sum_j alpha_i alpha_j y_i y_j K(x_i, x_j)
subject to  sum_i alpha_i y_i = 0
            0 <= alpha_i <= C
```

`C` controls the tradeoff between a wide margin and violations. If `0 < alpha_i < C`, the point has `xi_i = 0`, lies exactly on the margin, and can be used to compute `b`. If `alpha_i = C`, the point may be on the margin, inside the margin, or misclassified, but its influence is capped. If `alpha_i = 0`, the point is outside or on the margin and does not appear in the decision expansion.

## Kernel Condition

`K` must be an inner product in some feature space. Mercer-positive choices include:

```text
linear:      K(u, v) = u . v
polynomial:  K(u, v) = (gamma u . v + coef0)^d
radial:      K(u, v) = exp(-gamma ||u - v||^2), gamma = 1 / (2 sigma^2)
```

The training problem scales with the number of examples, and prediction scales with the number of support vectors, not with the explicit feature dimension.

## Classifier Skeleton

```python
import numpy as np

def linear_kernel(X, Y=None):
    Y = X if Y is None else Y
    return X @ Y.T

def _default_gamma(gamma, X):
    return 1.0 / X.shape[1] if gamma is None else gamma

def polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=0.0):
    Y = X if Y is None else Y
    gamma = _default_gamma(gamma, X)
    return (gamma * (X @ Y.T) + coef0) ** degree

def rbf_kernel(X, Y=None, gamma=None):
    Y = X if Y is None else Y
    gamma = _default_gamma(gamma, X)
    x2 = np.sum(X * X, axis=1)[:, None]
    y2 = np.sum(Y * Y, axis=1)[None, :]
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    return np.exp(-gamma * d2)

class MaxMarginKernelClassifier:
    def __init__(self, kernel=linear_kernel, C=1.0, tol=1e-7):
        self.kernel = kernel
        self.C = C
        self.tol = tol

    def fit(self, X, y):
        y = y.astype(float)
        K = self.kernel(X, X)
        P = (y[:, None] * y[None, :]) * K
        alpha = solve_qp(
            P=P,
            q=-np.ones(len(y)),
            A_eq=y,
            b_eq=0.0,
            bounds=(0.0, self.C),
        )

        sv = alpha > self.tol
        self.alpha = alpha[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        free = sv & (alpha < self.C - self.tol)
        bias_idx = free if np.any(free) else sv
        support_scores = (self.alpha * self.sv_y) @ K[np.ix_(sv, bias_idx)]
        self.b = float(np.mean(y[bias_idx] - support_scores))
        self.rho = -self.b
        return self

    def decision_function(self, X):
        Kx = self.kernel(X, self.sv_X)
        return Kx @ (self.alpha * self.sv_y) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
```

This is the binary, unweighted C-SVC core. LIBSVM implements the same dual as a minimization with `q = -1`, stores `sv_coef_i = y_i alpha_i`, stores `rho = -b`, evaluates `sum_i sv_coef_i K(x, SV_i) - rho`, and extends the core with per-class costs and one-vs-one multiclass classification.
