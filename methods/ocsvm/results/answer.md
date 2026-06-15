# One-Class SVM (OCSVM), Distilled

One-Class SVM estimates a high-mass support region from unlabelled data by learning a
maximum-margin hyperplane in kernel feature space. The data side is treated as normal;
the other side is treated as novel or anomalous. It keeps the standard SVM ingredients:
a kernel expansion, the `1/2 ||w||^2` regularizer, a convex dual QP, and support vectors.

## Objective

Given `x_1, ..., x_l ~ P`, choose `nu in (0, 1]` and solve

```text
min_{w, xi, rho}  1/2 ||w||^2 + (1/(nu l)) sum_i xi_i - rho
s.t.              <w, Phi(x_i)> >= rho - xi_i,   xi_i >= 0.
```

The signed decision is

```text
f(x) = sgn(<w, Phi(x)> - rho),
```

positive for inliers and negative for outliers.

## Dual And Offset

With multipliers `alpha_i >= 0` for the margin constraints and `beta_i >= 0` for
`xi_i >= 0`, stationarity gives

```text
w = sum_i alpha_i Phi(x_i),
alpha_i = 1/(nu l) - beta_i,
sum_i alpha_i = 1.
```

Substitution yields the dual

```text
min_alpha  1/2 sum_{ij} alpha_i alpha_j k(x_i, x_j)
s.t.       0 <= alpha_i <= 1/(nu l),   sum_i alpha_i = 1.
```

The kernel decision is

```text
f(x) = sgn(sum_i alpha_i k(x_i, x) - rho).
```

Recover `rho` from any non-bound support vector:

```text
if 0 < alpha_i < 1/(nu l), then
rho = <w, Phi(x_i)> = sum_j alpha_j k(x_j, x_i).
```

## What `nu` Controls

Let `U = 1/(nu l)`.

- If a point has positive slack, complementary slackness forces `beta_i = 0`, so
  `alpha_i = U`. Outliers are pinned at the upper box constraint.
- Since `sum_i alpha_i = 1`, at most `nu l` points can be pinned at `U`; the training
  outlier fraction is therefore `<= nu`.
- Each support vector has `alpha_i > 0` and contributes at most `U`; reaching total
  mass 1 requires at least `nu l` support vectors, so the support-vector fraction is
  `>= nu`.
- With an analytic nonconstant kernel and a distribution without discrete components,
  the margin-only fraction vanishes asymptotically, so the outlier and support-vector
  fractions both converge to `nu`.

At `nu = 1`, the box and equality force every `alpha_i = 1/l`, giving a thresholded
Parzen-window expansion `(1/l) sum_i k(x_i, x) - rho`. As `nu -> 0`, the upper box
constraint disappears and the slack penalty becomes infinite, giving the hard-margin
support estimator.

## Relation To SVDD

The soft feature-space ball formulation solves

```text
min_{R,a,xi}  R^2 + (1/(nu l)) sum_i xi_i
s.t.          ||Phi(x_i) - a||^2 <= R^2 + xi_i,   xi_i >= 0,
```

with dual

```text
max_alpha  sum_i alpha_i k(x_i, x_i)
           - sum_{ij} alpha_i alpha_j k(x_i, x_j)
s.t.       sum_i alpha_i = 1,   0 <= alpha_i <= 1/(nu l).
```

For kernels with constant `k(x,x)`, including the Gaussian kernel, the linear term is
constant under `sum_i alpha_i = 1`. The ball dual and the OCSVM dual therefore have
the same optimizer, up to the irrelevant positive factor `1/2`; the decision functions
also coincide. Geometrically, constant `k(x,x)` puts all mapped points on a sphere, so
the smallest enclosing segment of that sphere is cut by the maximum-margin hyperplane.

## SMO Pair Step

The dual has one equality, so the smallest feasible decomposition step updates a pair.
Freeze all variables except `alpha_1, alpha_2`, set

```text
Delta = 1 - sum_{i>=3} alpha_i,
alpha_1 + alpha_2 = Delta,
K_ij = k(x_i, x_j),
C_i = sum_{j>=3} alpha_j K_ij.
```

Eliminating `alpha_1 = Delta - alpha_2` and differentiating gives

```text
alpha_2 = [Delta(K_11 - K_12) + C_1 - C_2] /
          (K_11 + K_22 - 2K_12).
```

In terms of the current outputs

```text
O_i = K_1i alpha_1* + K_2i alpha_2* + C_i,
```

the Newton update is

```text
alpha_2 = alpha_2* + (O_1 - O_2) / (K_11 + K_22 - 2K_12).
```

Then clip `alpha_2` to the interval that keeps both variables feasible,

```text
max(0, Delta - U) <= alpha_2 <= min(U, Delta),  U = 1/(nu l),
```

set `alpha_1 = Delta - alpha_2`, recompute `rho`, and iterate over KKT-violating
pairs.

## Implementation

The canonical implementation path is PyOD's wrapper delegating to
`sklearn.svm.OneClassSVM`, which delegates the dual optimization to libsvm. PyOD
inverts sklearn's signed decision values so larger scores mean more anomalous.

```python
from sklearn.svm import OneClassSVM


class SupportRegionDetector:
    """PyOD-style wrapper around sklearn's libsvm-backed one-class solver."""

    def __init__(self, nu=0.5, gamma="auto", kernel="rbf"):
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X):
        # X is unlabelled feature data; scale features before fitting an RBF kernel.
        self.model.fit(X)
        return self

    def decision_function(self, X):
        # sklearn returns sum_i alpha_i k(x_i, x) - rho, positive for inliers.
        # PyOD inverts that ordering so larger values are more anomalous.
        return -self.model.decision_function(X)
```

For the PyOD-style settings used here, `kernel='rbf'`, `nu=0.5`, and `gamma='auto'`
mean `gamma = 1 / n_features`. In sklearn itself, `gamma='scale'` uses
`1 / (n_features * X.var())`; pass `gamma='auto'` when matching the PyOD wrapper.
