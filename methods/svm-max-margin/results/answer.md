# Maximum-Margin Classifier (Support-Vector Machine)

## Problem

Binary classification of $(x_i,y_i)$, $y_i\in\{-1,+1\}$, with good *generalization*: small expected test error, not merely small training error. Among the infinitely many hyperplanes that separate the training data, pick a single principled one whose capacity — and therefore generalization — is controlled, and do so even when the (feature-space) dimension is enormous or infinite.

## Key idea

Choose, among all separating hyperplanes, the one of **maximum margin** (largest distance to the nearest training points). The Vapnik–Chervonenkis bound on $\Delta$-margin separating hyperplanes — for data in a sphere of radius $R$, $h\le \min(\lceil R^2/\Delta^2\rceil, n)+1$ — shows that a large margin means low VC dimension *independent of the dimension* $n$. Maximizing the margin is thus structural risk minimization realized geometrically. The problem becomes a convex quadratic program; its Lagrangian dual depends on the data **only through inner products** (so any Mercer kernel can replace the dot product, giving nonlinear boundaries for free), its solution is **sparse** (a few support vectors), and **slack variables with a cost $C$** extend it to non-separable data.

## Derivation

Signed distance of $x$ to $\{w\cdot x+b=0\}$ is $D(x)/\lVert w\rVert$. A margin-$M$ separation means $y_i(w\cdot x_i+b)/\lVert w\rVert\ge M$. Fixing the scale by $M\lVert w\rVert=1$ gives $y_i(w\cdot x_i+b)\ge 1$ with the closest points at equality, supporting hyperplanes $w\cdot x+b=\pm1$, geometric margin $M=1/\lVert w\rVert$, and full gap $2/\lVert w\rVert$. Maximizing $M$ is equivalent to minimizing $\tfrac12\lVert w\rVert^2$.

**Primal QP (soft margin).**
$$ \min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2 + C\sum_{i=1}^{\ell}\xi_i \quad\text{s.t.}\quad y_i(w\cdot x_i+b)\ge 1-\xi_i,\ \ \xi_i\ge 0. $$
($\sum_i\xi_i$ upper-bounds the number of training errors; for separable data, a large enough $C$ recovers the hard-margin solution when the upper bound does not bind.)

**Lagrangian.** With $\alpha_i\ge0$ (margin constraints), $\mu_i\ge0$ (slack):
$$ L=\tfrac12\lVert w\rVert^2+C\textstyle\sum_i\xi_i-\sum_i\alpha_i[y_i(w\cdot x_i+b)-1+\xi_i]-\sum_i\mu_i\xi_i. $$
Stationarity: $w=\sum_i\alpha_i y_i x_i$, $\ \sum_i\alpha_i y_i=0$, $\ \alpha_i=C-\mu_i$ (so $\alpha_i\le C$).

**Dual QP.** Substituting back (slacks and $\mu_i$ cancel), with $K(x_i,x_j)=\Phi(x_i)\cdot\Phi(x_j)$:
$$ \max_{\alpha}\ \sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j) \quad\text{s.t.}\quad \sum_i\alpha_i y_i=0,\ \ 0\le\alpha_i\le C. $$
In matrix form $\max_\alpha\ \alpha^\top\mathbf 1-\tfrac12\alpha^\top D\alpha$, $D_{ij}=y_iy_jK(x_i,x_j)$. Size $=\ell$, not the feature dimension.

**KKT / support vectors.** Complementary slackness $\alpha_i[y_i(w\cdot x_i+b)-1+\xi_i]=0$:
- $\alpha_i=0$: correctly on or outside the margin but not contributing to the classifier;
- $0<\alpha_i<C$: $\xi_i=0$, on the margin (free support vector — used to compute $b$);
- $\alpha_i=C$: margin constraint is tight with $y_i(w\cdot x_i+b)=1-\xi_i$; the point may be on the margin, inside it, or misclassified (bounded support vector).

**Decision function.** $f(x)=\operatorname{sign}\!\big(\sum_i\alpha_i y_i K(x_i,x)+b\big)$, summed over support vectors only.

**Generalization.** Leave-one-out: removing a non-support vector leaves the boundary unchanged, so expected test error $\le \mathbb{E}[\#\text{support vectors}]/\ell$ — independent of dimension.

**Kernels (Mercer).** Under the usual Mercer assumptions, $K$ is a valid inner product when $\iint K(u,v)g(u)g(v)\,du\,dv\ge0$ for all $g\in L_2$. Standard choices: linear $u\cdot v$; polynomial $(u\cdot v+1)^d$; RBF $\exp(-\lVert u-v\rVert^2/2\sigma^2)$ (support vectors become RBF centers); sigmoid-shaped kernels only in parameter regimes where the resulting Gram matrices are positive semidefinite.

## Final form (code)

```python
import numpy as np

def linear_kernel(X, Y=None):
    Y = X if Y is None else Y
    return X @ Y.T

def poly_kernel(X, Y=None, degree=3, coef0=1.0):
    Y = X if Y is None else Y
    return (X @ Y.T + coef0) ** degree             # (x.x' + 1)^d

def rbf_kernel(X, Y=None, gamma=0.5):
    Y = X if Y is None else Y
    x2 = np.sum(X * X, axis=1)[:, None]
    y2 = np.sum(Y * Y, axis=1)[None, :]
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    return np.exp(-gamma * d2)                     # gamma = 1 / (2 sigma^2)

class MaxMarginClassifier:
    """min 1/2||w||^2 + C sum xi  ->  dual:
       max  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i,x_j)
       s.t. sum_i a_i y_i = 0,  0 <= a_i <= C.
       Decision: sign( sum_i a_i y_i K(x_i, x) + b )."""

    def __init__(self, kernel, C=1.0, tol=1e-6):
        self.kernel, self.C, self.tol = kernel, C, tol

    def fit(self, X, y):
        y = y.astype(float)
        K = self.kernel(X, X)                      # data enters only via inner products
        D = (y[:, None] * y[None, :]) * K          # D_ij = y_i y_j K(x_i,x_j)
        alpha = solve_qp(                          # min 1/2 a^T D a - 1^T a
            P=D, q=-np.ones_like(y),               #   equivalent to the dual maximum
            A_eq=y, b_eq=0.0,                      #   under sum_i a_i y_i = 0
            bounds=(0.0, self.C))
        sv = alpha > self.tol                       # KKT: a_i>0 only for support vectors
        self.alpha, self.sv_X, self.sv_y = alpha[sv], X[sv], y[sv]
        free = sv & (alpha < self.C - self.tol)     # 0<a<C  =>  xi=0, point on margin
        bias_idx = free if np.any(free) else sv
        margins = (self.alpha * self.sv_y) @ K[np.ix_(sv, bias_idx)]
        self.b = np.mean(y[bias_idx] - margins)
        return self

    def decision_function(self, X):
        Kx = self.kernel(X, self.sv_X)
        return (self.alpha * self.sv_y) @ Kx.T + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
```

The training QP scales with the number of examples, and prediction scales with the number of support vectors and one kernel evaluation per support vector, independent of the (possibly billion-dimensional or infinite) feature-space coordinates.
