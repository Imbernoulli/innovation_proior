# The kernel trick and the representer theorem

## Problem

Linear learning methods (the dual perceptron, ridge/least-squares regression, discriminant analysis, margin classifiers) touch the data only through pairwise inner products $\langle x_i, x_j\rangle$. This makes them efficient but flat: they can only produce linear decision/regression surfaces. Lifting the data through an explicit nonlinear feature map $\phi$ buys curvature but costs an unaffordable (sometimes infinite) number of feature coordinates. We want the modelling power of a rich feature space at the cost of the inner products we already compute, and we want any learning problem posed in such a space to remain finitely computable.

## Key idea

Two results, jointly.

**The kernel trick.** Replace every inner product $\langle x_i, x_j\rangle$ in an inner-product-only algorithm with a *kernel* $k(x_i, x_j)$. This is legitimate exactly when $k$ is *positive definite* — every Gram matrix $K=(k(x_i,x_j))$ is positive semidefinite — because positive definiteness is precisely the condition under which $k(x,x')=\langle\phi(x),\phi(x')\rangle$ for some feature map $\phi$ into a (possibly infinite-dimensional) inner-product space. The feature map is never computed; only $k$ is evaluated.

**The representer theorem.** For a regularized risk in the reproducing kernel Hilbert space $\mathcal H_k$ of a positive-definite kernel, where the loss depends on $f$ only through its values at the training points and the penalty is a strictly increasing function of $\|f\|_{\mathcal H_k}$, the minimizer lies in the finite span of the kernels at the training points:
$$ f(\cdot) = \sum_{i=1}^m \alpha_i\, k(x_i, \cdot). $$
The infinite-dimensional optimization collapses to an $m$-dimensional solve in the coefficients $\alpha$.

## Kernel ⟺ feature-map correspondence

For a finite domain $\{x_1,\dots,x_m\}$, write the Gram matrix $K=(k(x_i,x_j))$. $K$ symmetric ⇒ $K=V\Lambda V^\top$. Define $\Phi(x_i)=(\sqrt{\lambda_1}v_1^{(i)},\dots,\sqrt{\lambda_m}v_m^{(i)})$; then $\langle\Phi(x_i),\Phi(x_j)\rangle=\sum_t\lambda_t v_t^{(i)}v_t^{(j)}=K_{ij}$, requiring $\lambda_t\ge 0$. The nonnegativity is forced: if $\lambda_s<0$ and a feature map existed, then $z=\sum_i v_s^{(i)}\Phi(x_i)$ would have $\|z\|^2=v_s^\top K v_s=\lambda_s<0$, impossible. Hence

$$ k \text{ is an inner product of features} \iff K \succeq 0 \text{ for all finite samples} \iff k \text{ positive definite.} $$

In the continuous case **Mercer's theorem** supplies the same conclusion: a continuous symmetric positive-definite kernel on a compact domain expands as $k(x,x')=\sum_j \lambda_j\psi_j(x)\psi_j(x')$ with $\lambda_j\ge 0$, giving the feature map $\phi(x)=(\sqrt{\lambda_j}\psi_j(x))_j$.

The same $2\times2$ positivity gives the kernel Cauchy–Schwarz inequality. With $a=k(u,u)$, $b=k(u,v)$, $d=k(v,v)$ and $c=(d,-b)$,
$$0\le c^\top\begin{pmatrix}a&b\\b&d\end{pmatrix}c=d(ad-b^2).$$
If $d>0$, $b^2\le ad$; if $d=0$, positivity of $(t,1)K(t,1)^\top=a t^2+2bt$ for all $t$ forces $b=0$. Thus $k(u,v)^2\le k(u,u)k(v,v)$.

## RKHS and the reproducing property

For an arbitrary set $\mathcal X$ (no continuity or measure needed), take $\phi(x)=k(\cdot,x)$, form the span of these functions, and define
$$ \Big\langle \textstyle\sum_i\alpha_i k(\cdot,x_i),\ \sum_j\beta_j k(\cdot,x_j')\Big\rangle := \sum_{i,j}\alpha_i\beta_j\,k(x_i,x_j'). $$
This is well-defined: if an expansion of $f$ is changed by a zero function $r$, the pairing changes by $\sum_j\beta_j r(x_j')=0$, and the same symmetric argument handles changes in the expansion of the second function. The form is symmetric, bilinear, and positive by positive definiteness of $k$. The **reproducing property**
$$ \langle k(\cdot,x), f\rangle = f(x), \qquad \text{in particular } \langle k(\cdot,x),k(\cdot,x')\rangle = k(x,x'), $$
makes $\phi(x)=k(\cdot,x)$ a feature map. Cauchy–Schwarz $|f(x)|^2=|\langle k(\cdot,x),f\rangle|^2\le k(x,x)\langle f,f\rangle$ shows $\langle f,f\rangle=0\Rightarrow f\equiv 0$, so $\langle\cdot,\cdot\rangle$ is a genuine inner product. Completing the span yields the RKHS $\mathcal H_k$. **Moore–Aronszajn:** every positive-definite kernel corresponds to a unique RKHS, and conversely.

## Representer theorem — statement and proof

**Theorem.** Let $k$ be a positive-definite kernel on $\mathcal X$ with RKHS $\mathcal H_k$, let $(x_1,y_1),\dots,(x_m,y_m)$ be training data, let $c$ be an arbitrary cost function of $\{(x_i,y_i,f(x_i))\}$ (possibly $+\infty$-valued, possibly coupling points), and let $g:[0,\infty)\to\mathbb R$ be strictly monotonically increasing. Then any minimizer over $\mathcal H_k$ of
$$ c\big((x_1,y_1,f(x_1)),\dots,(x_m,y_m,f(x_m))\big) + g(\|f\|_{\mathcal H_k}) $$
admits $f(\cdot)=\sum_{i=1}^m \alpha_i\,k(x_i,\cdot)$.

**Proof.** Decompose any $f\in\mathcal H_k$ orthogonally with respect to $S=\mathrm{span}\{\phi(x_i)\}$, where $\phi(x_i)=k(\cdot,x_i)$:
$$ f=\sum_{i=1}^m\alpha_i\,\phi(x_i)+v,\qquad \langle v,\phi(x_j)\rangle=0\ \ \forall j. $$
*Fit term is independent of $v$.* By the reproducing property,
$$ f(x_j)=\langle f,\phi(x_j)\rangle=\Big\langle\sum_i\alpha_i\phi(x_i)+v,\phi(x_j)\Big\rangle=\sum_i\alpha_i\,k(x_i,x_j), $$
since $\langle v,\phi(x_j)\rangle=0$. So every value the cost sees is independent of $v$.
*Penalty is strictly reduced by $v=0$.* By Pythagoras, $\|f\|^2=\|\sum_i\alpha_i\phi(x_i)\|^2+\|v\|^2$, so
$$ g(\|f\|)=g\Big(\sqrt{\|\textstyle\sum_i\alpha_i\phi(x_i)\|^2+\|v\|^2}\Big)\ \ge\ g\Big(\|\textstyle\sum_i\alpha_i\phi(x_i)\|\Big), $$
with equality iff $v=0$ (strict monotonicity of $g$). Therefore setting $v=0$ leaves the cost unchanged while strictly decreasing the penalty unless $v=0$ already; any minimizer has $v=0$, hence $f=\sum_i\alpha_i\phi(x_i)=\sum_i\alpha_i k(x_i,\cdot)$. $\qquad\blacksquare$

The argument uses only (a) the cost depends on $f$ through $\{f(x_i)\}$ and (b) $g$ is a strictly increasing function of $\|f\|$. Convexity is not required for the representation (only for uniqueness of the reduced optimum). With $\lambda\|f\|^2$ and squared-error/hinge loss this recovers Wahba's classical statement (Kimeldorf & Wahba) and the SVM expansion; with general strictly-monotone $g$ it is the generalized form. A **semiparametric** extension adds an unpenalized parametric part $h\in\mathrm{span}\{\psi_p\}_{p=1}^M$ (e.g. the null space of the penalty), giving $f(\cdot)=\sum_i\alpha_i k(x_i,\cdot)+\sum_p\beta_p\psi_p(\cdot)$.

## Reduced finite problem

Substituting $f=\sum_i\alpha_i k(x_i,\cdot)$ gives $f(x_j)=(K\alpha)_j$ and $\|f\|^2=\alpha^\top K\alpha$, so the infinite-dimensional problem becomes
$$ \min_{\alpha\in\mathbb R^m}\ c\big((x_i,y_i,(K\alpha)_i)_{i=1}^m\big)+g\big(\sqrt{\alpha^\top K\alpha}\big), $$
an $m$-dimensional optimization expressed entirely through the Gram matrix $K$.

## Worked instance — kernel ridge regression

With squared-error loss and $g(\|f\|)=\lambda\|f\|^2$, the reduced problem is $\min_\alpha \|y-K\alpha\|^2+\lambda\,\alpha^\top K\alpha$. The gradient is $-2K(y-K\alpha)+2\lambda K\alpha$, hence stationarity requires $K((K+\lambda I)\alpha-y)=0$. For $\lambda>0$, $K+\lambda I$ is positive definite, and the canonical coefficient vector
$$\alpha=(K+\lambda I)^{-1}y$$
satisfies stationarity because $y-K\alpha=\lambda\alpha$; null-space differences in $\alpha$ do not change the represented RKHS function. Prediction is $f(x)=\sum_i\alpha_i k(x_i,x)$.

```python
import numpy as np

def gaussian_kernel(x, z, gamma):
    # Positive-definite k(x,z) = <phi(x), phi(z)> for an infinite-dimensional
    # feature map phi that is never constructed; only k is evaluated.
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")
    d = x - z
    return np.exp(-gamma * np.dot(d, d))

def gram_matrix(X, kernel):
    n = len(X)
    K = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j])
    return K

def is_positive_semidefinite(K, tol=1e-9):
    # Legitimacy of the substitution: every Gram matrix must be PSD
    # (a negative eigenvalue would be a negative squared length in feature space).
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    return bool(np.all(w >= -tol))

def fit(X, y, kernel, lam):
    # Representer theorem: minimizer of sum_i (y_i - f(x_i))^2 + lam ||f||^2
    # lies in span{k(x_i, .)} -> f = sum_i alpha_i k(x_i, .).
    if lam <= 0:
        raise ValueError("lam must be positive")
    K = gram_matrix(X, kernel)
    assert is_positive_semidefinite(K)
    alpha = np.linalg.solve(K + lam * np.eye(len(X)), y)
    return alpha

def predict(alpha, X_train, x, kernel):
    # f(x) = sum_i alpha_i k(x_i, x): the representer expansion, evaluated.
    return sum(alpha[i] * kernel(X_train[i], x) for i in range(len(X_train)))
```

## Why it matters

Positive definiteness is the exact dividing line between kernels that are inner products and those that are not, so it is the right and only condition the kernel trick needs. The representer theorem guarantees that however rich (even infinite-dimensional) the feature space, a regularized solution is pinned to the finite span of the training points — enabling kernelized regression, classification, PCA, and Gaussian-process / spline estimation to be carried out without ever materializing a feature vector.
