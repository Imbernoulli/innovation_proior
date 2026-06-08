# Matrix concentration: the matrix Laplace-transform method and the matrix Bernstein / Chernoff bounds

## Problem

Let $\mtx X_1,\dots,\mtx X_n$ be independent random self-adjoint matrices of dimension $d$. Bound the large-deviation probability of the largest eigenvalue of the sum,
$$
\Prob{\lambda_{\max}\Big(\sum\nolimits_k\mtx X_k\Big)\geq t},
$$
using only simple, checkable information about the individual summands (a uniform eigenvalue bound; the mean; the variance), with explicit constants and at finite dimension. By the self-adjoint dilation the same machinery controls the smallest eigenvalue and the spectral norm of a sum of rectangular matrices.

## Key idea

The scalar Laplace-transform (Cramér) method works because the mgf factorizes, equivalently the cgf is additive. The matrix exponential does **not** turn sums into products, and the Golden–Thompson inequality $\trace e^{\mtx A+\mtx H}\leq\trace(e^{\mtx A}e^{\mtx H})$ that substitutes for it only handles **two** matrices (its three-matrix form is false). Peeling $n$ summands apart with Golden–Thompson inflates the scale parameter from "the eigenvalue of a sum" $\lambda_{\max}(\sum_k\mtx A_k^2)$ to "the sum of eigenvalues" $\sum_k\lambda_{\max}(\mtx A_k^2)$ — a loss of up to a factor $d$ **in the exponent**.

The fix is to generalize cgf **additivity** rather than mgf multiplicativity, using Lieb's trace-concavity theorem as a probabilistic statement. Bounding $\lambda_{\max}$ by the trace converts the problem into a trace-exponential, whose concavity lets Jensen draw the expectation inside the cgf **without separating the summands**. The cgfs stay under a single exponential, so the scale parameter remains an eigenvalue-of-a-sum and the dimension survives only as a benign multiplicative prefactor $d$.

## The method, assembled

**Matrix Laplace transform.** For random self-adjoint $\mtx Y$ and all $t$,
$$
\Prob{\lambda_{\max}(\mtx Y)\geq t}\leq\inf_{\theta>0}e^{-\theta t}\,\Expect\trace e^{\theta\mtx Y}.
$$
*Proof.* Homogeneity of $\lambda_{\max}$, monotonicity of the scalar exponential, and Markov give $\Prob{\lambda_{\max}(\mtx Y)\geq t}\leq e^{-\theta t}\Expect e^{\lambda_{\max}(\theta\mtx Y)}$; then $e^{\lambda_{\max}(\theta\mtx Y)}=\lambda_{\max}(e^{\theta\mtx Y})\leq\trace e^{\theta\mtx Y}$ (spectral mapping; trace dominates $\lambda_{\max}$ for pd matrices). Optimize over $\theta$.

**Lieb's theorem.** For fixed self-adjoint $\mtx H$, the map $\mtx A\mapsto\trace\exp(\mtx H+\log\mtx A)$ is concave on the positive-definite cone.

**Cumulant inequality (its probabilistic corollary).** For fixed self-adjoint $\mtx H$ and random self-adjoint $\mtx X$,
$$
\Expect\trace\exp(\mtx H+\mtx X)\leq\trace\exp\big(\mtx H+\log\Expect e^{\mtx X}\big).
$$
*Proof.* Put $\mtx Y=e^{\mtx X}$ (pd), so $\mtx X=\log\mtx Y$; Lieb makes $\mtx Y\mapsto\trace\exp(\mtx H+\log\mtx Y)$ concave, and Jensen moves $\Expect$ inside the $\log$.

**Subadditivity of matrix cgfs.** For independent self-adjoint $\{\mtx X_k\}$ and $\theta\in\mathbb R$,
$$
\Expect\trace\exp\Big(\sum\nolimits_k\theta\mtx X_k\Big)\leq\trace\exp\Big(\sum\nolimits_k\log\Expect e^{\theta\mtx X_k}\Big).
$$
*Proof.* Apply the cumulant inequality once per summand via the tower property, with $\mtx H_m=\sum_{k<m}\theta\mtx X_k+\sum_{k>m}\log\Expect e^{\theta\mtx X_k}$ fixed relative to $\mtx X_m$.

**Master tail bound.** Substituting subadditivity into the Laplace transform bound,
$$
\Prob{\lambda_{\max}\Big(\sum\nolimits_k\mtx X_k\Big)\geq t}\leq\inf_{\theta>0}e^{-\theta t}\,\trace\exp\Big(\sum\nolimits_k\log\Expect e^{\theta\mtx X_k}\Big).
$$

**Deployment corollary.** If $\Expect e^{\theta\mtx X_k}\preceq e^{g(\theta)\mtx A_k}$ for $\theta>0$ (some scalar $g\geq0$ and fixed $\mtx A_k$), set $\rho=\lambda_{\max}(\sum_k\mtx A_k)$. Then
$$
\Prob{\lambda_{\max}\Big(\sum\nolimits_k\mtx X_k\Big)\geq t}\leq d\cdot\inf_{\theta>0}e^{-\theta t+g(\theta)\rho}=d\cdot\exp\big(-\rho\,g^*(t/\rho)\big).
$$
*Proof.* $\log$ is operator monotone, so $\log\Expect e^{\theta\mtx X_k}\preceq g(\theta)\mtx A_k$; $\trace\exp$ is monotone; $\trace\exp(g(\theta)\sum_k\mtx A_k)\leq d\,e^{g(\theta)\rho}$ since $\trace\leq d\,\lambda_{\max}$ on pd matrices and $g\geq0$.

## Named inequalities (each = one scalar inequality lifted by the transfer rule, then the corollary)

**Matrix Gaussian & Rademacher series.** $\{\mtx A_k\}$ fixed self-adjoint, $\{\xi_k\}$ independent standard normal or Rademacher. From $\cosh(\mtx A)\preceq e^{\mtx A^2/2}$ (Rademacher) and $\Expect e^{\gamma\mtx A}=e^{\mtx A^2/2}$ (Gaussian, exact), $g(\theta)=\theta^2/2$, $\sigma^2=\|\sum_k\mtx A_k^2\|$:
$$
\Prob{\lambda_{\max}\Big(\sum\nolimits_k\xi_k\mtx A_k\Big)\geq t}\leq d\cdot e^{-t^2/(2\sigma^2)},\qquad
\Prob{\Big\|\sum\nolimits_k\xi_k\mtx A_k\Big\|\geq t}\leq 2d\cdot e^{-t^2/(2\sigma^2)}.
$$
Rectangular: $\sigma^2=\max\{\|\sum_k\mtx B_k\mtx B_k^*\|,\|\sum_k\mtx B_k^*\mtx B_k\|\}$, dimension $d_1+d_2$.

**Matrix Chernoff.** $\{\mtx X_k\}$ psd with $\lambda_{\max}(\mtx X_k)\leq R$, $\mu_{\max}=\lambda_{\max}(\sum_k\Expect\mtx X_k)$, $\mu_{\min}=\lambda_{\min}(\sum_k\Expect\mtx X_k)$. From the chord $e^{\theta x}\leq1+(e^\theta-1)x$ on $[0,1]$ and the choices $\theta=R^{-1}\log(1+\delta)$ (upper) and $\theta=-R^{-1}\log(1-\delta)$ (lower):
$$
\Prob{\lambda_{\max}\Big(\sum\nolimits_k\mtx X_k\Big)\geq(1+\delta)\mu_{\max}}\leq d\Big[\tfrac{e^{\delta}}{(1+\delta)^{1+\delta}}\Big]^{\mu_{\max}/R}\ (\delta\geq0),\quad
\Prob{\lambda_{\min}\Big(\sum\nolimits_k\mtx X_k\Big)\leq(1-\delta)\mu_{\min}}\leq d\Big[\tfrac{e^{-\delta}}{(1-\delta)^{1-\delta}}\Big]^{\mu_{\min}/R}\ (\delta\in[0,1]).
$$

**Matrix Bernstein.** $\{\mtx X_k\}$ with $\Expect\mtx X_k=\mtx 0$ and $\lambda_{\max}(\mtx X_k)\leq R$; matrix variance $\sigma^2=\|\sum_k\Expect\mtx X_k^2\|$. In the normalized case $R=1$, the increasing remainder $f(x)=(e^{\theta x}-\theta x-1)/x^2=\int_0^\theta(\theta-s)e^{sx}\,ds$ gives $e^{\theta\mtx X}\preceq\Id+\theta\mtx X+(e^\theta-\theta-1)\mtx X^2$, so centering yields $\Expect e^{\theta\mtx X}\preceq\exp((e^\theta-\theta-1)\Expect\mtx X^2)$. The optimizer is $\theta=R^{-1}\log(1+Rt/\sigma^2)$ after rescaling:
$$
\Prob{\lambda_{\max}\Big(\sum\nolimits_k\mtx X_k\Big)\geq t}\leq d\cdot\exp\Big(-\frac{\sigma^2}{R^2}h\big(\tfrac{Rt}{\sigma^2}\big)\Big)\leq d\cdot\exp\Big(\frac{-t^2/2}{\sigma^2+Rt/3}\Big),\quad h(u)=(1+u)\log(1+u)-u.
$$
The smoothing uses $h(u)\geq(u^2/2)/(1+u/3)$ for $u\geq0$. Subgaussian for $t\lesssim\sigma^2/R$, subexponential for $t\gtrsim\sigma^2/R$. Rectangular: $\sigma^2=\max\{\|\sum_k\Expect\mtx Z_k\mtx Z_k^*\|,\|\sum_k\Expect\mtx Z_k^*\mtx Z_k\|\}$, dimension $d_1+d_2$.

## What the pieces mean

- **Matrix variance** $\sigma^2=\|\sum_k\Expect\mtx X_k^2\|=\|\Expect\mtx Y^2\|$ (centered) is the spectral norm of the expected squared deviation — one matrix encoding the variance in every direction; its norm is the scale of normal concentration. Rectangular matrices carry two independent squares ($\mtx B\mtx B^*$ row-space, $\mtx B^*\mtx B$ column-space), a noncommutative sum of squares.
- **Dimensional factor $d$** is the price of bounding $\lambda_{\max}$ by the trace. It is *necessary* (the diagonal Gaussian $\sum_k\gamma_k\mtx E_{kk}$ has norm $\approx\sqrt{2\log d}$, which the prefactor reproduces) and may *overstate* (GOE: bound off by $\sqrt{\log d}$ from the sharp $2\sqrt d$). It can be replaced by the *effective* dimension if the ranges lie in a lower-dimensional subspace.

## Optional numerical check

```python
import numpy as np

def lambda_max(M):
    return np.linalg.eigvalsh((M + M.conj().T) / 2).max()

def matrix_exp(M):
    w, V = np.linalg.eigh((M + M.conj().T) / 2)
    return (V * np.exp(w)) @ V.conj().T

# Master deployment bound:  P{ lambda_max(sum X_k) >= t } <= d * inf_theta e^{-theta t + g(theta) rho}
def master_bound(d, g, rho, t, thetas):
    return d * min(np.exp(-th * t + g(th) * rho) for th in thetas)

# Matrix Gaussian / Rademacher series  sum_k xi_k A_k
#   g(theta) = theta^2/2,  sigma^2 = || sum_k A_k^2 ||  (eigenvalue of a sum)
def gaussian_series_bound(A_list, t, two_sided=True):
    d = A_list[0].shape[0]
    sigma2 = lambda_max(sum(A @ A for A in A_list))
    pref = 2 * d if two_sided else d
    return pref * np.exp(-t**2 / (2 * sigma2))

# Matrix Bernstein (bounded case):  E X_k = 0,  lambda_max(X_k) <= R,  sigma^2 = || sum_k E X_k^2 ||
def bernstein_bound(EX2_list, R, t):
    d = EX2_list[0].shape[0]
    sigma2 = lambda_max(sum(EX2_list))
    return d * np.exp(-(t**2 / 2) / (sigma2 + R * t / 3))

# Matrix Chernoff (psd, lambda_max(X_k) <= R):  binomial-type tails of the extreme eigenvalues
def chernoff_upper(mu_max, R, delta, d):
    return d * (np.exp(delta) / (1 + delta)**(1 + delta))**(mu_max / R)

def chernoff_lower(mu_min, R, delta, d):
    return d * (np.exp(-delta) / (1 - delta)**(1 - delta))**(mu_min / R)

# Sanity check: a Rademacher series obeys its two-sided Gaussian-series bound.
def _check(seed=0, d=8, n=20, trials=20000):
    rng = np.random.default_rng(seed)
    A = []
    for _ in range(n):
        G = rng.standard_normal((d, d)); A.append((G + G.T) / 2)
    sims = []
    for _ in range(trials):
        s = sum(rng.choice([-1.0, 1.0]) * Ak for Ak in A)
        sims.append(np.linalg.norm(s, 2))
    sims = np.array(sims)
    for t in [6, 8, 10, 12]:
        emp = (sims >= t).mean()
        assert emp <= gaussian_series_bound(A, t) + 1e-12, (t, emp)
    return "ok"

if __name__ == "__main__":
    print(_check())
```
