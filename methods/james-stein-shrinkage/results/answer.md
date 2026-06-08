# The James–Stein estimator and Stein's paradox

## Problem

Observe one draw $X \sim N(\theta, \sigma^2 I)$ in $\mathbb{R}^p$ ($\sigma^2$ known; take $\sigma^2=1$ unless noted), with coordinates independent and the means $\theta_i$ unrelated. Estimate $\theta$ under total squared-error loss $L(\theta,d)=\|\theta-d\|^2$, with risk $R(\theta,\delta)=E_\theta\|\delta(X)-\theta\|^2$. The usual estimator $\delta_0(X)=X$ (the MLE / sample mean) has risk $R(\theta,\delta_0)=p\sigma^2$ for all $\theta$ and is unbiased, minimum-variance-unbiased, minimax, and best invariant.

## Key idea

In dimension $p\ge3$, $\delta_0=X$ is **inadmissible**: a single shrinkage estimator that pulls every coordinate toward a common point has strictly smaller total risk for *every* $\theta$, even though the coordinates are independent and unrelated. The shrinkage couples the problems only through the joint statistic $\|X\|^2$, which measures the systematic overshoot $E\|X\|^2=\|\theta\|^2+p\sigma^2$. The win is in the *sum* of squared errors; individual coordinates may be estimated worse. This is **Stein's paradox**.

## The estimator

$$\boxed{\ \hat\theta^{\mathrm{JS}}(X)=\Big(1-\frac{(p-2)\sigma^2}{\|X\|^2}\Big)X\ }$$

(shrink toward $0$; toward a chosen center $\mu_0$, apply it to $X-\mu_0$ and add $\mu_0$ back). The positive-part refinement $\hat\theta^{\mathrm{JS+}}=\big(1-(p-2)\sigma^2/\|X\|^2\big)_+X$ clamps the factor at $0$ and dominates $\hat\theta^{\mathrm{JS}}$ under total squared-error risk, but its risk is not the same formula below.

## Domination theorem

**Theorem.** For $p\ge3$,
$$R(\theta,\hat\theta^{\mathrm{JS}})=p\sigma^2-(p-2)^2\sigma^4\,E_\theta\!\Big[\frac1{\|X\|^2}\Big]\ <\ p\sigma^2=R(\theta,\delta_0)\quad\text{for all }\theta,$$
so $\delta_0=X$ is inadmissible. In the unit-variance case, equivalently $R(\theta,\hat\theta^{\mathrm{JS}})=p-E\big[(p-2)^2/(p-2+2K)\big]$ with $K\sim\mathrm{Poisson}(\|\theta\|^2/2)$. At $\theta=0$, the general-variance formula equals $2\sigma^2$.

### Stein's lemma (the engine)

For $Z\sim N(0,1)$ and absolutely continuous $f$ with $E|f'(Z)|<\infty$,
$$E[Z f(Z)]=E[f'(Z)].$$
*Proof.* $E[f'(Z)]=\int f'\phi=-\int f\phi'=\int f(z)\,z\,\phi(z)\,dz=E[Zf(Z)]$, using $\phi'(z)=-z\phi(z)$ and the vanishing boundary term. $\square$ For $X\sim N(\mu,\sigma^2)$, $\tfrac1{\sigma^2}E[(X-\mu)f(X)]=E[f'(X)]$; coordinatewise in $\mathbb R^p$ for $X\sim N(\mu,\sigma^2I)$ and $f:\mathbb R^p\to\mathbb R^p$,
$$\tfrac1{\sigma^2}\sum_i E[(X_i-\mu_i)f_i(X)]=E[\operatorname{div}f],\qquad \operatorname{div}f=\sum_i\partial f_i/\partial X_i.$$

### Unbiased risk estimate (SURE)

Write $\hat\theta=X-g(X)$. With $\sigma^2=1$,
$$R(\theta,\hat\theta)=E\|(X-\theta)-g\|^2=\underbrace{E\|X-\theta\|^2}_{p}-2\,E[(X-\theta)\cdot g]+E\|g\|^2=p-2E[\operatorname{div}g]+E\|g\|^2,$$
the middle step by Stein's lemma. (For general $\sigma^2$: $R=p\sigma^2-2\sigma^2E[\operatorname{div}g]+E\|g\|^2$.)

### Apply to James–Stein

In the unit-variance normalization, take $g(X)=c\,X/\|X\|^2$ (so $\hat\theta=(1-c/\|X\|^2)X$). Then $\|g\|^2=c^2/\|X\|^2$, and
$$\frac{\partial}{\partial X_i}\frac{cX_i}{\|X\|^2}=\frac{c}{\|X\|^2}-\frac{2cX_i^2}{\|X\|^4}\ \Rightarrow\ \operatorname{div}g=\frac{cp}{\|X\|^2}-\frac{2c}{\|X\|^2}=\frac{c(p-2)}{\|X\|^2}.$$
Hence
$$R(\theta,\hat\theta)=p-\big(2c(p-2)-c^2\big)\,E\Big[\tfrac1{\|X\|^2}\Big].$$
Maximizing the improvement $2c(p-2)-c^2$ over $c$ gives $c=p-2$, improvement $(p-2)^2$, and
$$R(\theta,\hat\theta^{\mathrm{JS}})=p-(p-2)^2E[1/\|X\|^2]<p.$$
Restoring $\sigma^2$ replaces $g$ by $c\sigma^2X/\|X\|^2$ and gives the theorem's factor $\sigma^4$ in the risk reduction. For $p\ge3$, $E[1/\|X\|^2]$ is finite and positive, so domination is strict for all $\theta$. At $\theta=0$, $\|X\|^2/\sigma^2\sim\chi^2_p$, $E[1/\|X\|^2]=1/(\sigma^2(p-2))$, so $R=p\sigma^2-(p-2)\sigma^2=2\sigma^2$. $\blacksquare$

## Why $p-2$, and why $p\ge3$

- **Optimization:** $c=p-2$ maximizes the SURE risk reduction $2c(p-2)-c^2$.
- **Geometry:** in the unit-variance normalization, the typical observation sits on a sphere of squared radius $\approx\|\theta\|^2+p$; the deterministic best pull-in factor is $1-(p-1)/\|X\|^2$, but averaging over the stochastic jiggle of $X$ shaves $p-1$ to $p-2$.
- **Empirical Bayes:** in the unit-variance normalization, under $\theta_i\sim N(0,A)$ the Bayes rule shrinks by $1/(A+1)$, and $(p-2)/\|X\|^2$ is an unbiased estimate of $1/(A+1)$ (since $\|X\|^2\sim(A+1)\chi^2_p$); the JS estimator is the empirical-Bayes plug-in. The domination holds for fixed, unrelated $\theta$ with no prior.
- **Threshold:** $p-2\le0$ for $p\le2$; $\delta_0$ is admissible at $p=1$ (Blyth 1951; Hodges–Lehmann 1951) and at $p=2$ (via the information inequality). The overshoot $\approx p$ is distinguishable from the $O(\sqrt p)$ noise in $\|X\|^2$ only for $p\ge3$.

For the $p=2$ information-inequality step, average any hypothetical dominator over rotations, giving spherical bias $b(\theta)=-\varphi(\|\theta\|^2)\theta$. The lower bound becomes $R\ge2+t\varphi^2-4\varphi-4t\varphi'$ with $t=\|\theta\|^2$; domination would force $0\ge\psi^2/t-4\psi'$ for $\psi=t\varphi$. Thus $\psi'\ge\psi^2/(4t)$. If $\psi$ is ever negative, integrating toward $0$ makes a logarithm diverge while $-1/\psi$ stays bounded; if it is ever positive, integrating toward $\infty$ gives the same contradiction. Hence $\psi\equiv0$, so no strict dominator exists.

## Code

```python
import numpy as np

def james_stein(X, sigma=1.0, center=0.0, positive_part=False):
    """Plain James-Stein estimate of theta from one draw X ~ N(theta, sigma^2 I), p >= 3.

    Shrinks toward `center` by 1 - (p-2) sigma^2 / ||X-center||^2; dominates X
    in total squared-error risk for p >= 3. positive_part enables the clamped refinement.
    """
    X = np.asarray(X, dtype=float)
    center = np.broadcast_to(np.asarray(center, dtype=float), X.shape)
    p = X.size
    if p < 3:
        raise ValueError("domination requires dimension p >= 3")
    Z = X - center
    norm2 = np.sum(Z**2)
    if norm2 == 0.0:
        if positive_part:
            return center.copy()
        raise ZeroDivisionError("plain James-Stein factor is undefined at the center")
    factor = 1.0 - (p - 2) * sigma**2 / norm2
    if positive_part:
        factor = max(factor, 0.0)
    return center + factor * Z

def usual_estimator(X, sigma=1.0):
    """The MLE / sample mean: report X itself. Risk is exactly p*sigma^2."""
    return np.asarray(X, dtype=float)

def plain_js_risk_from_identity(theta, sigma=1.0, n=200_000, rng=None):
    """Monte-Carlo evaluation of the exact plain-JS risk identity.

    R = p*sigma^2 - (p-2)^2*sigma^4*E[1/||X||^2].
    At theta = 0 this equals 2*sigma^2 exactly.
    """
    theta = np.asarray(theta, dtype=float)
    p = theta.size
    if p < 3:
        raise ValueError("plain James-Stein domination requires dimension p >= 3")
    rng = rng or np.random.default_rng()
    X = theta + sigma * rng.standard_normal((n, p))
    inv = np.mean(1.0 / np.sum(X**2, axis=1))
    return p * sigma**2 - (p - 2)**2 * sigma**4 * inv

def risk_montecarlo(estimator, theta, sigma=1.0, n=200_000, rng=None):
    """E||estimator(X) - theta||^2 by averaging over draws X ~ N(theta, sigma^2 I)."""
    theta = np.asarray(theta, dtype=float)
    rng = rng or np.random.default_rng()
    total = 0.0
    for _ in range(n):
        X = theta + sigma * rng.standard_normal(theta.shape)
        d = estimator(X, sigma=sigma)
        total += np.sum((d - theta) ** 2)
    return total / n
```
