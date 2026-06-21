I observe a single draw of a $p$-dimensional Gaussian vector $X\sim N(\theta,\sigma^2 I)$ with $\sigma^2$ known (take $\sigma^2=1$ unless noted), the coordinates independent and the means $\theta_i$ carrying physically unrelated quantities, and I must report the whole mean vector $\theta$ at once, paying the total squared-error loss $L(\theta,d)=\|\theta-d\|^2$ with risk $R(\theta,\delta)=E_\theta\|\delta(X)-\theta\|^2$. The estimator everyone reaches for is $\delta_0(X)=X$, report each coordinate as observed, and it wears every medal classical theory hands out: its risk is the clean constant $p\sigma^2$ for all $\theta$, it is the maximum-likelihood estimate, it is unbiased and minimum-variance among unbiased estimators, it is minimax under squared error, and it is the best translation-invariant rule. The natural expectation is therefore that $X$ is admissible — that no estimator can have risk weakly below $p\sigma^2$ everywhere and strictly below somewhere — and one sets out to prove it. The trouble is that none of those certificates actually says what we need: unbiasedness, minimaxity, and invariance are statements about single coordinates or about a symmetry one might be wrong to demand, never a statement that the *total* risk summed over $p$ coordinates cannot be lowered by a biased, coordinate-coupling rule. The Bayes shrinkage rule $\big(1-\tfrac1{A+1}\big)X$ would pull toward a point, but it needs a known prior variance $A$ and has no frequentist standing when $\theta$ is fixed and unrelated; the best invariant estimator only optimizes within a constraint. So the question of admissibility for $p\ge3$ is genuinely open, and the place to look is not the certificates but the data.

The diagnostic that breaks the expectation comes from staring at the squared length $\|X\|^2$. Writing $X=\theta+(X-\theta)$ and expanding, $\|X\|^2=\|\theta\|^2+\|X-\theta\|^2+2(X-\theta)\cdot\theta$, where the middle term has expectation $p$ and the cross term mean zero, so $E\|X\|^2=\|\theta\|^2+p$, and because the relative fluctuation of $\|X\|^2$ is of order $\sqrt p/p$ it concentrates as $\|X\|^2=\|\theta\|^2+p+O_p(\sqrt{p+\|\theta\|^2})$. The vector I observe is *systematically too long*: in high dimension I essentially know that $\|\theta\|^2\approx\|X\|^2-p$, so the truth lies on a sphere well inside the one where my observation sits, yet $\delta_0=X$ confidently keeps the full overshot length. That is a reason to pull in. The correction is invisible at $p=1,2$, where the overshoot of $1$ or $2$ is swamped by the $O(\sqrt p)$ noise — and indeed $X$ is provably admissible at $p=1$ and, by an information-inequality argument that forces any spherical dominator's $\psi=t\varphi$ to satisfy $\psi'\ge\psi^2/(4t)$ and hence vanish, at $p=2$ — but the overshoot $\approx p$ dominates the noise once $p\ge3$. So I propose to shrink, along the form of a spherically symmetric rule $\hat\theta(X)=(1-h(\|X\|^2))X$ that slides $X$ inward along its own ray, and I want the *clean* version with its *exact* risk, not the awkward offset-and-limit argument that only locates the right constant as the maximizer of a bracket.

The method is the James–Stein estimator,
$$\hat\theta^{\mathrm{JS}}(X)=\Big(1-\frac{(p-2)\sigma^2}{\|X\|^2}\Big)X,$$
which pulls every coordinate toward a common center (toward $0$ as written; toward a chosen $\mu_0$, apply it to $X-\mu_0$ and add $\mu_0$ back) by a single *data-driven* factor that couples the otherwise unrelated problems through the one joint statistic $\|X\|^2$ — precisely the statistic that measures the systematic overshoot. What makes this tractable, and what makes the strange constant $p-2$ inevitable, is an integration-by-parts identity for the normal. For $Z\sim N(0,1)$ with density $\phi$ and absolutely continuous $f$ with $E|f'(Z)|<\infty$, $E[f'(Z)]=\int f'\phi=-\int f\phi'=\int f(z)\,z\,\phi(z)\,dz=E[Zf(Z)]$, using the defining property $\phi'(z)=-z\phi(z)$ and the vanishing boundary term; this is Stein's lemma, $E[Zf(Z)]=E[f'(Z)]$. It is remarkable because it computes the covariance $E[(X-\mu)f(X)]$ — a quantity that ought to depend on the unknown $\mu$ — as the derivative $E[f'(X)]$ evaluated at the data without knowing $\mu$. Applied coordinatewise in $\mathbb R^p$ for $X\sim N(\mu,\sigma^2 I)$ it reads $\tfrac1{\sigma^2}\sum_i E[(X_i-\mu_i)f_i(X)]=E[\operatorname{div}f]$ with $\operatorname{div}f=\sum_i\partial f_i/\partial X_i$.

The payoff is that the risk of *any* estimator of the shrinkage shape becomes a single divergence calculation. Write $\hat\theta=X-g(X)$, so $g$ is the amount pulled in; then with $\sigma^2=1$,
$$R(\theta,\hat\theta)=E\|(X-\theta)-g\|^2=\underbrace{E\|X-\theta\|^2}_{p}-2\,E[(X-\theta)\cdot g]+E\|g\|^2=p-2E[\operatorname{div}g]+E\|g\|^2,$$
the middle step by Stein's lemma — an unbiased risk estimate (SURE) in which the bias has been integrated away entirely, with no asymptotics, no nuisance offset, and no hand-waved remainder. Plugging in the clean shrinkage $g(X)=c\,X/\|X\|^2$ gives $\|g\|^2=c^2/\|X\|^2$ for the price of shrinking, and for the divergence,
$$\frac{\partial}{\partial X_i}\frac{cX_i}{\|X\|^2}=\frac{c}{\|X\|^2}-\frac{2cX_i^2}{\|X\|^4}\ \Rightarrow\ \operatorname{div}g=\frac{cp}{\|X\|^2}-\frac{2c}{\|X\|^2}=\frac{c(p-2)}{\|X\|^2}.$$
There is the $p-2$, and it is neither assumed nor optimized into existence: the $p$ is the dimension from differentiating $p$ coordinates, and the $-2$ is the two derivatives that land on the $\|X\|^2$ in the denominator and fight back, leaving $p-2$ — manifestly $\le0$ for $p\le2$, which is exactly why those dimensions are safe. Assembling, $R(\theta,\hat\theta)=p-\big(2c(p-2)-c^2\big)E[1/\|X\|^2]$, and maximizing the improvement $2c(p-2)-c^2$ over $c$ gives $c=p-2$ with maximal improvement $(p-2)^2$, so
$$R(\theta,\hat\theta^{\mathrm{JS}})=p-(p-2)^2\,E_\theta\!\Big[\frac1{\|X\|^2}\Big]<p\quad\text{for all }\theta,$$
with $\sigma^2$ restored by taking $g=c\sigma^2 X/\|X\|^2$, which carries the scale through and yields $R=p\sigma^2-(p-2)^2\sigma^4 E[1/\|X\|^2]$. For $p\ge3$ the quantity $E[1/\|X\|^2]$ is finite and strictly positive, so the domination is strict for *every* $\theta$: the usual estimator is inadmissible. At $\theta=0$, $\|X\|^2/\sigma^2\sim\chi^2_p$ gives $E[1/\|X\|^2]=1/(\sigma^2(p-2))$ and hence risk exactly $2\sigma^2$ in every dimension $p\ge3$ — at $p=100$ that is a fiftyfold cut from $100$ to $2$ near the origin — while off the origin, writing $\|X\|^2$ as a noncentral chi-square with $K\sim\mathrm{Poisson}(\|\theta\|^2/2)$ gives $R=p-E[(p-2)^2/(p-2+2K)]$, interpolating from $2$ up toward $p$ as the truth runs far out, always strictly below $p$.

Three readings make the constant $p-2$ feel inevitable rather than lucky. It is the maximizer of the SURE risk reduction. It is also the stochastic correction of a geometric guess: the typical observation sits on a sphere of squared radius $\approx\|\theta\|^2+p$, for which the deterministic best pull-in factor would be $1-(p-1)/\|X\|^2$, but averaging the risk over the jiggle of $X$ around its typical point shaves $p-1$ down to $p-2$ — the lost unit being the price of $X$ being only approximately at its expected length, which is also why a careless geometric argument would wrongly suggest improvement at $p=2$. And it is the empirical-Bayes plug-in: under $\theta_i\sim N(0,A)$ the Bayes rule shrinks by the fixed factor $1/(A+1)$, and since the marginal makes $\|X\|^2\sim(A+1)\chi^2_p$, the quantity $(p-2)/\|X\|^2$ is an unbiased estimate of $1/(A+1)$, so James–Stein is exactly the Bayes shrink with its unknown factor estimated from $\|X\|^2$ — which is why one shrinks by a data-driven amount and not a fixed fraction, adapting to shrink hard when $\|X\|^2$ is small and barely at all when it is large. The prior is only scaffolding; the domination is a theorem about fixed, unrelated $\theta$ and needs no prior at all. The win lives entirely in the *sum*: shrinking introduces bias, so a coordinate with a large true mean is pulled away from the truth and its own risk can rise, and there is deliberately no claim that shrinkage helps each coordinate — the theorem says the summed variance reduction outweighs the summed squared bias, which is why the per-coordinate intuition resists (it asks the wrong question) and why the effect leans on the additivity of squared-error loss. One caveat the formula warns of: the factor $1-(p-2)\sigma^2/\|X\|^2$ goes negative when $\|X\|^2<(p-2)\sigma^2$, flipping the sign of $X$, so the positive-part refinement $\hat\theta^{\mathrm{JS+}}=\big(1-(p-2)\sigma^2/\|X\|^2\big)_+X$ clamps the factor at $0$ and dominates the plain rule, though it no longer satisfies the same exact risk formula.

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
