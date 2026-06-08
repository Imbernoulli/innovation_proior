# Stochastic approximation: the Robbins–Monro method

## Problem

A monotone function $M(x)$ is observable only through noise: at any level $x$ we may draw a random response $Y(x)$ with $\mathbb E[Y\mid x]=M(x)=\int y\,dH(y\mid x)$, but neither $M$ nor the response law $H$ is known. We want the root $\theta$ of
$$M(x)=\alpha,$$
choosing levels $x_1,x_2,\dots$ sequentially from past data so that $x_n\to\theta$.

## Key idea

Do not average many samples at each level to estimate $M$ and then root-find. Take **one** noisy measurement and a **small step against it**, with a step size that shrinks over time:
$$\boxed{\,x_{n+1}=x_n+a_n(\alpha-y_n)\,},\qquad y_n\sim H(\cdot\mid x_n),$$
where the gains satisfy the two **Robbins–Monro conditions**
$$\sum_{n=1}^{\infty}a_n=\infty\qquad\text{(reach the root from anywhere; the drift cannot stall),}$$
$$\sum_{n=1}^{\infty}a_n^2<\infty\qquad\text{(total injected noise is finite; the iterate settles).}$$
The canonical choice is $a_n=a/n$. Each observation does double duty: its mean $M(x_n)-\alpha$ steers toward $\theta$ (because $M$ is monotone), and its noise averages out across steps weighted by the shrinking $a_n$. This is the ancestor of stochastic gradient descent: replacing $\alpha-y_n$ by a negative noisy gradient gives SGD.

## Convergence theorem and proof

**Assumptions.** $M$ nondecreasing with unique root $M(\theta)=\alpha$; responses bounded, $\Pr[|Y(x)|\le C]=1$ for all $x$; gains $a_n>0$ with $\sum a_n^2<\infty$ and $\sum a_n=\infty$; and a strict non-degeneracy at the root — either the sharp form $M(x)\le\alpha-\delta$ for $x<\theta$, $M(x)\ge\alpha+\delta$ for $x>\theta$, or the smooth form $M'(\theta)>0$.

**Theorem.** $b_n:=\mathbb E[(x_n-\theta)^2]\to0$. Hence $x_n\to\theta$ in quadratic mean and in probability (and almost surely, via the supermartingale argument below).

**Proof (the $b_n$ recursion).** Expand using $x_{n+1}-x_n=a_n(\alpha-y_n)$ and condition on $x_n$ (so $\mathbb E[y_n\mid x_n]=M(x_n)$). With
$$d_n=\mathbb E[(x_n-\theta)(M(x_n)-\alpha)],\qquad e_n=\mathbb E[(\alpha-y_n)^2],$$
$$b_{n+1}-b_n=a_n^2\,e_n-2a_n\,d_n.\tag{$\star$}$$

*Sign of the drift.* $M$ nondecreasing with $M(\theta)=\alpha$ gives $(x-\theta)(M(x)-\alpha)\ge0$ pointwise (both factors share sign), so $d_n\ge0$: the $-2a_nd_n$ term always decreases $b_n$.

*Noise is finite.* Boundedness gives $e_n\le(C+|\alpha|)^2$, so $\sum_n a_n^2 e_n\le(C+|\alpha|)^2\sum_n a_n^2<\infty$.

*Summation.* Summing $(\star)$, and using $b_{n+1}\ge0$,
$$0\le b_{n+1}=b_1+\sum_{j\le n}a_j^2e_j-2\sum_{j\le n}a_jd_j\ \Rightarrow\ \sum_{j}a_jd_j\le\tfrac12\Big(b_1+\sum_j a_j^2e_j\Big)<\infty,$$
so the positive series $\sum a_nd_n$ converges and $\lim_n b_n=b\ge0$ **exists**.

*The limit is zero.* Let $A_n=|x_1-\theta|+(C+|\alpha|)\sum_{j<n}a_j$ be the a.s. range of $x_n$ and $k_n=\inf_{0<|x-\theta|\le A_n}\frac{M(x)-\alpha}{x-\theta}\ge0$ the worst-case slope on it. Then $(x-\theta)(M(x)-\alpha)\ge k_n(x-\theta)^2$, so $d_n\ge k_nb_n$. Under the non-degeneracy, $k_n\ge K/A_n$ for some $K>0$ (sharp form: $K=\delta$; smooth form: $K=\tfrac12\delta_0M'(\theta)$). Since $A_n$ grows like $\sum_{j<n}a_j$,
$$\sum_n a_nk_n\ge K\sum_n\frac{a_n}{A_n}=\infty,$$
because $A_n=|x_1-\theta|+(C+|\alpha|)\sum_{j<n}a_j$ and $\sum a_n=\infty$ makes $\sum a_n/A_n$ diverge.
Now $\sum a_nd_n<\infty$ and $d_n\ge k_nb_n\ge0$ give $\sum a_nk_nb_n<\infty$; with $\sum a_nk_n=\infty$ this forces $b_n<\varepsilon$ infinitely often for every $\varepsilon$. A convergent sequence that drops below every $\varepsilon$ infinitely often converges to $0$, so $b=0$. $\qquad\blacksquare$

**Almost-sure version (supermartingale / Lyapunov).** With potential $V(x)=(x-\theta)^2$,
$$\mathbb E[V(x_{n+1})\mid\mathcal F_n]=V(x_n)-\underbrace{2a_n(x_n-\theta)(M(x_n)-\alpha)}_{\psi_n\ge0}+\underbrace{a_n^2\,\mathbb E[(\alpha-y_n)^2\mid x_n]}_{\le\,a_n^2(C+|\alpha|)^2}.$$
This matches the **Robbins–Siegmund almost-supermartingale theorem**: for nonnegative $Z_n$ with $\mathbb E[Z_{n+1}\mid\mathcal F_n]\le(1+\eta_n)Z_n+\gamma_n-\psi_n$ and $\sum\eta_n,\sum\gamma_n<\infty$ a.s., the limit $\lim_n Z_n$ exists finite a.s. and $\sum_n\psi_n<\infty$ a.s. Under the bounded-response assumption, take $\eta_n=0$ and $\gamma_n=a_n^2(C+|\alpha|)^2$, summable because $\sum a_n^2<\infty$. So $V(x_n)\to V_\infty$ a.s. and $\sum_n a_n(x_n-\theta)(M(x_n)-\alpha)<\infty$ a.s.; the same slope-lower-bound-plus-$\sum a_n=\infty$ argument forces $V_\infty=0$, i.e. $x_n\to\theta$ **almost surely**. The two conditions split cleanly: $\sum a_n^2<\infty$ makes the squared-error potential converge; $\sum a_n=\infty$ makes it converge **at the root**.

## Gradient-free variant (Kiefer–Wolfowitz): maximizing a noisily-observed function

To reach the maximizer $\theta$ of $M$ (where $M'(\theta)=0$) from noisy **values** only, compare two probes at $z_n\pm c_n$ and step toward the larger side:
$$z_{n+1}=z_n+a_n\,\frac{y_{2n}-y_{2n-1}}{c_n},\qquad y_{2n}\sim H(\cdot\mid z_n+c_n),\ y_{2n-1}\sim H(\cdot\mid z_n-c_n)\ \text{independent}.$$
Conditions:
$$c_n\to0,\qquad\sum a_n=\infty,\qquad\sum a_nc_n<\infty\ \text{(finite-difference bias controlled)},\qquad\sum\frac{a_n^2}{c_n^2}<\infty\ \text{(noise inflated by }1/c_n^2\text{ still summable)}.$$
Canonical balance $a_n=1/n$, $c_n=n^{-1/3}$ (both the bias tail $\sum a_nc_n$ and the variance tail $\sum a_n^2/c_n^2$ become $\sum n^{-4/3}<\infty$). Under regularity on $M$ (Lipschitz-type upper bound on the slope; strictly positive slope away from $\theta$), $z_n\to\theta$ in probability. The $c_n$ is a second bias–variance knob: small $c_n$ shrinks the difference-quotient bias but inflates its variance.

## Code

```python
import numpy as np

# --- root of M(x)=alpha, M monotone, observed via noisy y, E[y|x]=M(x) ---
def root_find(observe, alpha, x0, n_steps, a=1.0):
    x = x0
    for n in range(1, n_steps + 1):
        y = observe(x)                 # one noisy measurement at the current level
        a_n = a / n                    # sum a_n^2 < inf (settle), sum a_n = inf (reach + no stall)
        x = x + a_n * (alpha - y)      # small step against it; mean steers, noise averages out
    return x

# Quantile / bioassay: M(x)=F(x), y in {0,1}, Pr[y=1|x]=F(x); distribution-free.
def estimate_quantile(observe_bernoulli, alpha, x0, n_steps, a=1.0):
    return root_find(observe_bernoulli, alpha, x0, n_steps, a)

# --- maximize M from noisy values by comparing two nearby probes ---
def maximize(observe, z0, n_steps, a=1.0):
    z = z0
    for n in range(1, n_steps + 1):
        a_n = a / n
        c_n = n ** (-1.0 / 3.0)        # both bias and variance tails ~ sum n^{-4/3} < inf
        y_plus, y_minus = observe(z + c_n), observe(z - c_n)
        z = z + a_n * (y_plus - y_minus) / c_n  # step toward the larger probe value
    return z
```
