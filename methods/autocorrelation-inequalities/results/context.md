# Context: extremal constants for autocorrelation inequalities

## Research question

For a nonnegative function $f$ on the real line, the *autocorrelation* $g(t)=\int_\R f(x)f(x+t)\,dx$ measures how much $f$ overlaps a shifted copy of itself. A family of problems in additive combinatorics asks how large this overlap must be — how concentrated the autocorrelation has to stay — relative to simple size measurements of $f$. The cleanest version fixes a symmetric, decreasing weight $w$ (with $\|w\|_1=\|w\|_\infty=1$) and asks for the smallest constant $C_{opt}(w)$ such that

$$\iint_{\R^2} f(x)f(y)\,w(x-y)\,dx\,dy \;\le\; C_{opt}(w)\,\|f\|_1\,\|f\|_2 \qquad \text{for all } f\in L^1(\R)\cap L^2(\R).$$

Two weights are of primary interest. Taking $w=\chi_{[-1/2,1/2]}$ turns the left side into the **average of the autocorrelation over a unit window**, $\int_{-1/2}^{1/2}\!\int_\R f(x)f(x+t)\,dx\,dt$; this is the average-autocorrelation inequality. Taking $w=e^{-\pi x^2}$ replaces the hard window by a Gaussian mean. In both cases $C_{opt}(w)$ is the supremum of a Rayleigh-type quotient over an infinite-dimensional cone of functions.

The quantity $C_{opt}(w)$ is the supremum of a ratio, so any explicit nonnegative test function $f$ produces a *lower* bound $C_{opt}\ge$ (its quotient), while analytic inequalities produce *upper* bounds. For the indicator weight, a known example gives $\ge 0.8$ and analytic upper bounds stand at $0.91$ and then $0.864$. The question is to determine $C_{opt}(w)$ for these weights.

## Background

The motivating thread comes from generalized Sidon sets. Cilleruelo, Ruzsa and Vinuesa (CRV, *Advances in Mathematics* 2010) connected the asymptotic size of $B_h[g]$ (generalized Sidon) sets to the best constant $c_{\max}$ in a *maximum* autocorrelation inequality: the largest universal $c$ with

$$\max_{-1/2\le t\le 1/2}\int_\R f(t-x)f(x)\,dx \;\ge\; c\Big(\int_{-1/4}^{1/4} f(x)\,dx\Big)^2$$

for all nonnegative $f$ supported in $[-1/4,1/4]$. A line of work (Cilleruelo–Ruzsa–Trujillo; Green; Martin–O'Bryant; Yu; Matolcsi–Vinuesa; Cloninger–Steinerberger) pushed this constant, reaching $c_{\max}\ge 1.28$ from below (Cloninger–Steinerberger 2017) and $c_{\max}\le 1.52$ from above (Matolcsi–Vinuesa 2010). The numerical extremizers found for this maximum problem are non-smooth and not symmetric-decreasing.

The relevant structural facts about these problems are classical. The bilinear form $\iint f(x)f(y)w(x-y)$ is a quadratic form $\langle f, K_w f\rangle$ in the convolution operator $K_w$ with kernel $w$; on the Fourier side it equals $\int |\hat f(\xi)|^2\,\hat w(\xi)\,d\xi$. For $w=\chi_{[-1/2,1/2]}$ this is $\int |\hat f(\xi)|^2 \frac{\sin\pi\xi}{\pi\xi}\,d\xi$, so the whole question is a weighted average of $|\hat f|^2$ against a sinc. Among the classical inequalities in the standard analytic toolbox for such forms are the **Riesz rearrangement inequality** (for nonnegative $f$ and a symmetric-decreasing kernel $w$, replacing $f$ by its symmetric-decreasing rearrangement $f^*$ never decreases $\iint f(x)f(y)w(x-y)$ while preserving $\|f\|_1$ and $\|f\|_2$) and the **sharp Hausdorff–Young (Beckner) inequality**, $\|\hat f\|_{q}\le A_p\|f\|_p$ with the optimal Babenko–Beckner constant, which trades an $L^q$ norm of $\hat f$ for $L^p$ norms of $f$.

A related fact from the minimum version of the autocorrelation problem is that $g(t)=\int f(x)f(x+t)\,dx$ has nonnegative Fourier transform, $\hat g(\xi)=|\hat f(\xi)|^2\ge0$. If $g$ is forced to stay above a window, writing the excess as $p\ge0$ forces $\int p\ge -\inf_x \frac{\sin x}{x}=0.217\ldots$, a Fourier-positivity constraint on the autocorrelation. The best-known extremizers of the maximum problem are non-smooth and not monotone.

## Baselines

**Barnard–Steinerberger (2020, *Journal of Number Theory* 207).** They proposed the average problem as a relaxation of CRV's maximum problem and proved the first clean bound,

$$\int_{-1/2}^{1/2}\!\int_\R f(x)f(x+t)\,dx\,dt \le 0.91\,\|f\|_1\|f\|_2,$$

by normalizing the left side, using Wiener–Khinchine to write it as $\int |\hat f(\xi)|^2\frac{\sin\pi\xi}{\pi\xi}\,d\xi$, rearranging the positive part of the sinc weight, bounding $|\hat f|\le\|f\|_1$, and using Plancherel to control the interval on which $\hat f$ can stay large. They also exhibited an explicit example forcing the constant to be $\ge 0.8$.

**Madrid–Ramos (2021, *Communications on Pure and Applied Analysis* 20(1)).** They improved the upper bound to $0.864$ by recasting the problem dually through the Fourier transform and applying the **sharp Hausdorff–Young inequality**: for each $p>1$,
$$\int_{-1/2}^{1/2}\!\int_\R f f \le \frac{(2p)^{1/p}(p-1)^{(p-1)/2p}}{(p+1)^{(p+1)/2p}}\Big(\textstyle\int|\frac{\sin\pi\xi}{\pi\xi}|^p d\xi\Big)^{1/p}\|f\|_1^{2/p}\|f\|_2^{2-2/p},$$
then optimizing the resulting $C_p$ over $p\ge2$ to get $0.864$. The same scheme handles the Gaussian mean, giving $(8a/27\pi)^{1/4}$ for weight $\propto e^{-at^2}$. They further proved, by functional-analytic compactness in a restricted class of functions, that extremizers exist there, and **conjectured** that extremizers exist in full generality and can be taken compactly supported.

**Cloninger–Steinerberger (2017) and Matolcsi–Vinuesa (2010), on the maximum problem.** These attack $c_{\max}$ computationally and analytically. The maximum problem's extremizers are non-smooth and not symmetric-decreasing; the discretization error of their numerical scheme is first order, and its cost grows exponentially in the discretization level.

## Evaluation settings

The natural yardstick is the pair of weights $w=\chi_{[-1/2,1/2]}$ (the original average/Barnard–Steinerberger problem) and $w=e^{-\pi x^2}$ (the Gaussian mean, equivalently the Madrid–Ramos $e^{-at^2}$ family after a change of variables that sets $a=\pi$). The object measured is the single scalar $C_{opt}(w)$: the smallest constant in the inequality, equivalently the supremum of $\iint f f\,w \,/\, (\|f\|_1\|f\|_2)$ over nonnegative $f\in L^1\cap L^2$. The quality metrics are the gap between the upper and lower bounds on $C_{opt}$ and the computational scaling in the discretization resolution $1/\delta$.

## Code framework

Existing primitives: numerical integration (`scipy.integrate.quad`), dense linear algebra (`numpy.linalg` for matrix factorizations, symmetric eigensolves, and matrix inverses), and array/convolution machinery (`numpy`, `scipy.signal`/`scipy.linalg` for Toeplitz structure). The base objects are a symmetric decreasing weight, a discretization scale, and a finite grid of cells.

```python
import numpy as np
import scipy.integrate as si

def indicator_weight(t):
    return 1.0 if abs(t) <= 0.5 else 0.0

def gaussian_weight(t):
    return np.exp(-np.pi * t * t)

def estimate_constant(w, delta, support_radius, options):
    """Return upper and lower bounds on
        C_opt(w) = sup_{f >= 0} <f, K_w f> / (||f||_1 ||f||_2)
    over nonnegative cell functions of cell width delta.
    """
    pass  # TODO

def main():
    delta = 1e-3
    options = {}
    for w in [indicator_weight, gaussian_weight]:
        lo, hi = estimate_constant(w, delta, support_radius=4.0,
                                   options=options)
        pass  # TODO
```
