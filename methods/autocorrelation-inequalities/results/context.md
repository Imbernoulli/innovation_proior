# Context: extremal constants for autocorrelation inequalities

## Research question

For a nonnegative function $f$ on the real line, the *autocorrelation* $g(t)=\int_\R f(x)f(x+t)\,dx$ measures how much $f$ overlaps a shifted copy of itself. A family of problems in additive combinatorics asks how large this overlap must be — how concentrated the autocorrelation has to stay — relative to simple size measurements of $f$. The cleanest version fixes a symmetric, decreasing weight $w$ (with $\|w\|_1=\|w\|_\infty=1$) and asks for the smallest constant $C_{opt}(w)$ such that

$$\iint_{\R^2} f(x)f(y)\,w(x-y)\,dx\,dy \;\le\; C_{opt}(w)\,\|f\|_1\,\|f\|_2 \qquad \text{for all } f\in L^1(\R)\cap L^2(\R).$$

Two weights are of primary interest. Taking $w=\chi_{[-1/2,1/2]}$ turns the left side into the **average of the autocorrelation over a unit window**, $\int_{-1/2}^{1/2}\!\int_\R f(x)f(x+t)\,dx\,dt$; this is the average-autocorrelation inequality. Taking $w=e^{-\pi x^2}$ replaces the hard window by a Gaussian mean. In both cases $C_{opt}(w)$ is the supremum of a Rayleigh-type quotient over an infinite-dimensional cone of functions.

The pain point is sharpness. The quantity $C_{opt}(w)$ is the supremum of a ratio, so any explicit nonnegative test function $f$ produces a *lower* bound $C_{opt}\ge$ (its quotient), and clever analytic inequalities produce *upper* bounds; the two have stubbornly refused to meet. For the indicator weight the spread sat between an example giving $0.8$ and analytic upper bounds of $0.91$ and then $0.864$. A satisfactory answer would (i) prove that an extremizer exists at all and has nice structure, (ii) pin $C_{opt}$ between an upper and a lower bound that nearly coincide, and (iii) carry a certificate so the pinning is rigorous rather than a numerical impression — all at a computational cost that does not explode as the resolution increases.

## Background

The motivating thread comes from generalized Sidon sets. Cilleruelo, Ruzsa and Vinuesa (CRV, *Advances in Mathematics* 2010) connected the asymptotic size of $B_h[g]$ (generalized Sidon) sets to the best constant $c_{\max}$ in a *maximum* autocorrelation inequality: the largest universal $c$ with

$$\max_{-1/2\le t\le 1/2}\int_\R f(t-x)f(x)\,dx \;\ge\; c\Big(\int_{-1/4}^{1/4} f(x)\,dx\Big)^2$$

for all nonnegative $f$ supported in $[-1/4,1/4]$. A line of work (Cilleruelo–Ruzsa–Trujillo; Green; Martin–O'Bryant; Yu; Matolcsi–Vinuesa; Cloninger–Steinerberger) pushed this constant, reaching $c_{\max}\ge 1.28$ from below (Cloninger–Steinerberger 2017) and $c_{\max}\le 1.52$ from above (Matolcsi–Vinuesa 2010), but the gap stayed wide and the numerical extremizers came out non-smooth and not symmetric-decreasing.

The relevant structural facts about these problems are classical. The bilinear form $\iint f(x)f(y)w(x-y)$ is a quadratic form $\langle f, K_w f\rangle$ in the convolution operator $K_w$ with kernel $w$; on the Fourier side it equals $\int |\hat f(\xi)|^2\,\hat w(\xi)\,d\xi$. For $w=\chi_{[-1/2,1/2]}$ this is $\int |\hat f(\xi)|^2 \frac{\sin\pi\xi}{\pi\xi}\,d\xi$, so the whole question is a weighted average of $|\hat f|^2$ against a sinc. Two further tools are load-bearing. First, the **Riesz rearrangement inequality**: for nonnegative $f$ and a symmetric-decreasing kernel $w$, replacing $f$ by its symmetric-decreasing rearrangement $f^*$ never decreases $\iint f(x)f(y)w(x-y)$ while preserving $\|f\|_1$ and $\|f\|_2$. Second, the **sharp Hausdorff–Young (Beckner) inequality**, $\|\hat f\|_{q}\le A_p\|f\|_p$ with the optimal Babenko–Beckner constant, which lets one trade an $L^q$ norm of $\hat f$ for $L^p$ norms of $f$.

A related diagnostic from the minimum version of the autocorrelation problem is that $g(t)=\int f(x)f(x+t)\,dx$ has nonnegative Fourier transform, $\hat g(\xi)=|\hat f(\xi)|^2\ge0$. If $g$ is forced to stay above a window, writing the excess as $p\ge0$ forces $\int p\ge -\inf_x \frac{\sin x}{x}=0.217\ldots$ This is a useful warning: Fourier-positivity constraints make autocorrelation problems more rigid than their pointwise formulation first suggests. A second diagnostic, from the numerical record on the maximum problem, is that its best-known extremizers are visibly non-smooth and not monotone, so any discretization of *that* problem cannot expect better than first-order accuracy.

## Baselines

**Barnard–Steinerberger (2020, *Journal of Number Theory* 207).** They proposed the average problem as a relaxation of CRV's maximum problem and proved the first clean bound,

$$\int_{-1/2}^{1/2}\!\int_\R f(x)f(x+t)\,dx\,dt \le 0.91\,\|f\|_1\|f\|_2,$$

by normalizing the left side, using Wiener–Khinchine to write it as $\int |\hat f(\xi)|^2\frac{\sin\pi\xi}{\pi\xi}\,d\xi$, rearranging the positive part of the sinc weight, bounding $|\hat f|\le\|f\|_1$, and using Plancherel to control the interval on which $\hat f$ can stay large. They also exhibited an explicit example forcing the constant to be $\ge 0.8$. Gap: the $0.91$–$0.8$ window is wide, the method gives no existence/uniqueness of an extremizer, and there is no certificate that either end is close to truth.

**Madrid–Ramos (2021, *Communications on Pure and Applied Analysis* 20(1)).** They improved the upper bound to $0.864$ by recasting the problem dually through the Fourier transform and applying the **sharp Hausdorff–Young inequality**: for each $p>1$,
$$\int_{-1/2}^{1/2}\!\int_\R f f \le \frac{(2p)^{1/p}(p-1)^{(p-1)/2p}}{(p+1)^{(p+1)/2p}}\Big(\textstyle\int|\frac{\sin\pi\xi}{\pi\xi}|^p d\xi\Big)^{1/p}\|f\|_1^{2/p}\|f\|_2^{2-2/p},$$
then optimizing the resulting $C_p$ over $p\ge2$ to get $0.864$. The same scheme handles the Gaussian mean, giving $(8a/27\pi)^{1/4}$ for weight $\propto e^{-at^2}$. They further proved, by functional-analytic compactness in a restricted class of functions, that extremizers exist there, and **conjectured** that extremizers exist in full generality and can be taken compactly supported. Gap: the upper bound $0.864$ is still not matched by any lower bound; the dual Hausdorff–Young route is inherently one-sided (it produces upper bounds only) and stops improving once $p$ is optimized; and the existence result is conditional on an artificial function class.

**Cloninger–Steinerberger (2017) and Matolcsi–Vinuesa (2010), on the maximum problem.** These attack $c_{\max}$ computationally and analytically. Their relevance here is as a cautionary baseline: the maximum problem's extremizers are non-smooth and not symmetric-decreasing, the discretization error is only first order, and the proposed numerical scheme's cost grows *exponentially* in the discretization level. Any method for the average problem wants to avoid exactly these failure modes.

## Evaluation settings

The natural yardstick is the pair of weights $w=\chi_{[-1/2,1/2]}$ (the original average/Barnard–Steinerberger problem) and $w=e^{-\pi x^2}$ (the Gaussian mean, equivalently the Madrid–Ramos $e^{-at^2}$ family after a change of variables that sets $a=\pi$). The object measured is the single scalar $C_{opt}(w)$: the smallest constant in the inequality, equivalently the supremum of $\iint f f\,w \,/\, (\|f\|_1\|f\|_2)$ over nonnegative $f\in L^1\cap L^2$. The quality metrics are the width of the certified bracket (upper minus lower bound on $C_{opt}$) and the computational scaling in the discretization resolution $1/\delta$.

## Code framework

Existing primitives: numerical integration (`scipy.integrate.quad`), dense linear algebra (`numpy.linalg` for matrix factorizations, symmetric eigensolves, and matrix inverses), and array/convolution machinery (`numpy`, `scipy.signal`/`scipy.linalg` for Toeplitz structure). The base objects are a symmetric decreasing weight, a discretization scale, a finite grid of cells, and a one-dimensional search grid. The finite search routine returns a feasible vector and an upper certificate.

```python
import numpy as np
import scipy.integrate as si

def indicator_weight(t):
    return 1.0 if abs(t) <= 0.5 else 0.0

def gaussian_weight(t):
    return np.exp(-np.pi * t * t)

def prepare_weight_grid(w, delta, max_offset):
    """Represent the weight on a finite cell grid."""
    pass  # TODO

def build_operator(weight_data, n_cells):
    """Assemble the finite convolution operator on n cells."""
    pass  # TODO

def prepare_search_form(search_value, n_cells, delta):
    """Prepare the finite-dimensional quadratic form used in the search."""
    pass  # TODO

def leading_symmetric_eigenpair(matrix, options):
    """Return the leading symmetric eigenpair using available linear algebra."""
    pass  # TODO

def solve_finite_problem(operator, search_value, delta, n_cells, options):
    """Solve one finite-dimensional search problem."""
    pass  # TODO

def evaluate_test_function(f_vec, operator, start, delta):
    """Evaluate the scale-invariant quotient for a nonnegative cell function."""
    pass  # TODO

def make_upper_certificate(cell_value, search_value, delta):
    """Convert one finite search value into an upper certificate."""
    pass  # TODO

def estimate_constant(w, delta, support_radius, search_grid, options):
    """Return upper and lower bounds on
        C_opt(w) = sup_{f >= 0} <f, K_w f> / (||f||_1 ||f||_2)
    over nonnegative cell functions.
    """
    pass  # TODO

def main():
    delta = 1e-3
    options = {}
    search_grid = np.linspace(0.25, 4.0, 100)
    for w in [indicator_weight, gaussian_weight]:
        lo, hi = estimate_constant(w, delta, support_radius=4.0,
                                   search_grid=search_grid,
                                   options=options)
        pass  # TODO
```
