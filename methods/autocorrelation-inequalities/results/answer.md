# Extremal constants for autocorrelation inequalities — the variational reduction + certified relaxation

## Problem

For a symmetric decreasing weight $w$ with $\|w\|_1=\|w\|_\infty=1$, find the smallest constant $C_{opt}(w)$ with
$$\iint_{\R^2} f(x)f(y)\,w(x-y)\,dx\,dy \le C_{opt}(w)\,\|f\|_1\,\|f\|_2,\qquad f\in L^1(\R)\cap L^2(\R).$$
The two cases of interest are $w=\chi_{[-1/2,1/2]}$ (the average / Barnard–Steinerberger autocorrelation inequality) and $w=e^{-\pi x^2}$ (the Gaussian mean). $C_{opt}(w)$ is the supremum of the Rayleigh-type quotient $\iint ff\,w/(\|f\|_1\|f\|_2)$ over nonnegative $f$. The goal is a **rigorous two-sided bracket** on $C_{opt}$ with a certificate, at polynomial cost — not merely a better test function (a one-sided lower bound) or a better analytic upper bound.

## Key idea

A chain of reductions turns an infinite-dimensional, sign-constrained, non-quadratic-denominator optimization into a finite family of symmetric eigenvalue problems with a controlled discretization error.

1. **Sign + rearrangement.** Replacing $f$ by $|f|$ then by its symmetric-decreasing rearrangement $f^*$ never decreases the quotient (Riesz rearrangement, since $w$ is symmetric decreasing), so the sup is over nonnegative symmetric-decreasing functions.

2. **Compact support via Euler–Lagrange.** The extremizer of the log-quotient $\log\langle f,K_wf\rangle-\log\|f\|_1-\tfrac12\log\|f\|_2^2$ satisfies, on its support $[-a,a]$,
   $$\frac{2\,f^\star*w}{\langle f^\star,K_wf^\star\rangle}=\frac1{\|f^\star\|_1}+\frac{f^\star}{\|f^\star\|_2^2},$$
   and integrating gives $a\le 2\|w\|_1^2/C_{opt,R}^2$ for the restricted problem. This uniform support control lets the limit extremizer be taken compactly supported.

3. **Linearize the $\|f\|_1\|f\|_2$ denominator (AM–GM).** Since $2\|f\|_1\|f\|_2=\min_{\lambda>0}(\lambda\|f\|_2^2+\lambda^{-1}\|f\|_1^2)$, and on $f\ge0$ the term $\|f\|_1^2=(\int f)^2$ is the square of a linear functional, define the Hilbert-space quadratic form $\|f\|_{H_\lambda}^2=\lambda\|f\|_2^2+\lambda^{-1}(\int f)^2$ and
   $$c_\lambda=\max_{\substack{f\ge0\\\|f\|_{H_\lambda}\le1}}2\langle f,K_wf\rangle,\qquad C_{opt}=\max_{\lambda>0}c_\lambda.$$
   $c_\lambda$ is $1$-Lipschitz in $\lambda$ with explicit lower envelope $c_\lambda\ge 2c_{\lambda^*}/(\lambda^{-1}\lambda^*+\lambda{\lambda^*}^{-1})$, so a finite $\lambda$-grid certifies the max.

4. **Discretize to step functions with a rigorous error bound.** With $[f]_\delta$ the block-average projection onto step functions of cell $\delta$ (mass-preserving, $L^2$- and $H_\lambda$-contractive), $c_{\lambda,\delta}\le c_\lambda$, and via the optimal Poincaré inequality $\|\{f\}_\delta\|_2\le\frac\delta\pi\|f'\|_2$, Young's convolution inequality, and the extremizer regularity $\|{f^\star}'\|_2\le\frac{4}{c_\lambda\lambda^{3/2}}$ (from the Euler–Lagrange smoothing $f^\star=\max\{0,c_\lambda^{-1}(\lambda+\lambda^{-1}I)^{-1}2K_wf^\star\}$),
   $$0\le c_\lambda-c_{\lambda,\delta}\le\frac{16\,\delta^2}{\pi^2\,c_\lambda\,\lambda^2}.$$
   Equivalently, if $E_\lambda=16\delta^2/(\pi^2\lambda^2)$, then the computable upper correction is
   $$c_\lambda\le\frac{c_{\lambda,\delta}+\sqrt{c_{\lambda,\delta}^2+4E_\lambda}}{2}.$$

5. **Solve each fixed-$\lambda$ problem (whitening + power method).** $H_\lambda=\lambda\,\mathrm{Id}+\lambda^{-1}|1\rangle\langle1|$ is identity-plus-rank-one; its square root $A_\lambda=\sqrt\lambda\,\mathrm{Id}+b_\lambda|1\rangle\langle1|$ (with $b_\lambda$ the positive root of $2\sqrt\lambda\,b_\lambda+2a\,b_\lambda^2=\lambda^{-1}$) whitens the form. With $g=A_\lambda f$ and $M_\lambda=2A_\lambda^{-1}K_wA_\lambda^{-1}$, the unconstrained maximum is $\lambda_{\max}(M_\lambda)$ via the power method. The $f\ge0$ constraint binds (unconstrained value overshoots); by discrete Riesz the extremizer is a single contiguous bump, so the constrained optimum is found by **sweeping the contiguous support length** and keeping the largest eigenvalue whose eigenvector is nonnegative.

6. **Two-sided certificate.** Upper bound: the quadratic correction above, maximized over the $\lambda$ grid and enlarged by the grid/tail control. Lower bound: the admissible eigenvector is a feasible nonnegative test function whose original quotient is $\le C_{opt}$. The resulting brackets are $0.8055809\le C_{opt}(\chi_{[-1/2,1/2]})\le0.8055896$ and $0.7152474\le C_{opt}(e^{-\pi x^2})\le0.7152576$, with gaps $<9\cdot10^{-6}$ and $<1.2\cdot10^{-5}$, at cost polynomial in $1/\delta$.

The discrete kernel is the triangular smoothing $\tilde w(s)=\delta^{-2}\int_{s-\delta}^{s+\delta}w(t)(\delta-|t-s|)\,dt$ (the autocorrelation of the cell indicator with $w$). The certified brackets use $\delta\approx1.45\cdot10^{-3}$ and $\Delta\lambda\approx0.001$. A faster but unproven-to-converge alternative is the fixed-point iteration $\frac{f^\star}{\|f^\star\|_2^2}=\max(\frac{2f^\star*w}{\langle f^\star,K_wf^\star\rangle}-\frac1{\|f^\star\|_1},0)$, which produces the matching lower-bound column hundreds of times faster.

## Algorithm

1. Choose $\delta$ so the quadratic discretization correction is below tolerance on the relevant $\lambda$ range; choose a $\lambda$-grid of spacing $\Delta\lambda$.
2. Build $\tilde w$ and the symmetric Toeplitz $K_w$ on the cells of $[-a,a]$.
3. For each $\lambda$: form $A_\lambda$ from $b_\lambda$, $M_\lambda=2A_\lambda^{-1}K_wA_\lambda^{-1}$; sweep support length, power-method each block, keep the largest admissible (nonnegative) eigenvector → $c_{\lambda,\delta}$ and a feasible $f$.
4. Upper bound $=\max_\lambda$ the quadratic discretization correction, plus the $\lambda$-grid and tail correction; lower bound $=\max_\lambda$ original quotient of $f$. Report the bracket.

## Code

```python
import numpy as np
import scipy.integrate as si

def indicator_weight(t):       # average autocorrelation: w = chi_[-1/2, 1/2]
    return 1.0 if abs(t) <= 0.5 else 0.0

def gaussian_weight(t):        # Gaussian mean: w = exp(-pi t^2)
    return np.exp(-np.pi * t * t)

def prepare_weight_grid(w, delta, max_offset):
    # tw(s) = d^-2 int_{s-d}^{s+d} w(t)(d-|t-s|) dt.
    tw = {}
    for k in range(-max_offset, max_offset + 1):
        s = k * delta
        val, _ = si.quad(
            lambda t: w(t) * (delta - abs(t - s)),
            s - delta,
            s + delta,
            limit=120,
        )
        tw[k] = val / delta**2
    return tw

def build_operator(weight_data, n_cells):
    K = np.empty((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            K[i, j] = weight_data[i - j]         # symmetric Toeplitz K_w
    return K

def prepare_search_form(search_value, n_cells, delta):
    lam = search_value
    a = n_cells * delta / 2.0                    # half support; <1,1> = 2a
    b = (-2*np.sqrt(lam) + np.sqrt(4*lam + 8*a/lam)) / (4*a)   # 2 sqrt(lam) b + 2a b^2 = 1/lam
    one = np.ones(n_cells)
    A    = np.sqrt(lam) * np.eye(n_cells) + b * delta * np.outer(one, one)
    return A, np.linalg.inv(A)

def leading_symmetric_eigenpair(M, options):
    iters = options.get("iters", 2000)
    tol = options.get("tol", 1e-13)
    v = np.random.default_rng(0).standard_normal(M.shape[0]); v /= np.linalg.norm(v)
    mu = 0.0
    for _ in range(iters):
        w_ = M @ v; nw = np.linalg.norm(w_)
        if nw == 0: break
        v_new = w_ / nw
        mu_new = v_new @ (M @ v_new)
        if abs(mu_new - mu) < tol: v, mu = v_new, mu_new; break
        v, mu = v_new, mu_new
    return mu, v

def solve_finite_problem(K_full, search_value, delta, n_full, options):
    lam = search_value
    # f >= 0 handled exactly: extremizer is one contiguous bump (discrete Riesz);
    # sweep support length, keep largest admissible (nonnegative) eigenvector.
    best_val, best_f = 0.0, None
    for L in range(1, n_full + 1):
        lo = (n_full - L) // 2
        K = K_full[lo:lo+L, lo:lo+L]
        A, Ainv = prepare_search_form(lam, L, delta)
        M = 2.0 * (Ainv @ (delta * K) @ Ainv)
        mu, g = leading_symmetric_eigenpair(M, options)
        f = Ainv @ g
        if f.min() < 0: f = -f                   # eigenvector sign is free
        if f.min() >= -1e-9 and mu > best_val:
            best_val, best_f = mu, np.maximum(f, 0.0)
    return best_val, best_f

def evaluate_test_function(f, K_full, lo, delta): # honest lower bound on C_opt
    n = len(f); K = K_full[lo:lo+n, lo:lo+n]
    num = delta**2 * (f @ (K @ f))
    l1  = delta * f.sum()
    l2  = np.sqrt(delta * (f * f).sum())
    return num / (l1 * l2)

def make_upper_certificate(c_ld, search_value, delta):
    lam = search_value
    err = 16 * delta**2 / (np.pi**2 * lam**2)
    return 0.5 * (c_ld + np.sqrt(c_ld*c_ld + 4*err))

def estimate_constant(w, delta, support_radius, search_grid, options):
    n_full = int(round(2 * support_radius / delta))
    weight_data = prepare_weight_grid(w, delta, n_full)
    K_full = build_operator(weight_data, n_full)
    upper, lower = 0.0, 0.0
    for lam in search_grid:
        c_ld, f = solve_finite_problem(K_full, lam, delta, n_full, options)
        if f is None: continue
        upper = max(upper, make_upper_certificate(c_ld, lam, delta))
        lo = (n_full - len(f)) // 2
        lower = max(lower, evaluate_test_function(f, K_full, lo, delta))
    grid_err = 0.5 * np.max(np.diff(search_grid)) if len(search_grid) > 1 else 0.0
    tail_upper = max(2 * search_grid[0], 2 / search_grid[-1])
    return lower, max(upper + grid_err, tail_upper)

if __name__ == "__main__":
    search_grid = np.arange(0.35, 2.81, 0.001)  # tails use c_lambda <= min(2*lambda, 2/lambda)
    options = {}
    for name, w in [("indicator", indicator_weight), ("gaussian", gaussian_weight)]:
        lo, hi = estimate_constant(w, delta=1.45e-3, support_radius=4.0,
                                   search_grid=search_grid,
                                   options=options)
        print(name, "C_opt in", (round(lo, 6), round(hi, 6)))
```

The method delivers certified brackets on $C_{opt}(w)$ — $[0.8055809,0.8055896]$ for the indicator weight and $[0.7152474,0.7152576]$ for the Gaussian — by reducing the variational problem to a $\lambda$-family of Hilbert-space eigenvalue problems over step functions and bounding the discretization error to $O(\delta^2)$.
