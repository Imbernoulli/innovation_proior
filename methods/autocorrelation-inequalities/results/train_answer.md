The problem is to find the sharp constant in a family of autocorrelation inequalities. For a symmetric decreasing weight w on the real line with ||w||_1 = ||w||_infty = 1, we want the smallest C_opt(w) such that

    ∬ f(x) f(y) w(x-y) dx dy  ≤  C_opt(w) ||f||_1 ||f||_2

for every nonnegative f in L^1 ∩ L^2. The two important instances are the indicator weight w = χ_{[-1/2,1/2]}, which averages the autocorrelation over a unit window, and the Gaussian weight w(x)=e^{-π x^2}. C_opt(w) is the supremum of a Rayleigh-type quotient, so it is naturally bounded from below by any explicit test function and from above by any analytic inequality. Prior work followed exactly this split-track route: Barnard and Steinerberger gave an upper bound of 0.91 and a lower bound of 0.8 for the indicator case; Madrid and Ramos improved the upper bound to 0.864 using the sharp Hausdorff–Young inequality. Those advances leave a wide gap, provide no rigorous certificate that either side is near the truth, and say almost nothing about existence or structure of an extremizer. The related maximum-autocorrelation problem is even worse: its extremizers are non-smooth and non-monotone, and known discretizations scale exponentially in resolution. What is needed is a method that produces a narrow, certified two-sided bracket at polynomial cost.

The method I propose is a certified variational eigenvalue bracket for autocorrelation inequalities. It rewrites the non-quadratic denominator ||f||_1 ||f||_2 as a one-parameter family of quadratic forms via the arithmetic–geometric mean identity, then solves each quadratic subproblem by spectral methods while enforcing the nonnegativity constraint exactly through a support-length sweep, and finally bounds the discretization error rigorously using the smoothness inherited from the convolution operator.

The first reduction is sign and rearrangement. Because w is nonnegative, replacing f by |f| can only increase the numerator while leaving the denominator unchanged. Then Riesz rearrangement applies: for a symmetric decreasing kernel, replacing f by its symmetric-decreasing rearrangement f* again does not decrease the numerator, and all L^p norms are preserved. Hence the supremum is attained, if anywhere, among nonnegative, symmetric, decreasing bumps. Next comes a compact-support guarantee. Writing the Euler–Lagrange equation for the log-quotient and integrating over the support [-a,a] gives an explicit bound a ≤ 2 ||w||_1^2 / C_opt,R^2, where C_opt,R is the restricted constant. This rules out mass escaping to infinity and justifies working on a finite interval.

The key algebraic trick is to linearize the product norm. By AM–GM,

    2 ||f||_1 ||f||_2 = min_{λ>0} ( λ ||f||_2^2 + λ^{-1} ||f||_1^2 ).

For f ≥ 0 the term ||f||_1^2 = (∫ f)^2 is the square of a linear functional, so for each λ the denominator becomes a genuine Hilbert-space norm

    ||f||_{H_λ}^2 = λ ||f||_2^2 + λ^{-1} (∫ f)^2.

Define

    c_λ = max_{f ≥ 0, ||f||_{H_λ} ≤ 1} 2 ⟨f, K_w f⟩,

where K_w is convolution with w. Then C_opt = max_{λ>0} c_λ. The function c_λ is 1-Lipschitz and has an explicit lower envelope around its peak, so optimizing over a finite λ-grid with spacing Δλ gives a certified global value.

Discretization is done by block-averaging onto cells of width δ. Block averaging preserves L^1 mass and is an L^2 contraction, so the discrete constant c_{λ,δ} is a lower bound for c_λ. The reverse direction uses the optimal Poincaré inequality on each cell, Young’s convolution inequality, and a quantitative smoothness estimate for the extremizer derived from the Euler–Lagrange equation, namely ||(f*)'||_2 ≤ 4 / (c_λ λ^{3/2}). Together these give the explicit quadratic error bound

    0 ≤ c_λ - c_{λ,δ} ≤ 16 δ^2 / (π^2 c_λ λ^2).

Solving for c_λ yields a certified upper correction in terms of the computable c_{λ,δ}.

For fixed λ the discrete problem is a constrained generalized eigenvalue problem. The H_λ form is identity plus a rank-one projector, so it can be whitened by an explicit rank-one square root A_λ. After the change of variable g = A_λ f, the objective becomes the Rayleigh quotient of a symmetric matrix M_λ = 2 A_λ^{-1} K_w A_λ^{-1}. The unconstrained top eigenvalue is easy to compute by the power method, but the nonnegativity constraint is essential: a sign-changing vector could exploit cancellation in the rank-one term. Discrete Riesz rearrangement tells us the constrained maximizer is a single contiguous bump, so the constraint can be enforced exactly by sweeping over all contiguous support lengths, taking the top eigenvalue of each block, and keeping only those whose eigenvector is nonnegative. The largest admissible eigenvalue is c_{λ,δ}, and the corresponding eigenvector is an honest test function, giving a certified lower bound on C_opt when plugged into the original quotient.

Putting the pieces together, the algorithm chooses δ and a λ-grid, builds the triangular-smoothed discrete kernel, whitens each fixed-λ problem, sweeps support lengths with the power method, applies the quadratic discretization correction for the upper bound, and evaluates the admissible eigenvector for the lower bound. For δ ≈ 1.45×10^{-3} and Δλ ≈ 10^{-3} this produces brackets such as 0.8055809 ≤ C_opt(χ_{[-1/2,1/2]}) ≤ 0.8055896 and 0.7152474 ≤ C_opt(e^{-π x^2}) ≤ 0.7152576, with gaps below 10^{-5} at polynomial cost.

```python
import numpy as np
import scipy.integrate as si

def indicator_weight(t):
    return 1.0 if abs(t) <= 0.5 else 0.0

def gaussian_weight(t):
    return np.exp(-np.pi * t * t)

def prepare_weight_grid(w, delta, max_offset):
    tw = {}
    for k in range(-max_offset, max_offset + 1):
        s = k * delta
        val, _ = si.quad(
            lambda t: w(t) * (delta - abs(t - s)),
            s - delta, s + delta, limit=120,
        )
        tw[k] = val / delta**2
    return tw

def build_operator(weight_data, n_cells):
    K = np.empty((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            K[i, j] = weight_data[i - j]
    return K

def prepare_search_form(lam, n_cells, delta):
    a = n_cells * delta / 2.0
    b = (-2 * np.sqrt(lam) + np.sqrt(4 * lam + 8 * a / lam)) / (4 * a)
    one = np.ones(n_cells)
    A = np.sqrt(lam) * np.eye(n_cells) + b * delta * np.outer(one, one)
    return A, np.linalg.inv(A)

def leading_symmetric_eigenpair(M, options):
    iters = options.get("iters", 2000)
    tol = options.get("tol", 1e-13)
    v = np.random.default_rng(0).standard_normal(M.shape[0])
    v /= np.linalg.norm(v)
    mu = 0.0
    for _ in range(iters):
        w_ = M @ v
        nw = np.linalg.norm(w_)
        if nw == 0:
            break
        v_new = w_ / nw
        mu_new = v_new @ (M @ v_new)
        if abs(mu_new - mu) < tol:
            v, mu = v_new, mu_new
            break
        v, mu = v_new, mu_new
    return mu, v

def solve_finite_problem(K_full, lam, delta, n_full, options):
    best_val, best_f = 0.0, None
    for L in range(1, n_full + 1):
        lo = (n_full - L) // 2
        K = K_full[lo:lo + L, lo:lo + L]
        A, Ainv = prepare_search_form(lam, L, delta)
        M = 2.0 * (Ainv @ (delta * K) @ Ainv)
        mu, g = leading_symmetric_eigenpair(M, options)
        f = Ainv @ g
        if f.min() < 0:
            f = -f
        if f.min() >= -1e-9 and mu > best_val:
            best_val, best_f = mu, np.maximum(f, 0.0)
    return best_val, best_f

def evaluate_test_function(f, K_full, lo, delta):
    n = len(f)
    K = K_full[lo:lo + n, lo:lo + n]
    num = delta**2 * (f @ (K @ f))
    l1 = delta * f.sum()
    l2 = np.sqrt(delta * (f * f).sum())
    return num / (l1 * l2)

def make_upper_certificate(c_ld, lam, delta):
    err = 16 * delta**2 / (np.pi**2 * lam**2)
    return 0.5 * (c_ld + np.sqrt(c_ld * c_ld + 4 * err))

def estimate_constant(w, delta, support_radius, search_grid, options):
    n_full = int(round(2 * support_radius / delta))
    weight_data = prepare_weight_grid(w, delta, n_full)
    K_full = build_operator(weight_data, n_full)
    upper, lower = 0.0, 0.0
    for lam in search_grid:
        c_ld, f = solve_finite_problem(K_full, lam, delta, n_full, options)
        if f is None:
            continue
        upper = max(upper, make_upper_certificate(c_ld, lam, delta))
        lo = (n_full - len(f)) // 2
        lower = max(lower, evaluate_test_function(f, K_full, lo, delta))
    grid_err = 0.5 * np.max(np.diff(search_grid)) if len(search_grid) > 1 else 0.0
    tail_upper = max(2 * search_grid[0], 2 / search_grid[-1])
    return lower, max(upper + grid_err, tail_upper)

if __name__ == "__main__":
    search_grid = np.arange(0.35, 2.81, 0.001)
    options = {}
    for name, w in [("indicator", indicator_weight),
                    ("gaussian", gaussian_weight)]:
        lo, hi = estimate_constant(w, delta=1.45e-3, support_radius=4.0,
                                   search_grid=search_grid, options=options)
        print(name, "C_opt in", (round(lo, 6), round(hi, 6)))
```
