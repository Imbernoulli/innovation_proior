We want to solve the linear program $\min c^\top x$ subject to $Ax = b,\ x \ge 0$ (with dual $\max b^\top y$ subject to $A^\top y + s = c,\ s \ge 0$), and we want two things that, around the early 1980s, no single method delivered together: a worst-case running time polynomial in $n$ and in the bit-length $L$ of the data, and genuine speed on the large sparse problems of operations research. The two existing poles each give only one. Dantzig's simplex method is the practical champion — it maintains a vertex, pivots along an edge to a better adjacent vertex, keeps the work cheap and the data sparse, and empirically finishes in roughly a linear number of pivots — but its whole life is on the boundary, and Klee and Minty built a squashed cube on which the pivot rule marches through all $2^n$ vertices, so the worst case is exponential. The opposite pole is Khachiyan's ellipsoid method, the first proof that LP is in $\mathrm{P}$ at $O(n^6 L^2)$, but it is useless in practice: the constants are enormous, it always behaves like its worst case, it ignores sparsity, and its round-off errors *accumulate*, so the precision required grows with the iteration count. The lesson from staring at why simplex is fragile is that the boundary is the enemy — the exponential blowup is a property of the polytope's vertex combinatorics. So the bet is to never touch a vertex: keep a point $x > 0$ strictly inside the feasible region and cut across the smooth interior toward the optimum, where $c^\top x$ is just a linear function and the machinery of gradients, Hessians and Newton's method is available.

I propose the interior-point method for linear programming, built on the central path and made polynomial by self-concordance, and realized in practice as a primal-dual predictor-corrector iteration. The obstacle is that the optimum lives *on* the boundary, so a strictly interior point can only approach it, with the constraint $x \ge 0$ threatening to be violated the instant a coordinate hits zero. The device is a penalty that is mild deep inside and explodes at the wall: for each $x_j \ge 0$ the cleanest function finite for $x_j > 0$ and $\to +\infty$ as $x_j \to 0^+$ is $-\ln x_j$, giving the logarithmic barrier $\varphi(x) = -\sum_j \ln x_j$, convex and smooth on the open orthant. Rather than minimize $c^\top x$ over a hard-bounded region, weight the objective against the barrier with a knob $t > 0$ and minimize $F_t(x) = t\,c^\top x - \sum_j \ln x_j$ over $\{x : Ax = b,\ x > 0\}$. Setting up the Lagrangian with multiplier $y$ for $Ax = b$, the stationarity condition $t\,c - (1/x) = A^\top \tilde y$, after rescaling $y := \tilde y / t$ and defining $s_j := 1/(t x_j)$, becomes exactly the system

$$Ax = b,\ x > 0 \qquad A^\top y + s = c,\ s > 0 \qquad x_j s_j = 1/t \ \text{ for all } j.$$

This is the load-bearing observation: these are the KKT optimality conditions of the LP *except* that the hard complementarity $x_j s_j = 0$ has been softened to $x_j s_j = 1/t$. The barrier did not change the problem; it gave a smooth deformation of optimality, and as $t \to \infty$ the relaxation $1/t \to 0$ recovers it. The minimizers $x^*(t)$ trace a smooth curve through the interior — the central path — ending at the optimum. And the soft version pays an unexpected dividend: $s > 0$ with $A^\top y + s = c$ is precisely dual feasibility, so every central point hands over a feasible dual certificate for free. Because $c^\top x = (A^\top y + s)^\top x = b^\top y + s^\top x$, the duality gap at any feasible pair is $x^\top s$, and on the central path

$$\text{gap} = x^\top s = \sum_j x_j s_j = n/t,$$

so a central point with parameter $t$ is provably within $n/t$ of optimal — to be $\varepsilon$-optimal we need $t \approx n/\varepsilon$. The barrier is not a treacherous penalty trick; it is a parametric family whose stationary points carry their own optimality certificate, with $t$ exactly the inverse gap.

The algorithm is therefore to follow the path, raising $t$, until $n/t < \varepsilon$, then round to a vertex. To move at fixed $t$, apply Newton's method to $F_t$; with $X = \mathrm{diag}(x)$ and barrier Hessian $X^{-2}$, eliminating $\Delta x$ from the KKT block gives the normal equations $A X^2 A^\top \Delta y = -t A X^2 c + b$, a single solve with a matrix of the form $A D^2 A^\top$ ($D$ diagonal positive) that is sparse when $A$ is and yields to a Cholesky factorization. Here is the historical objection: Newton converges quadratically only *near* the minimizer, and the textbook region is stated via the condition number of $\nabla^2 F_t$ and the Lipschitz constant of the Hessian — and as $t \to \infty$ and $x$ nears the boundary, $X^{-2}$ becomes wildly ill-conditioned, which is exactly why SUMT-style barrier methods lost favor with no polynomial bound. But something about that statement is wrong: Newton's method is affine-invariant — under $x = M\tilde x$ the iterates map over exactly — yet its convergence theory is phrased in condition number and Lipschitz constant, both frame-*dependent*. The theory smuggles in an arbitrary Euclidean ruler the method itself cannot see. The fix is to measure in the function's own local inner product $\langle u, v\rangle_x = u^\top \nabla^2 f(x)\, v$, in which the Hessian at $x$ is the identity by construction — perfectly conditioned in every frame. The only thing left to control is how fast the Hessian *changes*, measured in that same ruler, i.e. a self-referential bound of the third derivative by the second. Homogeneity in $h$ forces the powers, giving

$$|D^3 f(x)[h,h,h]| \le 2\,(D^2 f(x)[h,h])^{3/2},$$

a function with this property being called self-concordant. The exponents $3/2$ and $1/2$ are not a choice; the constant $2$ is, and it is chosen so that $-\ln x$ — for which $f'' = 1/x^2$, $f''' = -2/x^3$, so $|f'''| = 2(f'')^{3/2}$ — satisfies it with equality and *no* rescaling. So $-\ln x$ is the canonical building block, the property survives summation and the addition of the linear term $t\,c^\top x$ (zero third derivative), and $F_t$ is self-concordant for every $t$, automatically. Measuring progress by the Newton decrement $\lambda(x,f) = \sqrt{\nabla f(x)^\top [\nabla^2 f(x)]^{-1} \nabla f(x)}$, the damped step $x^+ = x - \frac{1}{1+\lambda}[\nabla^2 f(x)]^{-1}\nabla f(x)$ satisfies $\lambda(x^+) \le 2\lambda(x)^2$, so once $\lambda < 1/2$ the decrement squares each step and the region of quadratic convergence is $\{x : \lambda(x,f) \le 1/4\}$ — no condition number, no Lipschitz constant, frame-independent. The feared ill-conditioning was an illusion of the fixed ruler.

This lets the path be followed carefully instead of jumping $t$: keep $\lambda(x, F_t) \le 0.1$, nudge $t$ up just enough to stay inside the quadratic region of the new center, take one Newton step, return to $\lambda \le 0.1$. The allowable nudge comes from the barrier's stronger property — it is a $\vartheta$-self-concordant *barrier*, $|Df(x)[h]| \le \sqrt{\vartheta}\,\sqrt{D^2 f(x)[h,h]}$, with $\vartheta = n$ for $\varphi = -\sum_j \ln x_j$ (each $-\ln x_j$ contributes $1$), and this $\vartheta$ is exactly the gap constant $c^\top x^*(t) - \min c^\top x \le \vartheta/t = n/t$. The half-power $\sqrt{\vartheta}$ permits the multiplicative update

$$t \leftarrow (1 + 0.1/\sqrt{n})\cdot t, \quad \text{then one Newton step on } F_t,$$

so driving $n/t$ below $\varepsilon$ takes $O(\sqrt{n}\,\log(1/\varepsilon))$ steps — the $\sqrt{n}$, not $n$, being the prize, and coming directly from $\sqrt{\vartheta}$. Each step is one Cholesky solve with $A D^2 A^\top$, and because each step re-derives the certificate from the current point, round-off does not accumulate as in the ellipsoid method: an error is just a slightly-off interior point the next step corrects.

The same interior idea has a second, equivalent face. Reduce the LP to the simplex $\Delta = \{x \ge 0, \sum_j x_j = 1\} \cap \{Ax = 0\}$ with target value $0$ — manufactured by folding optimality into feasibility, stacking primal, dual and the no-gap condition $c^\top x - b^\top u = 0$ into one feasibility problem whose optimal objective is $0$. At the simplex center $a_0 = e/n$ the inscribed-to-circumscribed ball ratio is $R/r = n-1$, controlled, so optimizing the objective over the inscribed ball drops the gap to optimum by a factor $1 - 1/(n-1)$ per step. To repeat the step one must re-center, not by translation but by a body-preserving projective map $T(a,a_0): x_j \mapsto (x_j/a_j)/\sum_k (x_k/a_k)$, i.e. $x' = D^{-1}x/(e^\top D^{-1}x)$ with $D = \mathrm{diag}(a)$, which fixes the simplex's vertices and facets, keeps the constraints homogeneous, and sends $a$ to the center. Since the linear objective is *not* projective-invariant, progress is tracked instead by the Karmarkar potential

$$f(x) = n\,\ln(c^\top x) - \sum_j \ln x_j \qquad ( = \vartheta\ln(c^\top x) + \varphi(x),\ \vartheta = n),$$

a log-of-ratios that *is* sent to the same form by $T$, so per-step reduction composes across frames, while driving $f \to -\infty$ forces $c^\top x \to 0$. With step factor $\alpha = 1/4$ each step reduces the potential by a constant $\delta \approx 1/8$, giving geometric objective reduction; updating the $A D^2 A^\top$ solve incrementally by rank-one Sherman–Morrison corrections (since $D$ changes little) gives $O(n^{2.5})$ average work per step and $O(n^{3.5} L)$ overall, better than the ellipsoid's $O(n^6 L^2)$ by $n^{2.5}$. This potential $\vartheta\ln(c^\top x) + \varphi(x)$ is the path follower seen as running to infinity along a ray, i.e. sliding down the central path — the two views are the same animal.

What actually wins in practice keeps primal and dual together rather than following only the primal path. Treat the deformed KKT system as one nonlinear map $F(x,y,s) = (Ax - b,\ A^\top y + s - c,\ XSe - \mu e)$ and apply Newton, targeting $\mu$ slightly below the current average $\mu = x^\top s/n$ while keeping $x, s > 0$. The Newton system, with a centering parameter $\sigma \in [0,1]$ interpolating between the affine/predictor direction ($\sigma = 0$, straight at optimality) and pure centering ($\sigma = 1$, straight at the central point), eliminates to the same $A(S^{-1}X)A^\top$ kernel with $D^2 = S^{-1}X$. The Mehrotra predictor-corrector does this in two solves sharing one factorization: a predictor solves with $\sigma = 0$ and measures how much $\mu$ would drop, setting $\sigma = (\mu_{\mathrm{aff}}/\mu)^3$ adaptively (a great affine step pushes $\sigma \to 0$, a poor one $\to 1$); a corrector reuses the factorization to fold in the second-order term $\Delta x_{\mathrm{aff}}\!\circ\!\Delta s_{\mathrm{aff}}$ that the affine step left in the bilinear complementarity equation; the two directions are added and step lengths chosen by the fraction-to-boundary rule, taking $\eta = 0.9995$ of the largest step that keeps $x > 0$ and $s > 0$, separately in primal and dual. It stops when the relative KKT residual falls below tolerance — the duality gap plus feasibility, the very certificate the barrier produced at the start.

```python
import numpy as np
from scipy.linalg import solve_triangular

def solve_lp(A, b, c, tol=1e-8, max_iter=100, eta=0.9995):
    """
    Primal-dual interior-point (Mehrotra predictor-corrector) for
        min cᵀx  s.t.  Ax = b, x ≥ 0,   dual  max bᵀy s.t. Aᵀy + s = c, s ≥ 0.
    Drives the deformed KKT system  Ax=b, Aᵀy+s=c, x∘s = μe, x,s>0  to μ→0.
    """
    A = np.asarray(A, float); b = np.asarray(b, float); c = np.asarray(c, float)
    m, n = A.shape
    x, y, s = initial_point(A, b, c)
    bc = 1.0 + max(np.linalg.norm(b), np.linalg.norm(c))

    for k in range(max_iter):
        rb = A @ x - b                        # primal infeasibility
        rc = A.T @ y + s - c                  # dual infeasibility
        rxs = x * s                           # complementarity x∘s
        mu = rxs.mean()                       # = gap/n on the central path
        if np.linalg.norm(np.concatenate([rb, rc, rxs])) / bc < tol:
            break

        d = x / s                             # D² = S⁻¹X
        M = A @ (d[:, None] * A.T)            # A diag(x/s) Aᵀ  (SPD)
        M[np.diag_indices_from(M)] += 1e-10   # tiny diagonal shift for stability
        chol = np.linalg.cholesky(M)

        # Predictor (affine, σ = 0).
        dx_a, dy_a, ds_a = newton_dir(A, chol, d, x, s, rb, rc, rxs)
        ax, as_ = step_len(x, s, dx_a, ds_a, 1.0)
        mu_aff = ((x + ax * dx_a) @ (s + as_ * ds_a)) / n
        sigma = (mu_aff / mu) ** 3            # adaptive centering

        # Corrector: re-center by σμ, cancel 2nd-order term.
        rxs_cc = rxs + dx_a * ds_a - sigma * mu
        dx, dy, ds = newton_dir(A, chol, d, x, s, rb, rc, rxs_cc)

        ax, as_ = step_len(x, s, dx, ds, eta)  # fraction-to-boundary
        x = x + ax * dx; y = y + as_ * dy; s = s + as_ * ds

    return x, y, s, float(c @ x)


def newton_dir(A, chol, d, x, s, rb, rc, rxs):
    """
    Newton system  A Δx=-rb, AᵀΔy+Δs=-rc, S Δx+X Δs=-rxs.
    Eliminate Δs=-(rxs+s∘Δx)/x and Δx, leaving
        A diag(x/s) Aᵀ Δy = -rb - A diag(x/s)(rc - rxs/x).
    """
    rhs = -rb - A @ (d * (rc - rxs / x))
    dy = solve_triangular(chol.T, solve_triangular(chol, rhs, lower=True), lower=False)
    dx = d * (A.T @ dy + rc - rxs / x)
    ds = -(rxs + s * dx) / x
    return dx, dy, ds


def step_len(x, s, dx, ds, eta):
    """Largest fraction η of the step keeping x>0 and s>0."""
    ax = 1.0; neg = dx < 0
    if neg.any(): ax = min(ax, eta * np.min(-x[neg] / dx[neg]))
    as_ = 1.0; neg = ds < 0
    if neg.any(): as_ = min(as_, eta * np.min(-s[neg] / ds[neg]))
    return ax, as_


def initial_point(A, b, c):
    """Mehrotra start: least-norm primal/dual, then shift into the strict interior."""
    AAt = A @ A.T
    y = np.linalg.solve(AAt, A @ c)
    s = c - A.T @ y
    x = A.T @ np.linalg.solve(AAt, b)
    dx = max(-1.5 * x.min(), 0.0); ds = max(-1.5 * s.min(), 0.0)
    x = x + dx; s = s + ds
    pdct = 0.5 * (x @ s)
    x = x + pdct / s.sum(); s = s + pdct / x.sum()
    return x, y, s
```
