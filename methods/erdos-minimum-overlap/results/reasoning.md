OK, let me start from the object that refuses to go away. I split \(\{1,\ldots,2n\}\) into \(A\) and \(B\), each of size \(n\), and \(M_k\) counts the pairs \((a,b)\in A\times B\) with \(a-b=k\). Summing over \(k\) counts all \(n^2\) pairs once. There are \(4n-1\) possible differences, so some \(M_k\) is larger than \(n/4\). That is too crude, but it says the answer should be a constant times \(n\).

The discrete problem has too many local choices. If I stare at a large partition at scale \(n\), the only stable object is a density: how much of a small interval belongs to \(A\). So I rescale to an interval. I take \(f:[-1,1]\to[0,1]\), normalize \(\int_{-1}^1 f=1\), and put \(g=1-f\). The overlap at shift \(x\) is
\[
M(x)=\int_{-1}^1 f(t)g(x+t)\,dt,\qquad -2\le x\le2,
\]
with \(f,g\) extended by zero outside \([-1,1]\). The continuous quantity \(\inf_f\|M\|_\infty\) is not just an analogy; the step-function reduction says it is the limiting constant \(\mu\). A finite partition gives a step density after rescaling, and a bounded step density can be approximated by choosing the corresponding number of integer points in each small block. So I can work on the variational problem and come back to partitions afterward.

The first continuous identity is just the old averaging argument:
\[
\int_{-2}^2M(x)\,dx
=\int_{-1}^1 f(t)\int_{-2}^2g(x+t)\,dx\,dt
=\left(\int f\right)\left(\int g\right)=1.
\]
So the average height of \(M\) over an interval of length \(4\) is \(1/4\). Nothing new yet.

The next thing I can measure is where the mass of \(M\) sits. Define
\[
E(M)=\int_{-2}^2xM(x)\,dx,\qquad
E(f)=\int_{-1}^1xf(x)\,dx,\qquad
E(g)=\int_{-1}^1xg(x)\,dx.
\]
In the inner integral, write \(x=(x+t)-t\). Then
\[
E(M)=\int f(t)\int ((x+t)-t)g(x+t)\,dx\,dt=E(g)-E(f).
\]
Since \(f+g=1\) on the symmetric interval, \(E(f)+E(g)=0\), so \(E(f)=-E(M)/2\) and \(E(g)=E(M)/2\).

Now the second moment:
\[
\begin{aligned}
\int x^2M(x)\,dx
&=\int f(t)\int x^2g(x+t)\,dx\,dt\\
&=\int f(t)\int\bigl((x+t)^2-2t(x+t)+t^2\bigr)g(x+t)\,dx\,dt\\
&=\int x^2g(x)\,dx-2E(f)E(g)+\int t^2f(t)\,dt\\
&=\int_{-1}^1x^2(f(x)+g(x))\,dx-2E(f)E(g)\\
&=\frac23+\frac12E(M)^2.
\end{aligned}
\]
The centered variance is therefore \(2/3-E(M)^2/2\le2/3\). If all I know is that \(M\) has mass \(1\) and height at most \(\omega\), the way to make variance as small as possible is to pack a block of height \(\omega\) around the mean. That block has width \(1/\omega\) and variance \(1/(12\omega^2)\). Since every admissible \(M\) has centered variance at most \(2/3\), I get \(1/(12\omega^2)\le2/3\), hence \(\omega\ge1/\sqrt8\). Moser squeezes this moment argument further and gets \(\sqrt{4-\sqrt{15}}\approx0.35639395869\), but the wall is visible: moments do not remember that \(M\) came from \(f\) and \(1-f\).

The phrase "came from \(f\) and \(1-f\)" sounds like convolution. Correlations factor under Fourier transform, so I move there. On \([-2,2]\), set
\[
\widehat h(k)=\frac14\int_{-2}^2e^{-\pi ikx/2}h(x)\,dx.
\]
For \(k\ne0\),
\[
\begin{aligned}
\widehat M(k)
&=\frac14\int e^{-\pi ikx/2}\int f(t)g(x+t)\,dt\,dx\\
&=\frac14\int f(t)e^{\pi ikt/2}\int e^{-\pi ik(x+t)/2}g(x+t)\,dx\,dt\\
&=4\,\overline{\widehat f(k)}\,\widehat g(k).
\end{aligned}
\]
Because \(g=\mathbf 1_{[-1,1]}-f\),
\[
\widehat{\mathbf 1}_{[-1,1]}(k)
=\frac14\int_{-1}^1e^{-\pi ikx/2}\,dx
=\frac{1}{k\pi}\sin\frac{k\pi}{2},
\]
and so
\[
\widehat M(k)
=\frac{4}{k\pi}\sin\frac{k\pi}{2}\,\overline{\widehat f(k)}
-4|\widehat f(k)|^2.
\]

Now something rigid appears. At even frequencies \(k=2r\), the indicator coefficient vanishes. What remains is
\[
\widehat M(2r)=-4|\widehat f(2r)|^2\le0.
\]
This is the missing structure. A generic bounded mass-one function with the right variance does not need its even Fourier modes to have this sign, but every overlap with a translated complement does.

I want real constraints for an optimization program. Write \(f\) on \([-2,2]\) as
\[
f(x)=\frac14+\sum_{m\ge1}a_m\cos\frac{\pi mx}{2}
       +\sum_{m\ge1}b_m\sin\frac{\pi mx}{2},
\]
and write \(M\) as
\[
M(x)=\frac14+\sum_{m\ge1}A_m\cos\frac{\pi mx}{2}
       +\sum_{m\ge1}B_m\sin\frac{\pi mx}{2}.
\]
The coefficient identities are \(2\widehat f(m)=a_m-ib_m\),
\(A_m=\widehat M(m)+\widehat M(-m)\), and
\(B_m=i(\widehat M(m)-\widehat M(-m))\). Substitution gives
\[
A_m=\frac{4\sin(m\pi/2)}{m\pi}a_m-2(a_m^2+b_m^2),
\qquad
B_m=-\frac{4\sin(m\pi/2)}{m\pi}b_m.
\]
So
\[
A_{2r}=-2(a_{2r}^2+b_{2r}^2)\le0,\qquad B_{2r}=0.
\]
This already gives a clean family of inequalities, but the odd coefficients still depend on how the \([-1,1]\) series embeds into \([-2,2]\).

On the natural interval for \(f\), write
\[
f(x)=\frac12+\sum_{k\ge1}c_k\cos(\pi kx)+\sum_{k\ge1}d_k\sin(\pi kx),
\qquad c_0=\frac12.
\]
Orthogonality gives the even-frequency relation immediately:
\[
a_m=\frac12c_{m/2},\quad b_m=\frac12d_{m/2}\qquad(m\ \text{even}).
\]
For odd \(m\), the half-frequency basis on \([-2,2]\) does not line up with the integer basis on \([-1,1]\), so the coefficients spread across all \(c_k,d_k\):
\[
a_m=\frac{2m\sin(\pi m/2)}{\pi}
\sum_{k=0}^\infty\frac{(-1)^k}{m^2-4k^2}c_k,
\]
\[
b_m=\frac{4\sin(\pi m/2)}{\pi}
\sum_{k=1}^\infty\frac{k(-1)^k}{m^2-4k^2}d_k.
\]
The \(c_0\) term in the first sum is \(1/(2m^2)\). The tails must be controlled if I want a finite program. Parseval and \(0\le f\le1\) give
\[
1\ge\int_{-1}^1f^2
=\frac12+\sum_{k\ge1}(c_k^2+d_k^2),
\]
and then Cauchy-Schwarz bounds the tails. For \(1\le m<2T\),
\[
\left|\frac{2m}{\pi}\sum_{k=T+1}^\infty
\frac{(-1)^k\sin(\pi m/2)}{m^2-4k^2}c_k\right|
\le
\frac{1}{4-m^2/T^2}\frac{2m}{\pi\sqrt{6T^3}},
\]
\[
\left|\frac{4}{\pi}\sum_{k=T+1}^\infty
\frac{k(-1)^k\sin(\pi m/2)}{m^2-4k^2}d_k\right|
\le
\frac{1}{4-m^2/T^2}\frac{4}{\pi\sqrt{2T}}.
\]
So the infinite series can be replaced by finitely many \(c_k,d_k\) plus explicit slack variables \(\epsilon_m,\delta_m\).

Now I need a relaxation that every true \(M\) enters. I cut \([0,2]\) into \(N\) intervals of width \(L=2/N\). Let
\[
w_j=\frac1L\int_{(j-1)L}^{jL}M(x)\,dx,\qquad
v_j=\frac1L\int_{-jL}^{-(j-1)L}M(x)\,dx.
\]
If \(\Omega=\|M\|_\infty\), then \(0\le w_j,v_j\le\Omega\), and the mass identity becomes
\[
L\sum_{j=1}^N(w_j+v_j)=1.
\]
Any true overlap produces such variables. Therefore, if I minimize \(\Omega\) subject only to necessary conditions, the optimum is no larger than the true \(\|M\|_\infty\). A certified lower bound on that relaxation optimum is automatically a lower bound on every true overlap.

The interval averages can safely bound Fourier coefficients. Let \(\alpha_{j,m}^-\le\cos(\pi mx/2)\le\alpha_{j,m}^+\) and \(\beta_{j,m}^-\le\sin(\pi mx/2)\le\beta_{j,m}^+\) on the \(j\)-th positive interval. Then
\[
\frac{L}{2}\sum_j\alpha_{j,m}^-(w_j+v_j)
\le A_m\le
\frac{L}{2}\sum_j\alpha_{j,m}^+(w_j+v_j),
\]
and
\[
\frac{L}{2}\sum_j(\beta_{j,m}^-w_j-\beta_{j,m}^+v_j)
\le B_m\le
\frac{L}{2}\sum_j(\beta_{j,m}^+w_j-\beta_{j,m}^-v_j).
\]
The midpoint value plus or minus the Lipschitz error \(\pi mL/4\) gives explicit safe arrays.

The mean and second moment can be enclosed in the same way. If \(h_1\le E(M)\le h_2\), then
\[
h_1\le L^2\sum_j\bigl(jw_j-(j-1)v_j\bigr),
\]
because the right side is an upper interval estimate for \(E(M)\), and
\[
L^2\sum_j\bigl((j-1)w_j-jv_j\bigr)\le h_2,
\]
because the left side is a lower interval estimate. For the second moment, the lower Riemann estimate obeys
\[
L^3\sum_j(j-1)^2(w_j+v_j)\le
\int_{-2}^2x^2M(x)\,dx
=\frac23+\frac12E(M)^2
\le\frac23+\frac12h_2^2.
\]
There is also a complementary upper interval estimate using \(j^2\), but the lower estimate and the \(h_2\) ceiling are the useful constraints for forcing mass inward.

Let me test the idea in the easiest artificial case: suppose \(M\) is even. Then \(v_j=w_j\), all sine information disappears, and the even cosine signs are just linear inequalities. The variables are \(\Omega,w_1,\ldots,w_N\), and the constraints are
\[
0\le w_j\le\Omega,\qquad \sum_jw_j=\frac{N}{4},
\]
\[
\sum_j\alpha_{j,2r}^-w_j\le0\qquad(1\le r\le R),
\]
\[
L^3\sum_j(j-1)^2w_j\le\frac13.
\]
This is a genuine linear program. With a large grid and about twenty even-frequency inequalities, the dual certifies a lower bound above \(0.375\) for even profiles. That is the first sign that the Fourier signs are doing real work beyond the moment bound.

But the real problem is not even. I keep \(w_j\) and \(v_j\), keep the sine bounds, and use the full coefficient identity
\[
\frac{L}{2}\sum_j\alpha_{j,m}^-(w_j+v_j)
\le
\frac{4\sin(m\pi/2)}{m\pi}a_m-2(a_m^2+b_m^2).
\]
This is still convex in the right direction: an affine lower estimate is constrained to sit below a concave quadratic expression. The sine constraints are affine:
\[
\frac{L}{2}\sum_j(\beta_{j,m}^-w_j-\beta_{j,m}^+v_j)
\le -\frac{4\sin(m\pi/2)}{m\pi}b_m
\le
\frac{L}{2}\sum_j(\beta_{j,m}^+w_j-\beta_{j,m}^-v_j).
\]
The Parseval ball \(\sum_{k=1}^T(c_k^2+d_k^2)\le1/2\), the coefficient bounds \(|c_k|,|d_k|\le2/\pi\), and the tail-slack bounds keep the truncated Fourier variables honest.

If I leave the low-order data completely free, the relaxation can imitate a nearly flat function and stay too close to \(1/4\). The low modes control the most dangerous escape. By replacing one large run with boxes
\[
h_1\le E(M)\le h_2,\qquad
p_1\le c_1\le p_2,\qquad
q_1\le d_1\le q_2,
\]
I can force each relaxation to respect the part of parameter space it represents. This especially matters for \(A_2\), because \(a_2=c_1/2\) and \(b_2=d_1/2\), so
\[
A_2=-\frac12(c_1^2+d_1^2).
\]
The interval upper estimate on \(A_2\) must therefore satisfy
\[
\frac{L}{2}\sum_j\alpha_{j,2}^+(w_j+v_j)
\ge -\frac12\bigl(p_2^2+\max\{q_1^2,q_2^2\}\bigr).
\]
That single low-frequency inequality blocks a lot of fake flatness.

The numerical optimum of this convex relaxation still is not the theorem. I need the proof to run in the other direction: produce a feasible dual point whose objective is at least the claimed lower bound. The quadratic constraints can be written as second-order cone constraints, so the relaxation becomes an SOCP. In the SOCP dual, weak duality says every feasible dual assignment has objective no larger than the primal optimum. Since the true \(M\) maps into the primal feasible region with \(\Omega=\|M\|_\infty\), the chain is
\[
\text{dual objective}\le\text{relaxation optimum}\le\|M\|_\infty.
\]
So a verified dual objective of \(0.379005\) proves \(\|M\|_\infty\ge0.379005\) in the covered parameter box.

The word "verified" is the hard part. Floating point feasibility is not feasibility. The dual equalities are eliminated by choosing, for each primal coordinate, a cone vector with one unique nonzero coefficient in that coordinate and solving for the corresponding dual scalar. After elimination, feasibility is a finite list of cone inequalities and nonnegativity inequalities. Each left-minus-right margin must exceed a worst-case IEEE-754 error bound for the arithmetic used to compute it. Then the stored numbers define an actual feasible dual point, not just a solver's guess.

The boxes would be too many if I solved each one independently. The useful observation is that \(h_1,h_2,p_1,p_2\) enter the SOCP dual objective but not its feasibility constraints. One feasible dual point can therefore be reused while the objective is recomputed for nearby \(h,p\) values. Because the objective dependence is quadratic in those parameters, the set where the same dual point still certifies \(0.379005\) is an ellipse in the \((h,p)\) plane. First the coarse boxes reduce an optimal profile to the narrow region
\[
0\le E(M)\le0.06,\qquad 0.35\le c_1\le0.45,\qquad -0.02\le d_1\le0.02.
\]
Then seven verified dual points, at \(N=25000,T=7000,R=10\), have ellipses covering that remaining region. That closes the lower bound
\[
\mu\ge0.379005.
\]

The upper bound is the constructive half of the same continuous reduction. I use a symmetric step density on \([0,2]\). If the grid has \(n\) steps of width \(2/n\), the translate integral is exactly a nonperiodic cross-correlation:
\[
\int f(x)(1-f(x+s))\,dx
\quad\longrightarrow\quad
\frac{2}{n}\sum_i f_i(1-f_{i+r})
\]
for grid shifts \(r\), with only overlapping indices included. Haugland's 51-step symmetric density gives
\[
\mu\le0.3809268534330870.
\]
So the final interval is
\[
0.379005\le\mu\le0.3809268534330870.
\]

The code I want keeps the distinction clear: numerical primal solves are exploration, the step scorer is an explicit upper-bound evaluator, and the theorem-level lower bound comes from verified dual margins.

```python
import numpy as np


def symmetric_step_values(left_half_values):
    left = np.asarray(left_half_values, dtype=float)
    if left.ndim != 1 or left.size < 2:
        raise ValueError("left_half_values must be a one-dimensional array with a central step")
    if np.any((left < 0.0) | (left > 1.0)):
        raise ValueError("step heights must lie in [0, 1]")
    return np.concatenate([left, left[-2::-1]])


def step_overlap_objective(left_half_values):
    f = symmetric_step_values(left_half_values)
    n = f.size
    L = 2.0 / n
    g = 1.0 - f
    best = 0.0
    for shift in range(-(n - 1), n):
        if shift >= 0:
            value = L * np.dot(f[: n - shift], g[shift:])
        else:
            value = L * np.dot(f[-shift:], g[: n + shift])
        best = max(best, float(value))
    return best


def make_interval_bounds(N, R):
    L = 2.0 / N
    j = np.arange(1, N + 1, dtype=float)[:, None]
    m = np.arange(1, 2 * R + 1, dtype=float)[None, :]
    mid = L * (j - 0.5)
    slack = np.pi * m * L / 4.0
    return {
        "alpha_minus": np.cos(np.pi * m * mid / 2.0) - slack,
        "alpha_plus": np.cos(np.pi * m * mid / 2.0) + slack,
        "beta_minus": np.sin(np.pi * m * mid / 2.0) - slack,
        "beta_plus": np.sin(np.pi * m * mid / 2.0) + slack,
    }


def build_even_lp(N, R):
    from scipy.optimize import linprog

    L = 2.0 / N
    bounds = make_interval_bounds(N, R)
    alpha_minus = bounds["alpha_minus"]
    j = np.arange(1, N + 1, dtype=float)

    objective = np.r_[1.0, np.zeros(N)]
    rows = []
    rhs = []

    for idx in range(N):
        row = np.zeros(N + 1)
        row[0] = -1.0
        row[idx + 1] = 1.0
        rows.append(row)
        rhs.append(0.0)

    for r in range(1, R + 1):
        row = np.zeros(N + 1)
        row[1:] = alpha_minus[:, 2 * r - 1]
        rows.append(row)
        rhs.append(0.0)

    row = np.zeros(N + 1)
    row[1:] = L**3 * (j - 1.0) ** 2
    rows.append(row)
    rhs.append(1.0 / 3.0)

    equality = np.zeros((1, N + 1))
    equality[0, 1:] = 1.0
    return linprog(
        objective,
        A_ub=np.vstack(rows),
        b_ub=np.asarray(rhs),
        A_eq=equality,
        b_eq=np.asarray([N / 4.0]),
        bounds=[(0.0, 1.0)] + [(0.0, 1.0)] * N,
        method="highs",
    )


def admissible_properties_of_M(R, T):
    R, T = int(R), int(T)
    if not (0 < R < T):
        raise ValueError("tail bounds require positive integers with R < T")
    return {"R": R, "T": T}


def _tail_bounds(freq, T):
    denom = 4.0 - (freq / T) ** 2
    return (
        2.0 * freq / (np.pi * np.sqrt(6.0 * T**3) * denom),
        4.0 / (np.pi * np.sqrt(2.0 * T) * denom),
    )


def build_relaxation(N, properties, parameter_box):
    import cvxpy as cp

    R, T = properties["R"], properties["T"]
    needed = {"h1", "h2", "p1", "p2", "q1", "q2"}
    missing = needed - set(parameter_box)
    if missing:
        raise KeyError(f"missing parameter_box entries: {sorted(missing)}")

    h1, h2 = parameter_box["h1"], parameter_box["h2"]
    p1, p2 = parameter_box["p1"], parameter_box["p2"]
    q1, q2 = parameter_box["q1"], parameter_box["q2"]
    L = 2.0 / N
    j = np.arange(1, N + 1, dtype=float)
    k = np.arange(1, T + 1, dtype=float)
    bounds = make_interval_bounds(N, R)

    w = cp.Variable(N, nonneg=True)
    v = cp.Variable(N, nonneg=True)
    omega = cp.Variable(nonneg=True)
    c = cp.Variable(T)
    d = cp.Variable(T)
    eps = cp.Variable(R)
    delta = cp.Variable(R)

    cons = [w <= omega, v <= omega, omega <= 1.0]
    cons += [L * (cp.sum(w) + cp.sum(v)) == 1.0]
    cons += [L**2 * (j @ w - (j - 1.0) @ v) >= h1]
    cons += [L**2 * ((j - 1.0) @ w - j @ v) <= h2]
    cons += [L**3 * (((j - 1.0) ** 2) @ (w + v)) <= 2.0 / 3.0 + 0.5 * h2**2]
    cons += [cp.sum_squares(c) + cp.sum_squares(d) <= 0.5]
    cons += [cp.abs(c) <= 2.0 / np.pi, cp.abs(d) <= 2.0 / np.pi]
    cons += [p1 <= c[0], c[0] <= p2, q1 <= d[0], d[0] <= q2]

    for m in range(1, 2 * R + 1):
        s = np.sin(np.pi * m / 2.0)
        if m % 2 == 0:
            a_m = 0.5 * c[m // 2 - 1]
            b_m = 0.5 * d[m // 2 - 1]
        else:
            r = (m - 1) // 2
            eps_bound, delta_bound = _tail_bounds(m, T)
            cons += [cp.abs(eps[r]) <= eps_bound, cp.abs(delta[r]) <= delta_bound]
            a_m = eps[r] + (2.0 * m * s / np.pi) * (
                1.0 / (2.0 * m * m)
                + ((-1.0) ** k / (m * m - 4.0 * k * k)) @ c
            )
            b_m = delta[r] + (4.0 * s / np.pi) * (
                (k * (-1.0) ** k / (m * m - 4.0 * k * k)) @ d
            )

        idx = m - 1
        A_m = (4.0 * s / (m * np.pi)) * a_m - 2.0 * (cp.square(a_m) + cp.square(b_m))
        B_m = -(4.0 * s / (m * np.pi)) * b_m
        cons += [(L / 2.0) * (bounds["alpha_minus"][:, idx] @ (w + v)) <= A_m]
        cons += [(L / 2.0) * (bounds["beta_minus"][:, idx] @ w - bounds["beta_plus"][:, idx] @ v) <= B_m]
        cons += [(L / 2.0) * (bounds["beta_plus"][:, idx] @ w - bounds["beta_minus"][:, idx] @ v) >= B_m]

    cons += [
        (L / 2.0) * (bounds["alpha_plus"][:, 1] @ (w + v))
        >= -0.5 * (p2 * p2 + max(q1 * q1, q2 * q2))
    ]
    return cp.Problem(cp.Minimize(omega), cons)


def certify_lower_bound(relaxation=None, dual_certificate=None, solver=None):
    if dual_certificate is None:
        if relaxation is None:
            raise ValueError("provide either a relaxation or a dual certificate")
        value = relaxation.solve(solver=solver)
        return {"kind": "numeric_primal", "value": float(value), "status": relaxation.status}

    margins = np.asarray(dual_certificate["margins"], dtype=float)
    roundoff = np.asarray(dual_certificate["roundoff_bounds"], dtype=float)
    if margins.shape != roundoff.shape:
        raise ValueError("certificate margins and roundoff bounds must have the same shape")
    if np.any(margins <= roundoff):
        worst = float(np.min(margins - roundoff))
        raise ValueError(f"dual certificate is not safely interior; worst margin {worst}")
    return {
        "kind": "verified_dual",
        "lower_bound": float(dual_certificate["objective"]),
        "min_margin": float(np.min(margins - roundoff)),
    }
```

This is the shape I trust: the upper bound is a direct nonperiodic convolution calculation; the lower bound is a relaxation using mass, moment, Fourier sign, Fourier coefficient, interval-average, and tail constraints; and the number \(0.379005\) only becomes rigorous after a dual point survives the explicit roundoff check.
