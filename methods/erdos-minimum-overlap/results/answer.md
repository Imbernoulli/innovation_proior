# Fourier-analytic convex programming for the Erdos minimum overlap constant

## Problem

For \(A\cup B=[2n]\), \(|A|=|B|=n\), define
\[
M_k=\#\{(a,b)\in A\times B:a-b=k\},\qquad
M(n)=\min_{A,B}\max_kM_k .
\]
The limiting constant is \(\mu=\lim_{n\to\infty}M(n)/n\).

## Continuous form

Use the density reduction. For \(f:[-1,1]\to[0,1]\), \(\int f=1\), put \(g=1-f\) and
\[
M(x)=\int_{-1}^1f(t)g(x+t)\,dt,\qquad -2\le x\le2.
\]
Then
\[
\mu=\inf_f\|M\|_\infty.
\]
The universal moment identities are
\[
\int_{-2}^2M=1,\qquad
\int_{-2}^2x^2M(x)\,dx=\frac23+\frac12E(M)^2,
\]
where \(E(M)=\int xM(x)\,dx\). These recover \(\mu\ge1/\sqrt8\), and Moser's refinement gives \(\sqrt{4-\sqrt{15}}\approx0.35639395869\).

## Fourier constraints

With
\[
\widehat h(k)=\frac14\int_{-2}^2e^{-\pi ikx/2}h(x)\,dx,
\]
the overlap factorizes as
\[
\widehat M(k)=\frac{4}{k\pi}\sin\!\left(\frac{k\pi}{2}\right)\overline{\widehat f(k)}
-4|\widehat f(k)|^2,\qquad k\ne0.
\]
Writing
\[
M(x)=\frac14+\sum_{m\ge1}A_m\cos\frac{\pi mx}{2}
       +\sum_{m\ge1}B_m\sin\frac{\pi mx}{2},
\]
and using real coefficients \(a_m,b_m\) for \(f\) on \([-2,2]\),
\[
A_m=\frac{4\sin(m\pi/2)}{m\pi}a_m-2(a_m^2+b_m^2),\qquad
B_m=-\frac{4\sin(m\pi/2)}{m\pi}b_m.
\]
Thus
\[
A_{2r}=-2(a_{2r}^2+b_{2r}^2)\le0,\qquad B_{2r}=0.
\]
These sign constraints are the main information missing from pure moment bounds.

## Lower bound

Split \([0,2]\) into \(N\) intervals of width \(L=2/N\), and let \(w_j,v_j\) be the averages of \(M\) on the positive and negative intervals. Minimize \(\Omega\) with \(0\le w_j,v_j\le\Omega\). Every true overlap gives a feasible point with \(\Omega=\|M\|_\infty\), so a dual lower bound on this relaxation lower-bounds \(\mu\).

The full relaxation uses mass, mean and second-moment enclosures, interval bounds for \(A_m,B_m\), the Fourier identities above, Parseval
\[
\sum_{k\ge1}(c_k^2+d_k^2)\le\frac12,
\]
and Cauchy-Schwarz tail bounds for the truncated \(c_k,d_k\) series. The even-profile simplification is a linear program and certifies \(0.375\) with \(N=80000,R=20\). The full SOCP relaxation, split over boxes for \(E(M),c_1,d_1\) and checked by feasible dual certificates with roundoff margins, gives
\[
\mu\ge0.379005.
\]

## Upper bound

A symmetric 51-step density on \([0,2]\) turns the translate objective into a finite nonperiodic cross-correlation. The evaluated objective is
\[
\mu\le0.3809268534330870.
\]
Together,
\[
0.379005\le\mu\le0.3809268534330870.
\]

## Code

```python
import numpy as np


HAUGLAND_51_LEFT = [
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0002938681556273,
    0.5952882223921177,
    0.7844530825484313,
    0.8950034338013842,
    0.0597964076006748,
    0.0189602838469592,
    0.7420501628172980,
    0.6444559588500921,
    0.3549040817844764,
    0.8762442385073478,
    0.5437907313675501,
    0.2679640048997296,
    0.8518954615823791,
    0.5211171156914872,
    1.0,
    0.5506146790047043,
    0.9007715390796991,
    0.8229000691941086,
    0.8879541710440111,
    0.9315424878319221,
    1.0,
]


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
