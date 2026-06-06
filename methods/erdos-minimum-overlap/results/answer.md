# Fourier-analytic convex programming for the minimum overlap constant

## Problem

For partitions $A\cup B=[2n]$ with $|A|=|B|=n$, $M_k=\#\{(a,b)\in A\times B:a-b=k\}$ and $M(n)=\min\max_k M_k$. The limiting constant is $\mu=\lim_{n\to\infty}M(n)/n$.

## Key idea

Use the continuous reduction: for $f:[-1,1]\to[0,1]$, $\int f=1$, $g=1-f$, and
$$
M(x)=\int_{-1}^1 f(t)g(x+t)\,dt,\qquad -2\le x\le 2,
$$
the target is $\mu=\inf_f\|M\|_\infty$. The moment facts
$$
\int M=1,\qquad \int x^2M(x)\,dx=\frac23+\frac12E(M)^2
$$
give $\mu\ge1/\sqrt8$, and Moser's refinement gives $\sqrt{4-\sqrt{15}}\approx0.35639395869$.

The additional structure comes from Fourier analysis. With
$$
\widehat h(k)=\frac14\int_{-2}^2 e^{-\pi ikx/2}h(x)\,dx,
$$
the correlation identity is
$$
\widehat M(k)=\frac{4}{k\pi}\sin\!\left(\frac{k\pi}{2}\right)\overline{\widehat f(k)}-4|\widehat f(k)|^2,\qquad k\ne0.
$$
If
$$
M(x)=\frac14+\sum_{m\ge1}A_m\cos\frac{\pi mx}{2}+\sum_{m\ge1}B_m\sin\frac{\pi mx}{2}
$$
and $f$ has corresponding coefficients $a_m,b_m$ on $[-2,2]$, then
$$
A_m=\frac{4\sin(m\pi/2)}{m\pi}a_m-2(a_m^2+b_m^2),\qquad
B_m=-\frac{4\sin(m\pi/2)}{m\pi}b_m.
$$
Thus every even cosine coefficient obeys
$$
A_{2r}=-2(a_{2r}^2+b_{2r}^2)\le0,
$$
and every even sine coefficient is zero. These infinitely many constraints are invisible to the variance method.

## Lower bound

Split $[0,2]$ into $N$ intervals of width $L=2/N$. Let $w_j$ and $v_j$ be the averages of $M$ on the positive and negative intervals, and minimize $\Omega$ subject to $0\le w_j,v_j\le\Omega$. Every true $M$ gives a feasible point with $\Omega\le\|M\|_\infty$, so the relaxation optimum lower-bounds $\mu$.

The constraints include mass, the lower mean constraint and second-moment ceiling coming from a box for $E(M)$, interval bounds for $A_m,B_m$, the real Fourier identities above, Parseval
$$
\sum_{k\ge1}(c_k^2+d_k^2)\le\frac12,
$$
and Cauchy-Schwarz tail bounds after truncating the $c_k,d_k$ series at $T$. The even-only linear program with $N=80000,R=20$ certifies at least $0.375$ for even $M$. The full convex program, swept over boxes for $(E(M),c_1,d_1)$ and certified by feasible SOCP dual points checked against IEEE-754 roundoff, gives
$$
\mu\ge0.379005.
$$

## Upper bound

For a symmetric 51-step density on $[0,2]$, the step-function objective is a finite nonperiodic shift correlation. Haugland's 51-step heights give
$$
\mu\le0.3809268534330870.
$$

## Code

```python
import numpy as np


def step_overlap_objective(left_half_heights):
    """Worst nonperiodic grid-shift overlap for a symmetric odd-step density."""
    left = np.asarray(left_half_heights, dtype=float)
    f = np.concatenate([left, left[-2::-1]])
    n = f.size
    L = 2.0 / n
    g = 1.0 - f
    vals = []
    for shift in range(-(n - 1), n):
        if shift >= 0:
            vals.append(L * np.dot(f[: n - shift], g[shift:]))
        else:
            vals.append(L * np.dot(f[-shift:], g[: n + shift]))
    return float(max(vals))


def admissible_properties_of_M(R, T):
    R, T = int(R), int(T)
    if not (0 < R < T):
        raise ValueError("tail bounds require positive integers with R < T")
    return {"R": R, "T": T}


def _tail_bounds(freq, T):
    denom = 4.0 - (freq / T) ** 2
    eps = 2.0 * freq / (np.pi * np.sqrt(6.0 * T**3) * denom)
    delta = 4.0 / (np.pi * np.sqrt(2.0 * T) * denom)
    return eps, delta


def build_relaxation(N, properties, parameter_box):
    import cvxpy as cp

    R, T = properties["R"], properties["T"]
    if not (0 < R < T):
        raise ValueError("tail bounds require positive integers with R < T")
    missing = {"h1", "h2", "p1", "p2", "q1", "q2"} - set(parameter_box)
    if missing:
        raise KeyError(f"missing parameter_box entries: {sorted(missing)}")
    h1, h2 = parameter_box["h1"], parameter_box["h2"]
    p1, p2 = parameter_box["p1"], parameter_box["p2"]
    q1, q2 = parameter_box["q1"], parameter_box["q2"]
    L = 2.0 / N
    j = np.arange(1, N + 1, dtype=float)
    mid = L * (j - 0.5)

    w = cp.Variable(N, nonneg=True)
    v = cp.Variable(N, nonneg=True)
    Omega = cp.Variable(nonneg=True)
    c = cp.Variable(T)
    d = cp.Variable(T)
    eps = cp.Variable(R)
    delta = cp.Variable(R)

    cons = [w <= Omega, v <= Omega, Omega <= 1.0]
    cons += [L * (cp.sum(w) + cp.sum(v)) == 1.0]
    cons += [L**2 * (j @ w - (j - 1.0) @ v) >= h1]
    cons += [L**3 * (((j - 1.0) ** 2) @ (w + v)) <= 2.0 / 3.0 + 0.5 * h2**2]
    cons += [cp.sum_squares(c) + cp.sum_squares(d) <= 0.5]
    cons += [cp.abs(c) <= 2.0 / np.pi, cp.abs(d) <= 2.0 / np.pi]
    cons += [p1 <= c[0], c[0] <= p2, q1 <= d[0], d[0] <= q2]

    k = np.arange(1, T + 1, dtype=float)
    for m in range(1, 2 * R + 1):
        s = np.sin(np.pi * m / 2.0)
        if m % 2 == 0:
            a_m = 0.5 * c[m // 2 - 1]
            b_m = 0.5 * d[m // 2 - 1]
        else:
            r = (m + 1) // 2 - 1
            a_tail, b_tail = _tail_bounds(m, T)
            cons += [cp.abs(eps[r]) <= a_tail, cp.abs(delta[r]) <= b_tail]
            a_m = eps[r] + (2.0 * m * s / np.pi) * (
                1.0 / (2.0 * m * m) + ((-1.0) ** k / (m * m - 4.0 * k * k)) @ c
            )
            b_m = delta[r] + (4.0 * s / np.pi) * (
                (k * (-1.0) ** k / (m * m - 4.0 * k * k)) @ d
            )

        alpha_minus = np.cos(np.pi * m * mid / 2.0) - np.pi * m * L / 4.0
        beta_minus = np.sin(np.pi * m * mid / 2.0) - np.pi * m * L / 4.0
        beta_plus = np.sin(np.pi * m * mid / 2.0) + np.pi * m * L / 4.0

        A_m = (4.0 * s / (m * np.pi)) * a_m - 2.0 * (cp.square(a_m) + cp.square(b_m))
        B_m = -(4.0 * s / (m * np.pi)) * b_m
        cons += [(L / 2.0) * (alpha_minus @ (w + v)) <= A_m]
        cons += [(L / 2.0) * (beta_minus @ w - beta_plus @ v) <= B_m]
        cons += [(L / 2.0) * (beta_plus @ w - beta_minus @ v) >= B_m]

    alpha_plus_2 = np.cos(np.pi * mid) + np.pi * L / 2.0
    cons += [(L / 2.0) * (alpha_plus_2 @ (w + v)) >= -0.5 * (p2**2 + max(q1 * q1, q2 * q2))]
    return cp.Problem(cp.Minimize(Omega), cons)


def certify_lower_bound(relaxation, dual_certificate=None, solver=None):
    if dual_certificate is not None:
        raise NotImplementedError("verify the SOCP dual inequalities and roundoff margins from the certificate data")
    return relaxation.solve(solver=solver)
```

The causal chain is: continuous reduction, Fourier factorization, even-frequency sign constraints, a finite convex relaxation over interval averages, and a checked SOCP dual certificate. Together these give $0.379005\le\mu\le0.3809268534330870$.
