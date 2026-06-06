OK, let me think about the overlap problem from the beginning. I split $\{1,\dots,2n\}$ into two sets $A,B$ of size $n$, and $M_k$ counts the pairs $(a,b)\in A\times B$ with $a-b=k$. The average argument is unavoidable: summing over $k$ counts every pair once, so $\sum_k M_k=n^2$, and fewer than $4n$ shifts are available. That gives $M(n)>n/4$. It is clean, but it only uses total mass. The lower-bound problem is to find more global constraints that every overlap profile must satisfy.

The combinatorial object is too jagged, so I pass to the limiting density. I put a density $f:[-1,1]\to[0,1]$ in place of the indicator of $A$, set $g=1-f$, normalize $\int_{-1}^1 f=1$, and define
$$
M(x)=\int_{-1}^1 f(t)g(x+t)\,dt,\qquad -2\le x\le2.
$$
The limiting constant is the infimum of $\|M\|_\infty$ over these densities. That matters because now a lower bound can be proved by finding constraints on every such $M$, and an upper bound can be proved by writing down one good density.

The mass identity is the same argument in continuous form:
$$
\int_{-2}^2 M(x)\,dx
=\int_{-1}^1 f(t)\int_{-2}^2 g(x+t)\,dx\,dt
=\left(\int f\right)\left(\int g\right)=1.
$$
So the average of $M$ on an interval of length $4$ is $1/4$.

The next constraint is the second moment. Let $E(M)=\int_{-2}^2 xM(x)\,dx$, and similarly define $E(f),E(g)$. First,
$$
E(M)=\int f(t)\int xg(x+t)\,dx\,dt.
$$
Writing $x=(x+t)-t$ in the inner integral gives $E(M)=E(g)-E(f)$. Since $f+g=1$ on the symmetric interval, $E(f)+E(g)=0$, hence $E(f)=-E(M)/2$ and $E(g)=E(M)/2$.

Now expand the raw second moment in the same variables:
$$
\begin{aligned}
\int x^2M(x)\,dx
&=\int f(t)\int x^2g(x+t)\,dx\,dt\\
&=\int f(t)\int\bigl((x+t)^2-2t(x+t)+t^2\bigr)g(x+t)\,dx\,dt\\
&=\int x^2g(x)\,dx-2E(f)E(g)+\int t^2f(t)\,dt\\
&=\int_{-1}^1 x^2(f(x)+g(x))\,dx-2E(f)E(g)\\
&=\frac23+\frac12E(M)^2.
\end{aligned}
$$
The centered variance is therefore $2/3-E(M)^2/2$, at most $2/3$. If $M$ has mass $1$ and height at most $\mu$, the smallest possible centered variance comes from a block of height $\mu$ and width $1/\mu$ centered at the mean, and that variance is $1/(12\mu^2)$. Thus $1/(12\mu^2)\le2/3$, so $\mu\ge1/\sqrt8$. Moser's sharper use of the same moment obstruction gives $\sqrt{4-\sqrt{15}}\approx0.35639395869$. The important point is that this path is exhausted once the mass and second moment are exhausted.

I need constraints that know $M$ came from a correlation. Correlations should simplify under Fourier transform. Extend $f$ and $g$ by zero outside $[-1,1]$, and on $[-2,2]$ use
$$
\widehat h(k)=\frac14\int_{-2}^2 e^{-\pi ikx/2}h(x)\,dx.
$$
For $k\ne0$,
$$
\begin{aligned}
\widehat M(k)
&=\frac14\int e^{-\pi ikx/2}\int f(t)g(x+t)\,dt\,dx\\
&=\frac14\int f(t)e^{\pi ikt/2}\int e^{-\pi ik(x+t)/2}g(x+t)\,dx\,dt\\
&=4\,\overline{\widehat f(k)}\,\widehat g(k).
\end{aligned}
$$
Since $g=\mathbf 1_{[-1,1]}-f$ and
$$
\widehat{\mathbf 1}_{[-1,1]}(k)=\frac{1}{k\pi}\sin\left(\frac{k\pi}{2}\right),
$$
this becomes
$$
\widehat M(k)=\frac{4}{k\pi}\sin\left(\frac{k\pi}{2}\right)\overline{\widehat f(k)}-4|\widehat f(k)|^2.
$$

Now the even frequencies jump out. If $k=2r$, then $\sin(k\pi/2)=0$, so
$$
\widehat M(2r)=-4|\widehat f(2r)|^2\le0.
$$
This is not a moment; it is a whole family of sign constraints forced by the complement relation $g=1-f$.

I should express it in real coefficients because the optimization variables will be real. Write
$$
f(x)=\frac14+\sum_{m\ge1}a_m\cos\frac{\pi mx}{2}+\sum_{m\ge1}b_m\sin\frac{\pi mx}{2}
$$
on $[-2,2]$, and
$$
M(x)=\frac14+\sum_{m\ge1}A_m\cos\frac{\pi mx}{2}+\sum_{m\ge1}B_m\sin\frac{\pi mx}{2}.
$$
The exponential and real coefficients satisfy $2\widehat f(m)=a_m-ib_m$, $A_m=\widehat M(m)+\widehat M(-m)$, and $B_m=i(\widehat M(m)-\widehat M(-m))$. Substituting the Fourier identity gives
$$
A_m=\frac{4\sin(m\pi/2)}{m\pi}a_m-2(a_m^2+b_m^2),\qquad
B_m=-\frac{4\sin(m\pi/2)}{m\pi}b_m.
$$
So, in particular,
$$
A_{2r}=-2(a_{2r}^2+b_{2r}^2)\le0,\qquad B_{2r}=0.
$$

The coefficients $a_m,b_m$ on $[-2,2]$ must still be tied back to the coefficients of $f$ on its natural interval. If
$$
f(x)=\frac12+\sum_{k\ge1}c_k\cos(\pi kx)+\sum_{k\ge1}d_k\sin(\pi kx)
$$
on $[-1,1]$, with $c_0=1/2$, then orthogonality gives
$$
a_m=
\begin{cases}
\frac12c_{m/2},&m\text{ even},\\[2mm]
\frac{2m\sin(\pi m/2)}{\pi}\sum_{k=0}^{\infty}\frac{(-1)^k}{m^2-4k^2}c_k,&m\text{ odd},
\end{cases}
$$
and
$$
b_m=
\begin{cases}
\frac12d_{m/2},&m\text{ even},\\[2mm]
\frac{4\sin(\pi m/2)}{\pi}\sum_{k=1}^{\infty}\frac{k(-1)^k}{m^2-4k^2}d_k,&m\text{ odd}.
\end{cases}
$$
The odd-frequency formulas have tails, so I need finite variables plus rigorous slack. Parseval gives
$$
1\ge\int_{-1}^1 f^2=\frac12+\sum_{k\ge1}(c_k^2+d_k^2),
$$
because $0\le f\le1$ implies $f^2\le f$. For $1\le m<2T$, Cauchy-Schwarz gives
$$
\left|\frac{2m}{\pi}\sum_{k>T}\frac{(-1)^k\sin(\pi m/2)}{m^2-4k^2}c_k\right|
\le
\frac{1}{4-m^2/T^2}\frac{2m}{\pi\sqrt{6T^3}},
$$
and
$$
\left|\frac{4}{\pi}\sum_{k>T}\frac{k(-1)^k\sin(\pi m/2)}{m^2-4k^2}d_k\right|
\le
\frac{1}{4-m^2/T^2}\frac{4}{\pi\sqrt{2T}}.
$$
So I can truncate at $T$ and add bounded slack variables $\epsilon_m,\delta_m$ for the missing tails.

Now I need a finite relaxation in the correct direction. I cut $[0,2]$ into $N$ intervals of width $L=2/N$. Let $w_j$ be the average of $M$ on $[(j-1)L,jL]$, and $v_j$ the average on $[-jL,-(j-1)L]$. If $\Omega=\|M\|_\infty$, then $0\le w_j,v_j\le\Omega$, and
$$
L\sum_{j=1}^N(w_j+v_j)=1.
$$
Therefore any true $M$ gives a feasible point in a program that minimizes $\Omega$. The program is a relaxation, so its optimum can only be below the actual $\|M\|_\infty$; that is exactly the direction needed for a lower bound.

The interval averages also bound the Fourier coefficients. If $\alpha_{j,m}^\pm$ bound $\cos(\pi mx/2)$ on the $j$-th positive interval, and $\beta_{j,m}^\pm$ bound $\sin(\pi mx/2)$ there, then
$$
\frac{L}{2}\sum_j\alpha^-_{j,m}(w_j+v_j)\le A_m\le
\frac{L}{2}\sum_j\alpha^+_{j,m}(w_j+v_j),
$$
and
$$
\frac{L}{2}\sum_j(\beta^-_{j,m}w_j-\beta^+_{j,m}v_j)
\le B_m\le
\frac{L}{2}\sum_j(\beta^+_{j,m}w_j-\beta^-_{j,m}v_j).
$$
Taking midpoint values plus or minus the Lipschitz slack $\pi mL/4$ gives explicit safe $\alpha,\beta$ arrays.

The even symmetric test case is a useful sanity check. If $M$ is even, the $v_j$ duplicate the $w_j$, the sine constraints disappear, and the sign facts $A_{2r}\le0$ become linear inequalities:
$$
\sum_j\alpha^-_{j,2r}w_j\le0.
$$
Together with $\sum_jw_j=N/4$ and $L^3\sum_j(j-1)^2w_j\le1/3$, this is a linear program. With large $N$ and about twenty even-cosine constraints, it certifies a lower bound above $0.375$ in the even case. That tells me the Fourier signs are carrying real information beyond variance.

For the full problem I cannot assume symmetry. I keep $w_j$ and $v_j$, keep the sine constraints, and keep the quadratic formula for $A_m$. The constraint
$$
\frac{L}{2}\sum_j\alpha^-_{j,m}(w_j+v_j)\le
\frac{4\sin(m\pi/2)}{m\pi}a_m-2(a_m^2+b_m^2)
$$
is convex after moving all terms to one side, and Parseval is a quadratic norm bound. The sine constraints use the coefficient
$$
B_m=-\frac{4\sin(m\pi/2)}{m\pi}b_m,
$$
with the interval lower and upper bounds on either side. To stop the relaxation from drifting toward a fake flat profile, I run it over boxes for the low-order data. The lower endpoint $h_1$ gives the mean constraint, $h_2$ gives the second-moment ceiling, and the first Fourier coefficients are boxed as
$$
h_1\le E(M)\le h_2,\qquad p_1\le c_1\le p_2,\qquad q_1\le d_1\le q_2.
$$
The full range is too loose; the relaxation can fake something close to the flat $1/4$ profile. Splitting the ranges fixes the high-leverage quantities. In particular, since $a_2=c_1/2$ and $b_2=d_1/2$,
$$
A_2=-\frac12(c_1^2+d_1^2),
$$
so tight boxes for $c_1,d_1$ pin down the second cosine coefficient.

The numerical program still is not a proof. I need a dual certificate. The quadratic constraints are rewritten as second-order cone constraints, the SOCP dual is formed, and the equality constraints in the dual are eliminated using rows with a single nonzero $c$-vector entry. Then feasibility is a list of inequalities. If every inequality has a margin larger than the worst-case double-precision roundoff accumulation, weak duality turns the dual objective into a rigorous lower bound on the primal optimum. The endpoint data $h_1,h_2,p_1,p_2$ enters the dual objective but not dual feasibility, so one feasible dual point can certify a whole ellipse of $(h,p)$ values by recomputing that objective. A small collection of such ellipses covers the final narrow region, and the certified lower bound is
$$
\mu\ge0.379005.
$$

The upper-bound side uses the same continuous reduction in the opposite direction. I take a symmetric step density on $[0,2]$ and evaluate
$$
\max_k\int f(x)(1-f(x+k))\,dx
$$
as a finite nonperiodic shift correlation of the step heights. Optimizing those heights produces the 51-step upper bound $0.3809268534330870$.

Here is the code shape I would use to keep the two sides honest: a nonperiodic scorer for the constructive density, and a primal convex-relaxation builder whose constraints match the Fourier identities and interval bounds above.

```python
import numpy as np


def step_overlap_objective(left_half_heights):
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
    return (
        2.0 * freq / (np.pi * np.sqrt(6.0 * T**3) * denom),
        4.0 / (np.pi * np.sqrt(2.0 * T) * denom),
    )


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
            eps_bound, delta_bound = _tail_bounds(m, T)
            cons += [cp.abs(eps[r]) <= eps_bound, cp.abs(delta[r]) <= delta_bound]
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
    cons += [(L / 2.0) * (alpha_plus_2 @ (w + v)) >= -0.5 * (p2**2 + max(q1*q1, q2*q2))]
    return cp.Problem(cp.Minimize(Omega), cons)


def certify_lower_bound(relaxation, dual_certificate=None, solver=None):
    if dual_certificate is not None:
        raise NotImplementedError("verify the SOCP dual inequalities with explicit roundoff margins")
    return relaxation.solve(solver=solver)
```

The path is now clear: the variance identities give the old floor, the Fourier factorization creates sign and quadratic coefficient constraints, interval averages turn those constraints into a finite convex relaxation, and a verified dual point converts the numerical lower-bound computation into the theorem $\mu\ge0.379005$, while the nonperiodic step-function scorer supplies the upper-bound side.
