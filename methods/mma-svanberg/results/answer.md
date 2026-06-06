# MMA — the Method of Moving Asymptotes

## Problem

Solve the nonlinear, constrained design problem

```
minimize    f_0(x)
subject to  f_i(x) <= 0,        i = 1, ..., m
            x_min <= x <= x_max,
```

where `f_0` and `f_i` are smooth but *implicit*: each value and gradient costs a finite-element analysis. The number of variables `n` is large, the number of constraints `m` small. The goal is to converge in few analyses, building at each design point a cheap explicit subproblem that can be optimized to completion without any further FE call, and that is conservative enough to take large steps without overshooting — replacing fragile move limits and the single-monotone-constraint optimality-criteria heuristic with a general, tunable, convergent scheme.

## Key idea

At the current point `x^k`, replace each function by a **convex, separable** surrogate built from a pair of **moving asymptotes** `L_j < x_j^k < U_j` per variable:

```
f_i(x) ≈ r_i + Σ_j [ p_ij/(U_j − x_j) + q_ij/(x_j − L_j) ],   p_ij, q_ij >= 0.
```

The coefficients are set by a **sign split** of the gradient — put the weight on the upper-asymptote term where the derivative is positive, the lower-asymptote term where it is negative, scaled so the surrogate's slope matches the true gradient at `x^k`:

```
p_ij = (U_j − x_j^k)^2 · max(∂f_i/∂x_j, 0),
q_ij = (x_j^k − L_j)^2 · max(−∂f_i/∂x_j, 0),
r_i  = f_i(x^k) − Σ_j [ p_ij/(U_j − x_j^k) + q_ij/(x_j^k − L_j) ].
```

This is convex (`d^2/dx_j^2 = 2p_ij/(U_j−x_j)^3 + 2q_ij/(x_j−L_j)^3 >= 0`) and separable by construction. The **gradient match is independent of the asymptote positions**, but the **curvature** — hence the conservativeness and the step size — is set by how close `L_j, U_j` sit to `x_j^k`: near → strongly convex, cautious; far → nearly linear, aggressive. Fixing `L_j = 0`, `U_j → ∞` recovers direct/reciprocal convex linearization (CONLIN); moving the asymptotes is the generalization that gives a continuous, per-variable conservativeness dial.

The asymptotes are driven from the iterate history by an **oscillation heuristic**: with `s_j = (x_j^k − x_j^{k−1})(x_j^{k−1} − x_j^{k−2})`, relax (push the asymptote out, factor `1.2`) when `s_j > 0` (steady progress) and tighten (pull it in, factor `0.7`) when `s_j < 0` (zig-zag), clamping the asymptote distance to `[0.01, 10]·(x_max − x_min)`.

Each subproblem is **convex and separable** with few constraints, so it is solved through its **low-dimensional dual**: minimizing the Lagrangian splits into `n` one-variable problems with the closed form

```
x_j(λ) = ( L_j √p_j(λ) + U_j √q_j(λ) ) / ( √p_j(λ) + √q_j(λ) ),
   p_j(λ) = p_0j + Σ_i λ_i p_ij,   q_j(λ) = q_0j + Σ_i λ_i q_ij,
```

clamped to the trust box, and maximizing the concave `m`-dimensional dual `W(λ)` whose gradient is exactly the constraint residuals. In practice this is realized robustly as a **primal–dual interior-point** method on the subproblem KKT system, collapsing to a `min(m, n)`-sized linear solve per Newton step.

## Algorithm (one outer iteration)

1. Analyze at `x^k`: get `f_0, f_i` and gradients (the one expensive call).
2. Update asymptotes `L_j, U_j` from `x^k, x^{k−1}, x^{k−2}` (oscillation rule; defaults at first two iters).
3. Build `p_ij, q_ij, r_i` by the sign-split gradient match; set the trust box `α_j <= x_j <= β_j` (stay off the asymptotes by `albefa = 0.1`, cap the step by `move = 0.5`).
4. Solve the convex separable subproblem (interior point / dual) → `x^{k+1}` and multipliers.
5. Stop on the KKT residual of the original problem; else shift history and repeat.

Constraints are wrapped with cheap artificial variables `y_i >= 0`, `z >= 0` (objective `f_0 + a_0 z + Σ_i(c_i y_i + ½ d_i y_i^2)`, constraints `f_i − a_i z − y_i <= 0`; with `a_0 = 1, a_i = 0, d_i = 1, c_i` large this recovers the plain NLP but stays feasible even from an infeasible start, and also expresses min–max / least-squares forms).

## Code

Subproblem build — asymptote update, sign-split coefficients, trust box:

```python
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve

def mmasub(m, n, k, xval, xmin, xmax, xold1, xold2,
           f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d,
           move=0.5, asyinit=0.5, asydecr=0.7, asyincr=1.2,
           asymin=0.01, asymax=10, raa0=1e-5, albefa=0.1):
    een = np.ones((n, 1)); eem = np.ones((m, 1))
    xmami = np.maximum(xmax - xmin, 1e-5 * een)

    # Moving asymptotes (low = L, upp = U).
    if k <= 2:
        low = xval - asyinit * xmami
        upp = xval + asyinit * xmami
    else:
        s = (xval - xold1) * (xold1 - xold2)          # oscillation detector
        factor = een.copy()
        factor[s > 0] = asyincr                        # steady -> relax
        factor[s < 0] = asydecr                        # zig-zag -> tighten
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        low = np.maximum(low, xval - asymax * xmami)
        low = np.minimum(low, xval - asymin * xmami)
        upp = np.minimum(upp, xval + asymax * xmami)
        upp = np.maximum(upp, xval + asymin * xmami)

    # Trust box: off the asymptotes (albefa) and bounded step (move).
    alfa = np.maximum(np.maximum(low + albefa * (xval - low),
                                 xval - move * xmami), xmin)
    beta = np.minimum(np.minimum(upp - albefa * (upp - xval),
                                 xval + move * xmami), xmax)

    # Sign-split, gradient-matched, convex-by-construction coefficients.
    ux1 = upp - xval; xl1 = xval - low
    ux2, xl2 = ux1 * ux1, xl1 * xl1
    p0 = np.maximum(df0dx, 0); q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 / xmami             # strict-convexity floor
    p0 = (p0 + pq0) * ux2; q0 = (q0 + pq0) * xl2
    P = np.maximum(dfdx, 0); Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * (eem @ (1 / xmami).T)
    P = (diags(ux2.flatten()).dot((P + PQ).T)).T
    Q = (diags(xl2.flatten()).dot((Q + PQ).T)).T
    b = P @ (1 / ux1) + Q @ (1 / xl1) - fval           # b_i = -r_i

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, 1e-7, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp
```

Subproblem solve — primal–dual interior point tracing the central path; per Newton step the system is reduced to the `min(m, n)` side:

```python
def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    een = np.ones((n, 1)); eem = np.ones((m, 1))
    epsi = 1.0
    x = 0.5 * (alfa + beta); y = eem.copy(); z = np.array([[1.0]])
    lam = eem.copy(); s = eem.copy(); zet = np.array([[1.0]])
    xsi = np.maximum(een / (x - alfa), een)
    eta = np.maximum(een / (beta - x), een)
    mu = np.maximum(eem, 0.5 * c)

    while epsi > epsimin:                               # shrink barrier
        epsvecn = epsi * een; epsvecm = epsi * eem
        ux1 = upp - x; xl1 = x - low
        plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
        gvec = P @ (een / ux1) + Q @ (een / xl1)
        dpsidx = plam / ux1**2 - qlam / xl1**2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T @ lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm; rezet = zet * z - epsi; res = lam * s - epsvecm
        residu = np.vstack([rex, rey, rez, relam, rexsi, reeta, remu, rezet, res])
        residumax = np.max(np.abs(residu)); residunorm = np.linalg.norm(residu)

        inner = 0
        while residumax > 0.9 * epsi and inner < 200:
            inner += 1
            ux1 = upp - x; xl1 = x - low
            ux2, xl2, ux3, xl3 = ux1**2, xl1**2, ux1**3, xl1**3
            plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
            gvec = P @ (een / ux1) + Q @ (een / xl1)
            GG = (diags((een/ux2).flatten()).dot(P.T)).T \
               - (diags((een/xl2).flatten()).dot(Q.T)).T            # G_ij
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn/(x-alfa) + epsvecn/(beta-x)
            dely = c + d*y - lam - epsvecm/y
            delz = a0 - a.T @ lam - epsi/z
            dellam = gvec - a*z - y - b + epsvecm/lam
            diagx = 2*(plam/ux3 + qlam/xl3) + xsi/(x-alfa) + eta/(beta-x)  # D_x
            diagy = d + mu/y
            diaglamyi = s/lam + eem/diagy                                   # D_lam

            if m < n:                                   # eliminate dx -> (m+1)
                blam = dellam + dely/diagy - GG @ (delx/diagx)
                Alam = np.asarray(diags(diaglamyi.flatten())
                       + (diags((1/diagx).flatten()).dot(GG.T).T) @ GG.T)
                AA = np.block([[Alam, a], [a.T, -zet/z]])
                sol = solve(AA, np.vstack([blam, delz]))
                dlam = sol[:m]; dz = sol[m:m+1]
                dx = -delx/diagx - (GG.T @ dlam)/diagx
            else:                                        # eliminate dlam -> (n+1)
                diaglamyiinv = eem/diaglamyi; dellamyi = dellam + dely/diagy
                Axx = np.asarray(diags(diagx.flatten())
                      + (diags(diaglamyiinv.flatten()).dot(GG).T) @ GG)
                azz = zet/z + a.T @ (a/diaglamyi)
                axz = -GG.T @ (a/diaglamyi)
                bx = delx + GG.T @ (dellamyi/diaglamyi)
                bz = delz - a.T @ (dellamyi/diaglamyi)
                AA = np.block([[Axx, axz], [axz.T, azz]])
                sol = solve(AA, np.vstack([-bx, -bz]))
                dx = sol[:n]; dz = sol[n:n+1]
                dlam = GG @ dx/diaglamyi - dz*(a/diaglamyi) + dellamyi/diaglamyi

            dy   = -dely/diagy + dlam/diagy
            dxsi = -xsi + epsvecn/(x-alfa) - (xsi*dx)/(x-alfa)
            deta = -eta + epsvecn/(beta-x) + (eta*dx)/(beta-x)
            dmu  = -mu + epsvecm/y - (mu*dy)/y
            dzet = -zet + epsi/z - zet*dz/z
            ds   = -s + epsvecm/lam - (s*dlam)/lam

            # fraction-to-boundary step, then backtrack on residual norm
            xx  = np.vstack([y, z, lam, xsi, eta, mu, zet, s])
            dxx = np.vstack([dy, dz, dlam, dxsi, deta, dmu, dzet, ds])
            stmxx   = np.max(-1.01 * dxx / xx)
            stmalfa = np.max(-1.01 * dx / (x - alfa))
            stmbeta = np.max( 1.01 * dx / (beta - x))
            steg = 1.0 / max(stmxx, stmalfa, stmbeta, 1.0)

            xo, yo, zo, lo = x.copy(), y.copy(), z.copy(), lam.copy()
            xsio, etao, muo, zeto, so = (xsi.copy(), eta.copy(), mu.copy(),
                                         zet.copy(), s.copy())
            resinew = 2 * residunorm; it2 = 0
            while resinew > residunorm and it2 < 50:
                it2 += 1
                x = xo + steg*dx; y = yo + steg*dy; z = zo + steg*dz
                lam = lo + steg*dlam; xsi = xsio + steg*dxsi
                eta = etao + steg*deta; mu = muo + steg*dmu
                zet = zeto + steg*dzet; s = so + steg*ds
                ux1 = upp - x; xl1 = x - low
                plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
                gvec = P @ (een/ux1) + Q @ (een/xl1)
                dpsidx = plam/ux1**2 - qlam/xl1**2
                residu = np.vstack([
                    dpsidx - xsi + eta, c + d*y - mu - lam, a0 - zet - a.T@lam,
                    gvec - a*z - y + s - b, xsi*(x-alfa) - epsvecn,
                    eta*(beta-x) - epsvecn, mu*y - epsvecm,
                    zet*z - epsi, lam*s - epsvecm])
                resinew = np.linalg.norm(residu); steg *= 0.5
            residunorm = resinew; residumax = np.max(np.abs(residu)); steg *= 2
        epsi *= 0.1
    return x, y, z, lam, xsi, eta, mu, zet, s
```

Outer driver — one analysis per iteration:

```python
xval = x0.copy(); xold1 = x0.copy(); xold2 = x0.copy()
low = xmin.copy(); upp = xmax.copy()
a0, a, c, d = 1.0, np.zeros((m,1)), 1000*np.ones((m,1)), np.ones((m,1))
for k in range(1, maxit + 1):
    f0val, df0dx, fval, dfdx = fe_analysis(xval)          # expensive
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
        m, n, k, xval, xmin, xmax, xold1, xold2,
        f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d)
    _, kktnorm, _ = kktcheck(m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s,
                             xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
    if kktnorm < tol:
        break
    xold2, xold1, xval = xold1, xval, xmma.copy()
```

The optimizer never sees the structure — only values and gradients — so it drops in front of any FE/adjoint analysis (sizing, shape, or density-based continuum) as the constrained nonlinear-programming engine.
