The problem is to minimize a structural objective $f_0(x)$ — weight, compliance — over design variables $x = (x_1, \dots, x_n)$ such as member areas, plate thicknesses, or per-element densities, subject to behavioral constraints $f_i(x) \le 0$ (stresses, displacements, eigenvalues) and simple bounds $x_{\min} \le x \le x_{\max}$. What dominates everything is that the objective and every constraint, together with their gradients, are *implicit*: to know $f_i(x)$ and $\partial f_i/\partial x_j$ at one point I must run a finite-element analysis plus an adjoint sensitivity solve, and that single evaluation is the entire cost of the iteration. The variable count $n$ is large (one per element), the constraint count $m$ is small, and the responses are nonlinear and not even monotone in $x$. This forces the shape of any sane method before I design anything: I cannot afford a line search that probes $f_0$ at five trial points (five analyses for one step), nor a Hessian ($n^2$ second derivatives I will never compute). What I *can* afford, once per design point, is the values and gradients plus an unlimited amount of cheap arithmetic on them. So the method must be: at $x^k$, from value-and-gradient, build an explicit approximate subproblem that is cheap to solve to completion without touching the FE solver again, solve it, analyze at the new point, and repeat. The whole game is the quality of that subproblem.

The existing options each fail this test in a specific way. A first-order Taylor model in the direct variables, $f_i(x^k) + \sum_j (\partial f_i/\partial x_j)(x_j - x_j^k)$, carries no curvature, so the linear program it generates is unbounded and one is forced to bolt on artificial move limits $|x_j - x_j^k| \le \delta_j$ — and now the whole behavior is governed by those limits, too large overshooting, too small crawling, with the right value different for every problem and iteration. The optimality-criteria update, $x_e \leftarrow x_e \cdot ((-\partial f_0/\partial x_e)/(\lambda v_e))^\eta$ with $\lambda$ found by bisection, is cheap and scales, but it leans on a *single, monotone* resource constraint to give the multiplicative form a definite sign; with several general behavioral constraints of mixed sign there is no clean rule, no single $\lambda$ to bisect, and the exponent $\eta$ is hand-tuned with no theory. Reciprocal linearization in $1/x_j$ is the real seed — for a statically determinate structure a stress or displacement is *exactly* linear in the reciprocal areas (force over area), so $1/x_j$ is the natural coordinate — and convex linearization (CONLIN) exploits it by a sign split: keep the direct variable where $\partial f_i/\partial x_j > 0$ and switch to the reciprocal where $\partial f_i/\partial x_j < 0$, because the reciprocal term $c/x_j$ is convex only when its coefficient $c = -(\partial f_i/\partial x_j)(x_j^k)^2$ is nonnegative, i.e. when the derivative is negative. That yields convex, separable, conservative surrogates. But the reciprocal's singularity is *nailed at the origin* $x_j = 0$: once the iterate is fixed, the curvature of every term is fixed, with no dial — too gentle and I overshoot, too sharp and I crawl, and a sensitivity flipping sign jumps the model discontinuously, curable only by reimposing the very move limits the approach meant to remove.

I propose the Method of Moving Asymptotes (MMA). The idea is to *unnail* the reciprocal's singularity. Replace $1/(x_j - 0)$ by $1/(x_j - L_j)$ with a *lower asymptote* $L_j$ placed below $x_j^k$, and mirror it with $1/(U_j - x_j)$ using an *upper asymptote* $U_j > x_j^k$; both terms are convex and increasing/decreasing as needed, blow up as $x_j$ approaches the asymptote, and flatten toward a linear term as the asymptote recedes to infinity. The position of each asymptote is therefore a continuous, per-variable conservativeness knob — exactly the dial CONLIN lacked. With a pair $L_j < x_j^k < U_j$ per variable the surrogate is
$$ f_i(x) \approx r_i + \sum_j \Big[ \frac{p_{ij}}{U_j - x_j} + \frac{q_{ij}}{x_j - L_j} \Big], \qquad p_{ij}, q_{ij} \ge 0. $$
Convexity I guarantee by a clean construction: at each $(i,j)$ put the entire first-order weight on *one* term, the one whose slope has the right sign, and zero the other. Differentiating the basis functions, the $p$-term contributes slope $p_{ij}/(U_j - x_j)^2 > 0$ and the $q$-term slope $-q_{ij}/(x_j - L_j)^2 < 0$, so a positive derivative is matched by the $p$-term alone and a negative derivative by the $q$-term alone. Matching at $x = x^k$ fixes
$$ p_{ij} = (U_j - x_j^k)^2 \, \max\!\big(\tfrac{\partial f_i}{\partial x_j},\,0\big), \qquad q_{ij} = (x_j^k - L_j)^2 \, \max\!\big(-\tfrac{\partial f_i}{\partial x_j},\,0\big), $$
so both coefficients come out $\ge 0$ automatically — convexity for free — and the $(U_j - x_j^k)^2$, $(x_j^k - L_j)^2$ scalings are exactly the factors that cancel the $(U_j - x)^{-2}$, $(x - L)^{-2}$ from differentiating the basis so that the surrogate's slope equals the true slope at $x^k$. The constant $r_i = f_i(x^k) - \sum_j [\,p_{ij}/(U_j - x_j^k) + q_{ij}/(x_j^k - L_j)\,]$ matches the value. The second derivative is $\partial^2/\partial x_j^2 = 2p_{ij}/(U_j - x_j)^3 + 2q_{ij}/(x_j - L_j)^3 \ge 0$ between the asymptotes, so the surrogate is convex and separable by construction. The load-bearing property is the decoupling: the *gradient match is independent of where the asymptotes sit* (the scaling absorbs the position), while the *curvature* — hence conservativeness and step — is set entirely by how close $L_j, U_j$ are to $x_j^k$, near giving strong convexity and tiny cautious steps, far giving a nearly linear aggressive model. Fixing $L_j = 0$, $U_j \to \infty$ recovers direct/reciprocal convex linearization exactly, so MMA loses nothing and adds a control.

The asymptotes are driven from the iterate history, the only cheap information available, by an oscillation heuristic. With $s_j = (x_j^k - x_j^{k-1})(x_j^{k-1} - x_j^{k-2})$, the case $s_j > 0$ means the last two moves agreed in sign — steady progress, the model is too cautious, so relax by pushing the asymptote out with factor $1.2$; the case $s_j < 0$ means the moves reversed — oscillation, the model overshot, so tighten by pulling the asymptote in with factor $0.7$. The asymmetry is deliberate: relax gently, contract decisively, so damping wins when the two conflict. The distance from the asymptote is scaled by this factor relative to its previous value and clamped to $[0.01, 10]\cdot(x_{\max} - x_{\min})$ so a runaway product never drives an asymptote onto the iterate (curvature $\to \infty$) or off to infinity (curvature $\to 0$); the first two iterations, lacking history, place the asymptotes at a default half of the range. A separate guard keeps the optimizer off the singular asymptotes themselves by solving over a shrunk trust box $\alpha_j \le x_j \le \beta_j$ that stays a fraction $\texttt{albefa} = 0.1$ off each asymptote and caps the step at $\texttt{move} = 0.5$ of the range, intersected with the real bounds.

The resulting subproblem is convex, separable, with few constraints, so it is solved through its low-dimensional dual. Forming the Lagrangian with multipliers $\lambda_i \ge 0$, the $x$-part stays separable because every function is a sum of the same per-variable basis terms; defining $p_j(\lambda) = p_{0j} + \sum_i \lambda_i p_{ij}$ and $q_j(\lambda) = q_{0j} + \sum_i \lambda_i q_{ij}$, the inner minimization over $x$ splits into $n$ one-dimensional problems, each with a closed form found by setting the derivative to zero:
$$ x_j(\lambda) = \frac{L_j \sqrt{p_j(\lambda)} + U_j \sqrt{q_j(\lambda)}}{\sqrt{p_j(\lambda)} + \sqrt{q_j(\lambda)}}, $$
clamped to $[\alpha_j, \beta_j]$ — $O(n)$ analytic arithmetic, no inner iteration. The dual function $W(\lambda)$ is concave in the small $m$-dimensional multiplier space, and its gradient is the cleanest object in the method: by the envelope theorem $\partial W/\partial \lambda_i$ is simply the value of constraint $i$'s approximation at $x(\lambda)$, the primal residual. The expensive $n$ never enters as an $n$-dimensional system; the solve collapses to $m$. In practice I realize this robustly as a primal–dual interior-point method on the subproblem's KKT system rather than a bespoke dual maximizer with active-set bookkeeping for the clamped variables: introducing slacks and replacing each complementarity zero by a barrier parameter $\varepsilon > 0$ keeps every iterate strictly inside the box, so the active set emerges in the limit instead of being branched on. Each Newton step linearizes the relaxed conditions, eliminates the diagonal blocks for the bound and slack multipliers, and reduces to a dense linear solve of size $\min(m, n)$ — if $m < n$ eliminate $\Delta x$ to an $(m{+}1)$ system, otherwise eliminate $\Delta\lambda$ to an $(n{+}1)$ system — then takes a fraction-to-boundary step with a $1.01$ safety factor, backtracks until the residual norm decreases, and shrinks $\varepsilon$ by $0.1$ down the central path. To stay robust to infeasible or badly scaled starts, the standard NLP is embedded in an always-feasible problem with artificial variables $y_i \ge 0$ and a scalar $z \ge 0$, minimizing $f_0(x) + a_0 z + \sum_i (c_i y_i + \tfrac12 d_i y_i^2)$ subject to $f_i(x) - a_i z - y_i \le 0$; with $a_0 = 1$, $a_i = 0$, $d_i = 1$ and $c_i$ large this recovers the plain NLP at the optimum yet lets the method keep moving from an infeasible point, and the same template also expresses min–max and least-squares forms. One outer iteration is then: reuse the values and gradients at $x^k$, update the asymptotes, build $p_{ij}, q_{ij}, r_i$ and the trust box, solve the subproblem by interior point for $x^{k+1}$ and its multipliers, analyze at $x^{k+1}$, and stop on the original problem's KKT residual or reuse those fresh sensitivities for the next subproblem. One analysis per new design point; everything else is free.

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

```python
xval = x0.copy(); xold1 = x0.copy(); xold2 = x0.copy()
low = xmin.copy(); upp = xmax.copy()
a0, a, c, d = 1.0, np.zeros((m,1)), 1000*np.ones((m,1)), np.ones((m,1))
f0val, df0dx, fval, dfdx = fe_analysis(xval)
for k in range(1, maxit + 1):
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
        m, n, k, xval, xmin, xmax, xold1, xold2,
        f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d)
    xold2, xold1, xval = xold1, xval, xmma.copy()
    f0val, df0dx, fval, dfdx = fe_analysis(xval)          # expensive
    _, kktnorm, _ = kktcheck(m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s,
                             xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
    if kktnorm < tol:
        break
```
