# Alternating Direction Method of Multipliers (ADMM)

## Problem

Solve a convex problem whose objective splits into two blocks tied by one linear constraint:

$$ \min_{x,z}\ f(x)+g(z)\qquad\text{s.t.}\qquad Ax+Bz=c, $$

with $f,g$ closed proper convex (possibly nonsmooth, possibly $+\infty$-valued so constraints fold
in as indicators). The method must be **robust** — converge without strict convexity, finiteness,
or full rank of $A,B$ — *and* **decomposable** — each block updated separately, in parallel across
machines. The two classical tools each deliver one but not the other: dual decomposition is parallel
but fragile (its $x$-minimization is unbounded for affine / non-strictly-convex $f$); the method of
multipliers (augmented Lagrangian) is robust but serial (its quadratic penalty's cross terms couple
the blocks, so the joint minimization cannot be split).

## Key idea

Form the augmented Lagrangian of the two-block problem and, instead of minimizing it **jointly** over
$(x,z)$ (which re-couples the blocks), take a single **Gauss–Seidel sweep**: minimize over $x$, then
over $z$, then update the dual. Freezing one block makes the penalty's cross term a constant in the
other, so each subproblem is self-contained — the alternation keeps the augmented-Lagrangian
robustness while restoring per-block decomposition. The dual step size equals the penalty parameter
$\rho$, which makes one optimality condition hold automatically at every iterate. Abstractly this is
Douglas–Rachford splitting applied to the dual (an instance of the proximal point algorithm), which
supplies the convergence theory.

## Algorithm

Augmented Lagrangian:
$$ L_\rho(x,z,y)=f(x)+g(z)+y^\top(Ax+Bz-c)+\tfrac{\rho}{2}\|Ax+Bz-c\|_2^2,\qquad \rho>0. $$

**Unscaled ADMM:**
$$
\begin{aligned}
x^{k+1}&=\arg\min_x L_\rho(x,z^k,y^k)\\
z^{k+1}&=\arg\min_z L_\rho(x^{k+1},z,y^k)\\
y^{k+1}&=y^k+\rho\,(Ax^{k+1}+Bz^{k+1}-c).
\end{aligned}
$$

**Scaled form** ($u=y/\rho$; complete the square $y^\top r+\tfrac\rho2\|r\|^2=\tfrac\rho2\|r+u\|^2-\tfrac\rho2\|u\|^2$):
$$
\begin{aligned}
x^{k+1}&=\arg\min_x\Big\{f(x)+\tfrac\rho2\|Ax+Bz^k-c+u^k\|_2^2\Big\}\\
z^{k+1}&=\arg\min_z\Big\{g(z)+\tfrac\rho2\|Ax^{k+1}+Bz-c+u^k\|_2^2\Big\}\\
u^{k+1}&=u^k+Ax^{k+1}+Bz^{k+1}-c \qquad(\text{so } u^k=u^0+\textstyle\sum_{j\le k}r^j).
\end{aligned}
$$

**Residuals and stopping.** Primal $r^{k+1}=Ax^{k+1}+Bz^{k+1}-c$ and dual
$s^{k+1}=\rho A^\top B(z^{k+1}-z^k)$ are the residuals of the primal- and (first) dual-feasibility
conditions; the second dual condition $0\in\partial g(z^{k+1})+B^\top y^{k+1}$ holds automatically.
Both $\to 0$, and $f(x^k)+g(z^k)-p^\star\le -(y^k)^\top r^k+(x^k-x^\star)^\top s^k$, so stop when
$$ \|r^k\|_2\le\varepsilon^{\text{pri}},\qquad \|s^k\|_2\le\varepsilon^{\text{dual}}, $$
$$ \varepsilon^{\text{pri}}=\sqrt{p}\,\varepsilon^{\text{abs}}+\varepsilon^{\text{rel}}\max\{\|Ax^k\|,\|Bz^k\|,\|c\|\},\quad
\varepsilon^{\text{dual}}=\sqrt{n}\,\varepsilon^{\text{abs}}+\varepsilon^{\text{rel}}\|A^\top y^k\|. $$

**Convergence.** With $V^k=\tfrac1\rho\|y^k-y^\star\|^2+\rho\|B(z^k-z^\star)\|^2$ a Lyapunov function,
$V^{k+1}\le V^k-\rho\|r^{k+1}\|^2-\rho\|B(z^{k+1}-z^k)\|^2$, giving $r^k\to0$, $s^k\to0$, and
$f(x^k)+g(z^k)\to p^\star$. ADMM reaches modest accuracy fast and high accuracy slowly. A common
practical refinement is residual balancing of $\rho$ (inflate when $\|r\|\gg\|s\|$, deflate when
$\|s\|\gg\|r\|$; rescale $u\to u/\text{factor}$ when $\rho$ changes) and over-relaxation
$\alpha\in[1.5,1.8]$ (replace $Ax^{k+1}$ by $\alpha Ax^{k+1}-(1-\alpha)(Bz^k-c)$ in the $z$- and
$u$-updates).

**LASSO instance.** $\min\tfrac12\|Cx-b\|_2^2+\lambda\|x\|_1$ with the split $x-z=0$
($A=I,B=-I,c=0$), $f=\tfrac12\|Cx-b\|^2$, $g=\lambda\|z\|_1$:
$$
\begin{aligned}
x^{k+1}&=(C^\top C+\rho I)^{-1}\big(C^\top b+\rho(z^k-u^k)\big)\quad(\text{ridge; cache one Cholesky})\\
z^{k+1}&=S_{\lambda/\rho}(x^{k+1}+u^k),\qquad S_\kappa(a)=(a-\kappa)_+-(-a-\kappa)_+=\operatorname{sign}(a)(|a|-\kappa)_+\\
u^{k+1}&=u^k+x^{k+1}-z^{k+1},
\end{aligned}
$$
with $r^{k+1}=x^{k+1}-z^{k+1}$, $s^{k+1}=-\rho(z^{k+1}-z^k)$. The $z$-update is the prox of $\ell_1$
(soft-thresholding); for a different regularizer it becomes that regularizer's prox — a projection
$\Pi_{\mathcal C}$ for an indicator, block soft-thresholding $S_\kappa(a)=(1-\kappa/\|a\|_2)_+a$ for a
group norm.

**Consensus** ($\min\sum_i f_i(x)$ via copies $x_i=z$): $x_i^{k+1}=\arg\min_{x_i}f_i(x_i)+\tfrac\rho2\|x_i-z^k+u_i^k\|^2$
(parallel), $z^{k+1}=\overline{x}^{k+1}+\overline{u}^k$ (gather/average), $u_i^{k+1}=u_i^k+x_i^{k+1}-z^{k+1}$.
**Sharing** ($\min\sum_i f_i(x_i)+g(\sum_i x_i)$) is its dual; its $z$-update reduces to $n$ variables
by averaging.

## Code

```python
import numpy as np
from numpy.linalg import cholesky, norm


def shrinkage(a, kappa):
    # prox of kappa*||.||_1 : soft-threshold = (a-kappa)_+ - (-a-kappa)_+
    return np.maximum(0.0, a - kappa) - np.maximum(0.0, -a - kappa)


def factor(C, rho):
    # cache a Cholesky for the x-update; use the matrix-inversion-lemma form if short & fat
    q, d = C.shape
    if q >= d:
        L = cholesky(C.T @ C + rho * np.eye(d))
    else:
        L = cholesky(np.eye(q) + (1.0 / rho) * (C @ C.T))
    return L, L.T.conj()


def lasso_admm(C, b, lam, rho=1.0, alpha=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    # min 1/2||C x - b||^2 + lam||x||_1  via the split x - z = 0
    q, d = C.shape
    Ctb = C.T @ b
    L, U = factor(C, rho)
    x = np.zeros(d); z = np.zeros(d); u = np.zeros(d)   # u = scaled dual y/rho

    for k in range(n_iter):
        rhs = Ctb + rho * (z - u)                       # x-update: ridge regression
        if q >= d:
            x = np.linalg.solve(U, np.linalg.solve(L, rhs))
        else:
            tmp = np.linalg.solve(U, np.linalg.solve(L, C @ rhs))
            x = rhs / rho - (C.T @ tmp) / (rho ** 2)

        z_old = z
        x_hat = alpha * x + (1.0 - alpha) * z_old       # over-relaxation, alpha in [1, 2)
        z = shrinkage(x_hat + u, lam / rho)             # z-update: soft-threshold
        u = u + (x_hat - z)                             # scaled dual: running residual sum

        r_norm = norm(x - z)                            # primal residual r = x - z
        s_norm = norm(-rho * (z - z_old))               # dual residual s = -rho(z - z_old)
        eps_pri = np.sqrt(d) * abstol + reltol * max(norm(x), norm(z))
        eps_dual = np.sqrt(d) * abstol + reltol * norm(rho * u)
        if r_norm < eps_pri and s_norm < eps_dual:
            break
    return x, z


def admm(x_update, z_update, A, B, c, n, m, p, rho=1.0,
         n_iter=1000, abstol=1e-4, reltol=1e-2):
    # Generic scaled-form ADMM for min f(x)+g(z) s.t. A x + B z = c.
    # x_update(z, u, rho) = argmin_x f(x) + (rho/2)||A x + B z - c + u||^2
    # z_update(x, u, rho) = argmin_z g(z) + (rho/2)||A x + B z - c + u||^2
    x = np.zeros(n); z = np.zeros(m); u = np.zeros(p)   # u = scaled dual

    for k in range(n_iter):
        z_old = z
        x = x_update(z, u, rho)
        z = z_update(x, u, rho)
        r = A @ x + B @ z - c                           # primal residual
        u = u + r                                       # scaled dual update
        s = rho * (A.T @ (B @ (z - z_old)))             # dual residual rho A'B (z - z_old)

        eps_pri = np.sqrt(p) * abstol + reltol * max(norm(A @ x), norm(B @ z), norm(c))
        eps_dual = np.sqrt(n) * abstol + reltol * norm(rho * (A.T @ u))
        if norm(r) < eps_pri and norm(s) < eps_dual:
            break
    return x, z


def consensus_admm(prox_fi, N, d, rho=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    # min sum_i f_i(x) ; prox_fi[i](v, rho) = argmin_xi f_i(xi) + (rho/2)||xi - v||^2
    X = np.zeros((N, d)); z = np.zeros(d); Ud = np.zeros((N, d))

    for k in range(n_iter):
        z_old = z
        for i in range(N):                              # parallel local fits
            X[i] = prox_fi[i](z - Ud[i], rho)
        z = (X + Ud).mean(axis=0)                        # gather / average
        Ud = Ud + X - z                                 # local price updates

        r_norm = np.sqrt(((X - z) ** 2).sum())
        s_norm = np.sqrt(N) * rho * norm(z - z_old)
        eps_pri = np.sqrt(N * d) * abstol + reltol * max(norm(X), np.sqrt(N) * norm(z))
        eps_dual = np.sqrt(N * d) * abstol + reltol * norm(rho * Ud)
        if r_norm < eps_pri and s_norm < eps_dual:
            break
    return z, X
```
