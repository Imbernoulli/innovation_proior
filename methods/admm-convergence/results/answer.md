# Convergence of the Alternating Direction Method of Multipliers (ADMM)

## Problem

Solve a two-block convex program
$$
\min_{x,z}\; f(x)+g(z)\qquad\text{s.t.}\qquad Ax+Bz=c,
$$
with $f:\mathbb R^n\to\mathbb R\cup\{+\infty\}$, $g:\mathbb R^m\to\mathbb R\cup\{+\infty\}$ closed,
proper, convex, and $A\in\mathbb R^{p\times n}$, $B\in\mathbb R^{p\times m}$, $c\in\mathbb R^p$.
We want a method that is (i) **decomposable** — $x$ and $z$ updated separately, so per-term/per-machine
parallelism is possible — and (ii) **robust** — provably convergent with no strict convexity, no
finiteness, no full rank of $A,B$. Dual decomposition gives (i) but is fragile; the method of
multipliers gives (ii) but its quadratic penalty's cross term recouples the blocks.

## Key idea

Form the augmented Lagrangian
$$
L_\rho(x,z,y)=f(x)+g(z)+y^\top(Ax+Bz-c)+\tfrac{\rho}{2}\|Ax+Bz-c\|_2^2,\qquad\rho>0,
$$
and instead of the joint minimization of the method of multipliers, take a **single Gauss–Seidel
sweep**: minimize over $x$, then over $z$, then a dual ascent with step $\rho$. Freezing one block makes
the penalty's cross term a constant in the other, so each subproblem is a single-block penalized
solve — alternation restores decomposition while keeping augmented-Lagrangian robustness. Abstractly the
scheme is **Douglas–Rachford splitting on the dual**, which is the **proximal point algorithm** on a
maximal-monotone splitting operator with proximal-point stepsize fixed at $1$; firm nonexpansiveness of
the Douglas–Rachford resolvents is why convergence holds for **every** $\rho>0$, and stepsize $1$ is exactly what factors the
joint resolvent into the two sequential block solves.

## Algorithm

**Unscaled ADMM.**
$$
\begin{aligned}
x^{k+1}&=\arg\min_x L_\rho(x,z^k,y^k),\\
z^{k+1}&=\arg\min_z L_\rho(x^{k+1},z,y^k),\\
y^{k+1}&=y^k+\rho\,(Ax^{k+1}+Bz^{k+1}-c).
\end{aligned}
$$

**Scaled form** ($u=y/\rho$, using $y^\top r+\tfrac\rho2\|r\|^2=\tfrac\rho2\|r+u\|^2-\tfrac\rho2\|u\|^2$):
$$
\begin{aligned}
x^{k+1}&=\arg\min_x\big(f(x)+\tfrac\rho2\|Ax+Bz^k-c+u^k\|^2\big),\\
z^{k+1}&=\arg\min_z\big(g(z)+\tfrac\rho2\|Ax^{k+1}+Bz-c+u^k\|^2\big),\\
u^{k+1}&=u^k+Ax^{k+1}+Bz^{k+1}-c,\qquad u^k=u^0+\textstyle\sum_{j=1}^k r^j .
\end{aligned}
$$

**Residuals / optimality.** Optimality $=$ primal feasibility $Ax^\*+Bz^\*-c=0$ and dual feasibility
$0\in\partial f(x^\*)+A^\top y^\*$, $0\in\partial g(z^\*)+B^\top y^\*$. The $z$-step plus the
step-$\rho$ dual update make $0\in\partial g(z^{k+1})+B^\top y^{k+1}$ hold **at every iteration**; the
two remaining conditions are measured by
$$
r^{k+1}=Ax^{k+1}+Bz^{k+1}-c\ \ (\text{primal}),\qquad s^{k+1}=\rho A^\top B(z^{k+1}-z^k)\ \ (\text{dual}).
$$

**Stopping.** Terminate when $\|r^k\|_2\le\epsilon^{\mathrm{pri}}$ and $\|s^k\|_2\le\epsilon^{\mathrm{dual}}$,
$$
\epsilon^{\mathrm{pri}}=\sqrt p\,\epsilon^{\mathrm{abs}}+\epsilon^{\mathrm{rel}}\max\{\|Ax^k\|,\|Bz^k\|,\|c\|\},\qquad
\epsilon^{\mathrm{dual}}=\sqrt n\,\epsilon^{\mathrm{abs}}+\epsilon^{\mathrm{rel}}\|A^\top y^k\|.
$$

## Convergence theorem and proof

**Assumptions.** (A1) $f,g$ closed, proper, convex. (A2) the unaugmented Lagrangian $L_0$ has a saddle
point $(x^\*,z^\*,y^\*)$. (No strict convexity / finiteness / full rank.)

**Theorem.** Under (A1)–(A2) the ADMM iterates satisfy: $r^k\to0$ (primal feasibility), $s^k\to0$ (dual
feasibility), and $p^k:=f(x^k)+g(z^k)\to p^\*$ (objective convergence), with worst-case rate $O(1/k)$.

**Lyapunov function.** $V^k=\dfrac1\rho\|y^k-y^\*\|_2^2+\rho\|B(z^k-z^\*)\|_2^2\ge0$.

**Three inequalities (with $r^{k+1},\ \Delta z=z^{k+1}-z^k$, $p^{k+1}=f(x^{k+1})+g(z^{k+1})$):**

$$
V^{k+1}\le V^k-\rho\|r^{k+1}\|^2-\rho\|B\Delta z\|^2,\tag{A.1}
$$
$$
p^{k+1}-p^\*\le-(y^{k+1})^\top r^{k+1}-\rho(B\Delta z)^\top\big(-r^{k+1}+B(z^{k+1}-z^\*)\big),\tag{A.2}
$$
$$
p^\*-p^{k+1}\le (y^\*)^\top r^{k+1}.\tag{A.3}
$$

*Proof of (A.3).* Saddle inequality $L_0(x^\*,z^\*,y^\*)\le L_0(x^{k+1},z^{k+1},y^\*)$; left side $=p^\*$
(using $Ax^\*+Bz^\*=c$), right side $=p^{k+1}+(y^\*)^\top r^{k+1}$.

*Proof of (A.2).* $x^{k+1}$ minimizes $L_\rho(x,z^k,y^k)$, so
$0\in\partial f(x^{k+1})+A^\top y^k+\rho A^\top(Ax^{k+1}+Bz^k-c)$; substituting $y^k=y^{k+1}-\rho r^{k+1}$
gives $0\in\partial f(x^{k+1})+A^\top(y^{k+1}-\rho B\Delta z)$, i.e. $x^{k+1}$ minimizes
$f(x)+(y^{k+1}-\rho B\Delta z)^\top Ax$. Likewise $z^{k+1}$ minimizes $g(z)+(y^{k+1})^\top Bz$. Evaluate
each at the iterate vs. the optimum, add, use $Ax^\*+Bz^\*=c$.

*Proof of (A.1).* Add (A.2)+(A.3), multiply by $2$:
$$
2(y^{k+1}-y^\*)^\top r^{k+1}-2\rho(B\Delta z)^\top r^{k+1}+2\rho(B\Delta z)^\top B(z^{k+1}-z^\*)\le0.\tag{A.4}
$$
Using $y^{k+1}=y^k+\rho r^{k+1}$ and $r^{k+1}=\tfrac1\rho(y^{k+1}-y^k)$, the first term becomes
$\tfrac1\rho(\|y^{k+1}-y^\*\|^2-\|y^k-y^\*\|^2)+\rho\|r^{k+1}\|^2$. The remaining terms, after
substituting $z^{k+1}-z^\*=\Delta z+(z^k-z^\*)$ then $\Delta z=(z^{k+1}-z^\*)-(z^k-z^\*)$, become
$\rho\|r^{k+1}-B\Delta z\|^2+\rho(\|B(z^{k+1}-z^\*)\|^2-\|B(z^k-z^\*)\|^2)$. Hence
$$
V^k-V^{k+1}\ge\rho\|r^{k+1}-B\Delta z\|^2.\tag{A.6}
$$
Finally, $z^{k+1}$ minimizes $g(z)+(y^{k+1})^\top Bz$ and $z^k$ minimizes $g(z)+(y^k)^\top Bz$; adding the
two optimality inequalities gives $(y^{k+1}-y^k)^\top B\Delta z\le0$, and with $y^{k+1}-y^k=\rho r^{k+1}$,
$\rho>0$, this yields $(r^{k+1})^\top B\Delta z\le0$. Expanding
$\|r^{k+1}-B\Delta z\|^2=\|r^{k+1}\|^2-2(r^{k+1})^\top B\Delta z+\|B\Delta z\|^2\ge\|r^{k+1}\|^2+\|B\Delta z\|^2$,
so (A.6) gives (A.1). $\qquad\blacksquare$

**Consequences.** Telescoping (A.1): $\rho\sum_{k\ge0}(\|r^{k+1}\|^2+\|B\Delta z\|^2)\le V^0$, so
$r^k\to0$ and $B(z^{k+1}-z^k)\to0$ (hence $s^k=\rho A^\top B(z^{k+1}-z^k)\to0$). RHS of (A.2),(A.3) $\to0$,
so $p^k\to p^\*$. $V^k\le V^0$ bounds $y^k,Bz^k$. **Any $\rho>0$ works**: the $\tfrac1\rho$ and $\rho$
weights in $V$ make $\rho$ cancel in the decrease, which is $\ge0$ for every positive $\rho$.

**Stopping bound.** Using $-r^{k+1}+B(z^{k+1}-z^\*)=-A(x^{k+1}-x^\*)$ (from $c-Bz^\*=Ax^\*$) in (A.2),
$$
f(x^k)+g(z^k)-p^\*\le-(y^k)^\top r^k+(x^k-x^\*)^\top s^k\le\|y^k\|\,\|r^k\|+d\,\|s^k\|
$$
(with $\|x^k-x^\*\|\le d$), justifying the residual-based termination.

## Why it converges, structurally (Douglas–Rachford / proximal point)

Dualizing the canonical split $\min f(x)+g(Mx)$ gives $\max-(f^\*(-M^\top p)+g^\*(p))$; optimality is
$0\in\mathcal A p+\mathcal B p$ with $\mathcal A=\partial(f^\*\circ(-M^\top))$, $\mathcal B=\partial g^\*$
maximal monotone. The same dual-splitting calculation extends to $Ax+Bz=c$ by absorbing the linear maps
and constant into the two monotone pieces.
Douglas–Rachford splitting $z^{k+1}=G_{\lambda}(z^k)$, $G_\lambda=J_{\lambda\mathcal A}\circ(2J_{\lambda\mathcal B}-I)+(I-J_{\lambda\mathcal B})$,
applied to $\mathcal A,\mathcal B$ gives the alternating augmented-Lagrangian updates. Defining
$S_\lambda=(G_\lambda)^{-1}-I$ (maximal monotone), the DR iteration is the **proximal point algorithm on
$S_\lambda$ with stepsize $1$**: $G_\lambda=(I+S_\lambda)^{-1}$. Resolvents are firmly nonexpansive for
every resolvent parameter $\lambda=\rho>0$; the unit proximal-point step is what factors
$(I+S_\lambda)^{-1}$ into sequential evaluations $J_{\lambda\mathcal B}$ then $J_{\lambda\mathcal A}$.

## $O(1/k)$ rate (variational-inequality view)

Use the opposite-sign multiplier $\lambda=-y$ and stack $w=(x,z,\lambda)$,
$\theta(u)=f(x)+g(z)$, $F(w)=(-A^\top\lambda,\,-B^\top\lambda,\,Ax+Bz-c)$; $F$ is affine with a
skew-symmetric linear part, hence **monotone**. Optimality is the VI
$\theta(u)-\theta(u^\*)+(w-w^\*)^\top F(w^\*)\ge0\ \forall w\in\Omega$. With
$H=\operatorname{diag}(0,\rho B^\top B,\rho^{-1}I)$,
$\|w^k-w^{k+1}\|_H^2=\rho\|B(z^{k+1}-z^k)\|^2+\rho\|r^{k+1}\|^2$. The iterates obey
$$
\|w^{k+1}-w^\*\|_H^2\le\|w^k-w^\*\|_H^2-\|w^k-w^{k+1}\|_H^2\quad(\text{H-norm contraction, }=\text{(A.1)}),
$$
$$
\|w^k-w^{k+1}\|_H^2\le\|w^{k-1}-w^k\|_H^2\quad(\text{monotone non-increase}).
$$
The second inequality follows by writing the two consecutive VI optimality systems with
$M=\begin{pmatrix}I&0&0\\0&I&0\\0&-\rho B&I\end{pmatrix}$,
$Q=HM$, and using $(Q^\top+Q)-M^\top HM\succeq0$.
Telescoping the first ($\sum_t\|w^t-w^{t+1}\|_H^2\le\|w^0-w^\*\|_H^2$) and using the second
($(k{+}1)\|w^k-w^{k+1}\|_H^2\le\sum_{t=0}^k\|w^t-w^{t+1}\|_H^2$) gives
$$
\|w^k-w^{k+1}\|_H^2\le\frac{1}{k+1}\|w^0-w^\*\|_H^2,
$$
a worst-case $O(1/k)$ rate on the VI error measure ($\|w^k-w^{k+1}\|_H^2=0\Rightarrow w^{k+1}$ solves the
VI); the monotone, summable steps refine it to $o(1/k)$. The ergodic average attains the same $O(1/k)$ on
the VI gap.

## Generic driver

```python
import numpy as np

def admm(prox_f, prox_g, A, B, c, rho,
         x0, z0, y0, eps_abs=1e-6, eps_rel=1e-4, max_iter=10000):
    """
    minimize f(x) + g(z)  s.t.  A x + B z = c,   for any rho > 0.
    prox_f(z, y): argmin_x  f(x) + (rho/2)|| A x + B z - c + y/rho ||^2
    prox_g(x, y): argmin_z  g(z) + (rho/2)|| A x + B z - c + y/rho ||^2
    Freezing one block makes the penalty's cross term constant, so each solve is single-block.
    """
    x, z, y = np.array(x0, float), np.array(z0, float), np.array(y0, float)
    p, n = c.size, x.size
    for k in range(max_iter):
        z_prev = z
        x = prox_f(z, y)                      # x-minimization (block 1)
        z = prox_g(x, y)                      # z-minimization (block 2)
        r = A @ x + B @ z - c                 # primal residual
        y = y + rho * r                       # dual ascent, step = rho
        s = rho * (A.T @ (B @ (z - z_prev)))  # dual residual rho A^T B (z^{k+1}-z^k)
        eps_pri  = np.sqrt(p) * eps_abs + eps_rel * max(np.linalg.norm(A @ x),
                                                        np.linalg.norm(B @ z),
                                                        np.linalg.norm(c))
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(A.T @ y)
        if np.linalg.norm(r) <= eps_pri and np.linalg.norm(s) <= eps_dual:
            break
    return x, z, y
```

Example specialization (lasso $\min\tfrac12\|Cx-b\|^2+\lambda\|x\|_1$ as $f(x)=\tfrac12\|Cx-b\|^2$,
$g(z)=\lambda\|z\|_1$, $x-z=0$): $x$-update is a ridge solve $(C^\top C+\rho I)^{-1}(C^\top b+\rho(z-u))$
(cache the Cholesky factor), $z$-update is soft-thresholding $S_{\lambda/\rho}(x+u)$, dual update
$u\mathrel{+}= x-z$.
