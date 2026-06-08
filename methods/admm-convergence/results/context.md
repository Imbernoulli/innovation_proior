# Context: convergence theory for a decomposable, robust augmented-Lagrangian method

## Research question

We are given a convex program whose objective splits into two blocks tied by a single linear
constraint,
$$
\min_{x,z}\; f(x)+g(z)\quad\text{s.t.}\quad Ax+Bz=c,
$$
with $f:\mathbb R^n\to\mathbb R\cup\{+\infty\}$ and $g:\mathbb R^m\to\mathbb R\cup\{+\infty\}$ closed,
proper, and convex (possibly nonsmooth, possibly $+\infty$-valued, so set constraints fold in as
indicator functions), $A\in\mathbb R^{p\times n}$, $B\in\mathbb R^{p\times m}$, $c\in\mathbb R^{p}$.

The goal is an algorithm that is **both** decomposable and robust, and — the part that matters here —
an algorithm whose **convergence can actually be proved** under these weak assumptions:

- *Decomposable.* The $x$- and $z$-blocks should be updatable separately, so that when $f$ or $g$ is
  itself a sum of many terms the work splits across terms or machines. This is what lets the method
  scale to large, distributed problems.
- *Robust.* Convergence should hold **without** strict convexity of $f,g$, **without** finiteness
  (the functions may take the value $+\infty$), and **without** full rank of $A$ or $B$. The only
  hooks should be closedness/properness/convexity of $f,g$ and the existence of a saddle point of the
  ordinary Lagrangian.

The two classical tools each deliver exactly one of these two properties but not the other, and the
robust one is precisely the one that destroys decomposability. The open problem is to combine them and,
crucially, to **establish a convergence guarantee** — a quantity that provably decreases every
iteration, a proof that the constraint residual goes to zero and the objective reaches the optimum,
a verifiable stopping rule, and a worst-case rate — for an algorithm that merely alternates two cheap
block updates and a dual step, under assumptions weak enough to cover nonsmooth, rank-deficient
problems.

## Background

**Lagrangian duality and the saddle point.** For the equality-constrained problem
$\min_x f(x)$ s.t. $Ax=b$, the Lagrangian is $L(x,y)=f(x)+y^\top(Ax-b)$ and the dual function is
$g_{\mathrm d}(y)=\inf_x L(x,y)=-f^\*(-A^\top y)-b^\top y$, with $f^\*$ the convex conjugate. Under strong
duality the primal and dual optima coincide; a primal optimum is recovered as $x^\*=\arg\min_x
L(x,y^\*)$ when that minimizer is unique. A point $(x^\*,y^\*)$ is a **saddle point** of $L$ when
$L(x^\*,y)\le L(x^\*,y^\*)\le L(x,y^\*)$ for all $x,y$; existence of a saddle point is equivalent to
zero duality gap together with attainment, and it is the weakest natural assumption under which a
first-order primal–dual method can be expected to converge.

**Subdifferentials and optimality.** For closed proper convex $f$, the subdifferential $\partial f(x)
=\{v:\,f(z)\ge f(x)+v^\top(z-x)\ \forall z\}$ replaces the gradient; $0\in\partial f(x^\*)$ characterizes
a minimizer. The subdifferential of a sum of a subdifferentiable function and a differentiable function
on $\mathbb R^n$ is the sum of the subdifferential and the gradient. For the two-block problem the
optimality conditions are *primal feasibility* $Ax^\*+Bz^\*-c=0$ and *dual feasibility*
$0\in\partial f(x^\*)+A^\top y^\*$, $0\in\partial g(z^\*)+B^\top y^\*$.

**Proximal operators and resolvents.** For closed proper convex $h$ and $\rho>0$,
$\operatorname{prox}_{h/\rho}(v)=\arg\min_x\big(h(x)+\tfrac{\rho}{2}\|x-v\|^2\big)$ is single-valued and
firmly nonexpansive; it equals the resolvent $J_{(1/\rho)\partial h}=(I+(1/\rho)\partial h)^{-1}$ of the
maximal monotone operator $\partial h$. Minimizing a convex function plus a quadratic penalty is exactly
a resolvent evaluation. A maximal monotone operator $T$ has resolvent $J_{cT}=(I+cT)^{-1}$ that is
firmly nonexpansive for every $c>0$ (Minty; Rockafellar), which is the abstract reason penalty methods
are well behaved.

**The proximal point algorithm.** To find a zero of a maximal monotone $T$, iterate
$z^{k+1}=J_{cT}(z^k)=(I+cT)^{-1}(z^k)$ (Martinet; Rockafellar 1976). Because each $J_{cT}$ is firmly
nonexpansive and shares the zeros of $T$ as fixed points, the iterates contract toward a zero for any
fixed positive stepsize $c$; Rockafellar's more general form also allows standard positive stepsize
sequences, summable errors in the resolvent evaluations, and over/under-relaxation
$z^{k+1}=(1-\rho_k)z^k+\rho_kJ_{cT}(z^k)$ with $\rho_k\in(0,2)$ (Gol'shtein–Tret'yakov). The method of
multipliers is already known (Rockafellar 1976) to be one instance of this scheme applied to the dual.

**Operator splitting / Douglas–Rachford.** When $T=A+B$ with $A,B$ maximal monotone and the resolvent
$J_{cT}$ is hard but $J_{cA},J_{cB}$ are easy, one wants a method that uses only $J_{cA},J_{cB}$. The
Douglas–Rachford recursion (Douglas–Rachford 1956, originally for the discretized heat equation;
Lions–Mercier 1979 for monotone operators)
$$
z^{k+1}=J_{\lambda A}\big((2J_{\lambda B}-I)(z^k)\big)+(I-J_{\lambda B})(z^k)
$$
finds a point $z$ such that $J_{\lambda B}(z)\in\operatorname{zer}(A+B)$. Lions and Mercier showed the
Douglas–Rachford operator $G_{\lambda,A,B}=J_{\lambda A}\circ(2J_{\lambda B}-I)+(I-J_{\lambda B})$ is
firmly nonexpansive, hence its iteration converges.

**The variational-inequality form of a saddle problem.** A convex–concave saddle / two-block KKT
system can be written as a monotone variational inequality. With the opposite-sign multiplier
$\lambda=-y$, find $w^\*\in\Omega$ with
$\theta(u)-\theta(u^\*)+(w-w^\*)^\top F(w^\*)\ge0$ for all $w\in\Omega$, where $w=(x,z,\lambda)$,
$\theta(u)=f(x)+g(z)$, and $F(w)=(-A^\top\lambda,\,-B^\top\lambda,\,Ax+Bz-c)$. The map $F$ is affine
with a skew-symmetric linear part, hence monotone. This is the standard framing in which
iteration-complexity (worst-case rate) results for first-order primal–dual methods are stated.

## Baselines

**Dual ascent / dual decomposition (Everett ~1963; Dantzig–Wolfe; Benders).** Solve the dual by
gradient ascent: $x^{k+1}=\arg\min_x L(x,y^k)$; $y^{k+1}=y^k+\alpha^k(Ax^{k+1}-b)$. The residual
$Ax^{k+1}-b$ is $\nabla g_{\mathrm d}(y^k)$ when $g_{\mathrm d}$ is differentiable, and a supergradient of
the concave dual function otherwise (equivalently, its negative is a subgradient of $-g_{\mathrm d}$).
When $f(x)=\sum_i f_i(x_i)$ and $A=[A_1\cdots A_N]$, the Lagrangian separates and
the $x$-step splits into $N$ independent subproblems $x_i^{k+1}=\arg\min_{x_i}L_i(x_i,y^k)$ — the
decentralized "price update" (gather residuals, broadcast the price $y$). *Gap:* fragile. If $f$ is
affine in any coordinate the $x$-update is unbounded below and fails; if $g_{\mathrm d}$ is
nondifferentiable the dual subgradient method is nonmonotone and the stepsize $\alpha^k$ is delicate;
convergence needs assumptions (strict convexity, finiteness) that fail in many applications.

**Method of multipliers / augmented Lagrangian (Hestenes 1969; Powell 1969; Rockafellar; Bertsekas
monograph 1982).** Add a quadratic penalty:
$$
L_\rho(x,y)=f(x)+y^\top(Ax-b)+\tfrac{\rho}{2}\|Ax-b\|_2^2,\qquad \rho>0,
$$
which is the ordinary Lagrangian of the equivalent problem $\min_x f(x)+\tfrac{\rho}{2}\|Ax-b\|^2$ s.t.
$Ax=b$. The penalty makes the augmented dual $g_\rho(y)=\inf_x L_\rho(x,y)$ differentiable under mild
conditions. Iterate $x^{k+1}=\arg\min_x L_\rho(x,y^k)$; $y^{k+1}=y^k+\rho(Ax^{k+1}-b)$. Choosing the dual
stepsize equal to $\rho$ is what makes each iterate dual feasible: stationarity of the $x$-step gives
$0=\nabla f(x^{k+1})+A^\top(y^k+\rho(Ax^{k+1}-b))=\nabla f(x^{k+1})+A^\top y^{k+1}$. As the primal
residual $\to0$, optimality follows. *Gap:* robust (converges with $f$ taking value $+\infty$ or
non-strictly-convex) **but** when $f$ is separable the cross terms in $\tfrac{\rho}{2}\|\sum_i
A_ix_i-b\|^2$ couple the $x_i$, so $L_\rho$ is **not** separable and the $x$-minimization cannot be run
in parallel — decomposability is lost.

**Proximal point / Douglas–Rachford as solvers for the dual inclusion.** The dual of $\min f(x)+g(Mx)$
is $\max\,-\!\big(f^\*(-M^\top p)+g^\*(p)\big)$; optimality is a zero of
$\mathcal A+\mathcal B$ for the maximal monotone operators
$\mathcal A=\partial(f^\*\circ(-M^\top))$ and $\mathcal B=\partial g^\*$. The proximal point algorithm
on $\mathcal A+\mathcal B$ needs $J_{c(\mathcal A+\mathcal B)}$, which is as hard as the original
problem; Douglas–Rachford splitting uses only $J_{\lambda\mathcal A},J_{\lambda\mathcal B}$ and
converges by firm nonexpansiveness. *Gap (what is open):* these are
operator-theoretic existence/convergence statements; what is missing is an explicit, implementable
two-block primal algorithm whose convergence under the weak (closed/proper/convex + saddle point)
assumptions is proved directly — with a concrete Lyapunov certificate, residual-based stopping rule,
and worst-case rate — rather than left implicit in the abstract operator machinery.

## Evaluation settings

The natural problem families against which such a method would be exercised already exist and are the
yardstick:

- *Constrained / composite convex programs* $\min f(x)+g(z)$ s.t. $Ax+Bz=c$, including consensus and
  sharing formulations where $f=\sum_i f_i$ and the constraint enforces agreement across many local
  variables.
- *Regularized regression and statistics:* lasso $\min\tfrac12\|Cx-b\|_2^2+\lambda\|x\|_1$ and its
  relatives, where the $\ell_1$ term is handled by soft-thresholding and the quadratic by a cached
  factorization; group lasso; sparse inverse covariance.
- *Conic / cone-constrained programs* via indicator functions, where one block update is a projection.

The diagnostics that matter for a convergence study are the **primal residual** $r^k=Ax^k+Bz^k-c$
(constraint violation), the **dual residual** (a measure of the failure of dual feasibility), the
objective value $f(x^k)+g(z^k)$ relative to the optimum $p^\*$, and the per-iteration behavior of a
candidate Lyapunov quantity. Tolerances are set with an absolute term and a relative term scaled by the
ambient dimensions $\sqrt p,\sqrt n$ of the $\ell_2$ norms.

## Code framework

A generic scaffold for a two-block convex split with a single linear coupling constraint. What exists
already: closed-form/iterative block minimizers (a quadratic solve via a cached factorization,
soft-thresholding for $\ell_1$, a projection for an indicator), a Lagrange-multiplier vector, and an
outer loop that alternates "improve the primal, update the price."

```python
import numpy as np

# --- problem data: minimize f(x) + g(z)  s.t.  A x + B z = c ----------------
# f, g closed proper convex (possibly +inf-valued); A, B, c given.

def argmin_f_block(A, B, c, z, y, rho):
    """x-block update: minimize f plus the coupling/penalty terms, z and y fixed.
    For a quadratic f this is a linear solve (cache a factorization)."""
    raise NotImplementedError  # TODO: fill the x-subproblem

def argmin_g_block(A, B, c, x, y, rho):
    """z-block update: minimize g plus the coupling/penalty terms, x and y fixed.
    For an l1 term this is soft-thresholding; for an indicator, a projection."""
    raise NotImplementedError  # TODO: fill the z-subproblem

def dual_update(y, A, B, c, x, z, rho):
    """price / multiplier update from the constraint residual."""
    raise NotImplementedError  # TODO: fill the dual ascent step

def residuals(A, B, c, x, z, z_prev, rho):
    """primal feasibility residual and the dual-feasibility residual."""
    raise NotImplementedError  # TODO: fill after deriving the optimality conditions

def stopping(r, s, A, B, c, x, z, y, eps_abs, eps_rel):
    """termination from primal/dual residual sizes vs abs+rel tolerances."""
    raise NotImplementedError  # TODO: derived from the suboptimality bound

def solve(A, B, c, rho, x0, z0, y0, eps_abs=1e-4, eps_rel=1e-3, max_iter=1000):
    x, z, y = x0, z0, y0
    for k in range(max_iter):
        z_prev = z
        x = argmin_f_block(A, B, c, z, y, rho)      # block 1
        z = argmin_g_block(A, B, c, x, y, rho)      # block 2
        y = dual_update(y, A, B, c, x, z, rho)      # price
        r, s = residuals(A, B, c, x, z, z_prev, rho)
        if stopping(r, s, A, B, c, x, z, y, eps_abs, eps_rel):
            break
    return x, z, y
```
