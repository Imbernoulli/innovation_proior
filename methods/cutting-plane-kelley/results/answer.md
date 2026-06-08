# The Cutting-Plane Method (Kelley)

## Problem

Minimize a continuous convex, possibly nonsmooth, function over a compact convex set. By the
epigraph trick this is, with no loss of generality, the minimization of a **linear form** over a
convex set,

$$ \min\ c^\top x \qquad \text{s.t.}\qquad x\in R,\quad R=\{x:G(x)\le 0\},\ \ G\ \text{convex}, $$

with $R$ compact (e.g. confined to a box $S=\{x:Ax\ge b\}\supseteq R$). The aim is to solve this
curved program using only a linear-programming solver and a first-order oracle.

## Key idea

Convexity makes every subgradient a **cutting plane**. At any query point $t$ a supporting
hyperplane to $G$,

$$ p(x;t)=G(t)+\nabla p(t)\,(x-t)\ \le\ G(x)\quad\text{for all }x, $$

satisfies $p(x;t)\le 0$ for every feasible $x$ (since $G(x)\le 0$) while $p(t;t)=G(t)>0$ when
$t\notin R$. Hence the halfspace $\{x:p(x;t)\le 0\}$ **contains all of $R$ and excludes the
infeasible $t$** — a valid linear cut. Collecting one cut per query builds a **piecewise-linear
lower model** of the problem; minimizing it is an LP that yields the next iterate and a lower
bound. The model tightens monotonically, its values converge to the optimum, and the iterate
sequence has an optimal accumulation point.

## Algorithm

Start with a polyhedron $S_0=S\supseteq R$. For $k=0,1,2,\dots$:

1. **Solve the master LP** over the current outer approximation:
   $$ t_k=\arg\min\{c^\top x : x\in S_k\}. $$
2. If $G(t_k)\le 0$, stop: since $R\subseteq S_k$ and $t_k$ minimizes over $S_k$, the feasible
   point $t_k$ is optimal for $R$.
3. **Query the oracle** at $t_k$: get $G(t_k)$ and a subgradient $\nabla p(t_k)$.
4. **Add the cut** and re-solve:
   $$ S_{k+1}=S_k\cap\{x:\ G(t_k)+\nabla p(t_k)\,(x-t_k)\le 0\}. $$

Each step is an LP equal to the previous one plus one row. For a convex-objective formulation,
the same construction is used in epigraph form: the master LP minimizes the PWL lower model
$\hat f(x)=\max_{i\le k}\big(f(x_i)+g_i^\top(x-x_i)\big)$ subject to $x\in S$.

**Warm start (dual form).** With multipliers $u\ge0$ on $Ax\ge b$ and $v_i\ge0$ on the cuts,
$$ \max\ u^\top b+\sum_i v_i\big(G(t_i)-\nabla p(t_i)\,t_i\big)\ \ \text{s.t.}\ \ u^\top A-\sum_i v_i\,\nabla p(t_i)=c,\ u\ge0,\ v\ge0. $$
Adding a cut adds one dual variable $v_k$; padding the previous dual solution with $v_k=0$ is a
feasible start, so each iteration costs one new column plus the subgradient evaluation.

**Bounds and stopping.** In the linear-over-$R$ formulation, $f_k=c^\top t_k$ is a lower bound
until a feasible $t_k$ proves optimal. In the convex-objective epigraph formulation, because
each $\hat f\le f$, the master-LP value is a **lower bound** $\ell_k$ on $f^\star$; the best
feasible objective so far is an **upper bound** $u_k$. Stop when $u_k-\ell_k\le\epsilon$. A
small constraint violation $G(t_k)$ alone is not a certified objective gap.

## Convergence theorem

**Theorem.** Let $G$ be continuous convex on a compact convex set $S$, with a support
$p(x;t)=G(t)+\nabla p(t)(x-t)$ at every $t\in S$ satisfying $\lVert\nabla p(t)\rVert\le K<\infty$;
let $c^\top x$ be a linear form, $\lVert c\rVert<\infty$, and let
$R=\{x:G(x)\le0\}\subset S$ be nonempty. Define $S_0=S$,
$S_k=S_{k-1}\cap\{x:p(x;t_{k-1})\le 0\}$, and $t_k=\arg\min\{c^\top x:x\in S_k\}$. Then $\{t_k\}$
has a subsequence converging to a point $\tau\in R$ with $c^\top\tau\le c^\top x$ for all $x\in R$
(i.e. $\tau$ is optimal), and $f_k:=c^\top t_k\uparrow f^\star:=\min_R c^\top x$.

**Proof.**

*Monotone, bounded model values.* Cuts are only added, so $S_k\subseteq S_{k-1}$ and minimizing
over a smaller set gives $f_k\ge f_{k-1}$. Every cut holds on all of $R$, so $R\subseteq S_k$ and
$f_k=\min_{S_k}c^\top x\le\min_R c^\top x=f^\star$. Thus $f_0\le f_1\le\cdots\le f^\star$
converges.

*Cuts seen by later iterates.* Since $t_k$ minimizes $c^\top x$ over $S_k$ and $S_k$ enforces all
earlier cuts,
$$ G(t_i)+\nabla p(t_i)\,(t_k-t_i)\le 0,\qquad 0\le i\le k-1. \tag{$\ast$}$$

*Violations cannot persist.* If some $t_k\in R$, then $R\subseteq S_k$ and the minimizing property
of $t_k$ over $S_k$ already makes it optimal. Otherwise all generated iterates are infeasible.
Suppose $\exists\,r>0$ with $G(t_k)\ge r$ for infinitely many $k$. For two such indices $i<k$,
rearranging $(\ast)$ and using Cauchy–Schwarz and the Lipschitz bound,
$$ r\le G(t_i)\le -\nabla p(t_i)(t_k-t_i)=\nabla p(t_i)(t_i-t_k)\le\lVert\nabla p(t_i)\rVert\,\lVert t_i-t_k\rVert\le K\lVert t_i-t_k\rVert, $$
so $\lVert t_i-t_k\rVert\ge r/K$. The high-violation subsequence is therefore $r/K$-separated and
has no Cauchy (hence no convergent) subsequence — impossible in the compact set $S$. Contradiction.

*Conclusion.* In the nonterminating case, the positive violations converge to zero; by continuity
and compactness a subsequence has a limit $\tau\in S$ with $G(\tau)=0$, i.e. $\tau\in R$. Along it
$c^\top t_k=f_k\to c^\top\tau$; since
$f_k\uparrow$ a limit $\le f^\star$ while $\tau\in R$ forces $c^\top\tau\ge f^\star$, we get
$c^\top\tau=f^\star$. Thus $\tau$ is optimal and $f_k\uparrow f^\star$, with the gap
$f^\star-f_k\to 0$. $\qquad\blacksquare$

## Rate and practical safeguards

There is no useful a-priori bound on the suboptimality at finite $k$ for general $G$; the
worst-case rate is slow. Near the solution the accumulated cuts become nearly parallel, so the
master LP grows ill-conditioned and the raw "vertex of the relaxed LP" iterate can chatter or
oscillate. A practical safeguard is to prune redundant cuts and keep only the active supports
needed for the current relaxed LP.

## Master-LP implementation

```python
import numpy as np
from scipy.optimize import linprog

def kelley_cutting_plane(f, subgrad, x_lo, x_hi, eps=1e-6, max_iter=500):
    """Minimize convex (possibly nonsmooth) f over the box [x_lo, x_hi].

    f(x)       -> scalar objective value
    subgrad(x) -> a subgradient g in d f(x)  (slope of a supporting hyperplane)

    Decision vector [x ; t] (epigraph). Master LP:
        min t  s.t.  f(x_i) + g_i^T (x - x_i) <= t  for all queries i,  x in box.
    Optimal t is a lower bound on f*; min f(x_i) is an upper bound.
    """
    n = len(x_lo)
    A_ub, b_ub = [], []
    c = np.concatenate([np.zeros(n), [1.0]])           # minimize t
    bounds = list(zip(x_lo, x_hi)) + [(None, None)]    # x in box, t free
    best_x, best_ub = None, np.inf
    x = 0.5 * (np.asarray(x_lo) + np.asarray(x_hi))     # start at box center

    for _ in range(max_iter):
        fx, g = f(x), np.asarray(subgrad(x))            # oracle: value + cut
        if fx < best_ub:
            best_x, best_ub = x.copy(), fx
        # cut: f(x_i) + g^T (x - x_i) <= t  ->  [g, -1].[x;t] <= g^T x_i - f(x_i)
        A_ub.append(np.concatenate([g, [-1.0]]))
        b_ub.append(g @ x - fx)
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        x, lb = res.x[:n], res.x[n]                     # next query + lower bound
        if best_ub - lb <= eps:                         # certified gap
            return best_x, best_ub, lb
    return best_x, best_ub, lb
```
