Let me start from the problem that keeps defeating me and watch where my tools break. I want to minimize a convex function over a convex set — and the cases I actually care about are the ugly ones. The one in front of me is Chebyshev curve fitting: choose coefficients $x$ to make $\max_i \lvert a_i^\top x - b_i\rvert$ as small as possible. The objective is a maximum of linear pieces, so it has kinks all over; at the minimum the active pieces meet at a corner. There is no gradient there to descend along. Gradient projection wants a smooth objective and a projection onto the feasible set; here the objective isn't smooth and the natural feasible set is curved. So the smooth machinery doesn't even get its foot in the door.

What *does* work, reliably and at scale, is linear programming. The simplex method minimizes a linear form over a polyhedron all day long, and the revised version re-solves cheaply when I add a constraint. So the real question isn't "what new descent rule do I invent" — it's "how do I make a curved convex program look like a sequence of linear programs?" If I can do that, I inherit a fast, trustworthy solver for free.

First, let me get the curvature out of the objective and into the constraints, because an LP can only have a linear objective anyway. Suppose I want $\min F(x)$ over $x\in H$ with $F$ convex. Introduce one extra scalar $y$ and write

$$ \min_{x,y}\ y \qquad \text{s.t.}\qquad F(x)\le y,\ \ x\in H,\ \ y\le M, $$

where $M$ is any number above the max of $F$ on $H$. The objective is now the *linear* form $y$, and all the convexity has been pushed into the single convex constraint $F(x)-y\le 0$ together with $x\in H$. The feasible region is the epigraph of $F$ capped at $M$ — still a convex set. So with no loss of generality I can take my problem to be

$$ \min\ c^\top x \qquad \text{s.t.}\qquad x\in R,\quad R = \{x : G(x)\le 0\}, $$

a *linear* form minimized over a convex set carved out by one convex function $G\le 0$. (Several convex constraints fold into one by taking $G$ their pointwise max, which is still convex.) Good — the objective is already in LP shape. The whole fight is now with the curved set $R$.

How does LP see a feasible set? As an intersection of halfspaces, $\{x: Ax\ge b\}$. And here is the fact I should lean on: a closed convex set *is* the intersection of all the halfspaces that contain it. My $R$ is convex, so in principle $R$ is an intersection of linear inequalities — just infinitely many of them, one supporting halfspace per boundary point. If I could feed all of them to the simplex method I'd be done; I obviously can't. But I don't have to feed all of them at once. I can start with a few, solve the LP, and let the solution tell me which halfspace to add next. That is the shape of the thing.

So I need a way, given a point that is *outside* $R$, to manufacture a single linear inequality that throws that point out while keeping every point of $R$. Where would such an inequality come from? From convexity of $G$ itself. Stare at the support inequality. $G$ is convex, so at any point $t$ it lies above its tangent — and where $G$ has a kink, above some supporting affine function. Write that support as

$$ p(x;t) = G(t) + \nabla p(t)\,(x - t), $$

where $\nabla p(t)$ is the gradient when $G$ is smooth at $t$, and otherwise a subgradient — the coefficient row of a supporting hyperplane to the graph of $G$ at $t$. The defining property is the same either way:

$$ p(x;t) \le G(x)\quad\text{for all } x. $$

Now watch what this single affine function does, depending on where $t$ is.

If $x$ is feasible, $x\in R$, then $G(x)\le 0$, so $p(x;t)\le G(x)\le 0$. Every feasible point satisfies $p(x;t)\le 0$.

If $t$ itself is *infeasible*, $t\notin R$, then $p(t;t) = G(t) + \nabla p(t)\,(t-t) = G(t) > 0$. The point $t$ violates $p(\cdot;t)\le 0$.

That's exactly what I wanted. The halfspace $\{x : p(x;t)\le 0\}$ contains all of $R$ and excludes the infeasible point $t$. The hyperplane $p(x;t)=0$ sits between $t$ and the whole feasible set — it cuts $t$ off. One first-order query at $t$ — the value $G(t)$ and one subgradient — buys me a globally valid linear inequality that tightens my outer approximation. And it costs nothing beyond evaluating $G$ and a subgradient, which I can do even at a kink: if $G=\max_i g_i$, a subgradient at $t$ is $\nabla g_h(t)$ for any active index $h$, so the nonsmooth case is no obstacle. Convexity is doing all the work — without it the support inequality $p\le G$ fails and the cut could slice off feasible points.

Notice this is *not* "delete the bad point." Deleting one point from a continuum does nothing for an LP — the optimum would just slide to its neighbor. The power is that one subgradient deletes an entire halfspace's worth of bad points at once while provably sparing $R$. That's why this works where pointwise patching wouldn't.

Now I have the loop. Enclose $R$ in a polyhedron I can write down — say a box $S = \{x : Ax\ge b\}$ big enough to contain $R$ (in practice $\lvert x_j\rvert \le \lambda$ for some bound). Call it $S_0$. Minimize the linear form over it:

$$ t_0 = \arg\min\{c^\top x : x\in S_0\}. $$

If $t_0$ happens to be feasible, it is not merely a good point; it is optimal, because $R\subseteq S_0$ and no point of the smaller set $R$ can beat the minimum over $S_0$. Generally it isn't feasible — it's some vertex of the box, sticking out beyond $R$. So query $G$ at $t_0$, get the cut $p(x;t_0)\le 0$, and intersect it in:

$$ S_1 = S_0 \cap \{x : p(x;t_0)\le 0\}. $$

Re-solve the LP over $S_1$ to get $t_1$, cut again, and in general

$$ S_k = S_{k-1}\cap\{x : p(x;t_{k-1})\le 0\},\qquad t_k = \arg\min\{c^\top x : x\in S_k\}. $$

Each $t_k$ is the solution of a linear program: minimize $c^\top x$ subject to $Ax\ge b$ and, for every earlier query $0\le i\le k-1$,

$$ G(t_i) + \nabla p(t_i)\,(x - t_i)\le 0 \quad\Longleftrightarrow\quad -\nabla p(t_i)\,x \ge G(t_i) - \nabla p(t_i)\,t_i. $$

So the algorithm is a sequence of LPs, each one the previous LP with one extra row. It's an outgrowth of the Chebyshev-fitting work — there, $R$ comes from the linear pieces of the max, and the cuts are exactly the active pieces, which is why the picture felt familiar. And structurally it echoes the architecture Gomory uses for integer programs — solve the LP, cut off the unwanted optimum with a separating linear inequality, re-solve — only the cut is generated here from convexity of $G$ instead of from integrality of a tableau row. (The parallel is close enough that I suspect one could combine them to attack integer convex programs, cutting off lattice points the same way; I haven't worked that out and won't here.)

Before I trust it I have to ask whether re-solving from scratch each time is wasteful, because the LP grows by a row every step. Look at the dual. The primal is $\min c^\top x$ over $Ax\ge b$ and the accumulated cuts $-\nabla p(t_i)\,x \ge G(t_i)-\nabla p(t_i)\,t_i$. Dualizing, with a multiplier vector $u\ge 0$ on $Ax\ge b$ and a scalar $v_i\ge 0$ on each cut,

$$ \max\ u^\top b + \sum_{i} v_i\big(G(t_i) - \nabla p(t_i)\,t_i\big)\quad\text{s.t.}\quad u^\top A - \sum_i v_i\,\nabla p(t_i) = c,\ \ u\ge0,\ v_i\ge 0. $$

Adding a cut to the primal adds exactly one variable $v_k$ to the dual. So the previous dual solution — pad it with $v_k=0$ — is a feasible starting point for the next dual; one step of dual simplex moves me to the new optimum. The transition from $k$ to $k+1$ costs a single new column plus a subroutine that evaluates $\nabla p(t_k)$. That folds straight into the revised-simplex codes that already exist; I only bolt on the cut generator. Good — the loop is cheap.

Does it converge? This is where I have to be careful, because "intuitively the box shrinks toward $R$" is not a proof. Let $f_k = c^\top t_k = \min\{c^\top x : x\in S_k\}$ and let $f^\star = \min\{c^\top x : x\in R\}$ be what I'm after.

Two bracketing facts fall out immediately from how the sets nest. First, I only ever *add* cuts, so $S_k\subseteq S_{k-1}$, and minimizing over a smaller set can only raise the minimum: $f_k\ge f_{k-1}$. The model values are nondecreasing. Second, every cut is satisfied by all of $R$, so $R\subseteq S_k$ for every $k$, and minimizing over a *larger* set can only lower the minimum: $f_k = \min_{S_k} c^\top x \le \min_{R} c^\top x = f^\star$. So

$$ f_0 \le f_1 \le \cdots \le f_k \le f^\star : $$

the model values increase and are bounded above by $f^\star$, hence they converge to some limit $\le f^\star$. I'd like that limit to *be* $f^\star$, and I'd like the iterates $t_k$ to approach a genuine minimizer in $R$. That's the real content.

Here's the lever. Because $t_k$ minimizes $c^\top x$ over $S_k$, and the constraints defining $S_k$ include every earlier cut $p(\cdot;t_i)\le 0$ for $i<k$, the point $t_k$ must satisfy all of them:

$$ G(t_i) + \nabla p(t_i)\,(t_k - t_i)\le 0,\qquad 0\le i\le k-1. \tag{$\ast$}$$

If some $t_k$ is feasible, the same argument as at $t_0$ ends the matter: since $R\subseteq S_k$ and $t_k$ minimizes over $S_k$, feasibility forces $c^\top t_k=f^\star$. So the only case that still needs proof is the infinite one where every generated $t_k$ remains outside $R$, hence $G(t_k)>0$.

Now suppose, toward a contradiction, that these positive violations do *not* die out. Concretely, suppose there is some fixed $r>0$ with $G(t_k)\ge r$ for infinitely many $k$. Take two such indices $i<k$. Rearranging $(\ast)$,

$$ r \le G(t_i) \le -\nabla p(t_i)\,(t_k - t_i) = \nabla p(t_i)\,(t_i - t_k) \le \lVert \nabla p(t_i)\rVert\,\lVert t_i - t_k\rVert. $$

The middle equality is just moving the term across; the last step is Cauchy–Schwarz. And here is where the bounded-support assumption earns its keep: on the compact working set I require $\lVert \nabla p(t_i)\rVert \le K$ for some finite $K$, all $i$. Therefore

$$ r \le K\,\lVert t_i - t_k\rVert \quad\Longrightarrow\quad \lVert t_i - t_k\rVert \ge \frac{r}{K}. $$

So any two of these high-violation iterates are at least $r/K$ apart. That means the subsequence of $\{t_k\}$ with $G(t_k)\ge r$ is uniformly separated — no two of its points come within $r/K$ — so it contains no Cauchy subsequence and no convergent subsequence at all. But every one of these points lives in $S$, and $S$ is compact, so by Bolzano–Weierstrass it *must* contain a convergent subsequence. Contradiction.

So the supposition fails: in the nonterminating case, the positive violations must converge to zero. Compactness gives a convergent subsequence $t_{k_j}\to\tau\in S$, and continuity gives $G(\tau)=0$, so $\tau\in R$. Along that subsequence $c^\top t_{k_j}=f_{k_j}$ converges to $c^\top\tau$; but $\{f_k\}$ increases to its limit $\le f^\star$, and since $\tau\in R$ we also have $c^\top\tau \ge f^\star$. The only way both hold is $c^\top\tau = f^\star$. So $\tau$ is an optimal solution, the model values increase exactly to $f^\star$, and the gap $f^\star - f_k$ closes to zero.

That is the convergence theorem, and the two ingredients it leans on are precisely the two I assumed: a uniform Lipschitz bound $K$ on the supports (to turn a stubborn violation into a fixed separation $r/K$) and compactness of $S$ (to forbid an infinite, uniformly separated set). The mechanism is almost tactile: the only way the method can keep generating infeasible iterates with violation $\ge r$ is to keep them $\ge r/K$ apart, and a bounded region simply runs out of room. Each cut, evaluated at a *later* iterate through $(\ast)$, is what pins this down.

The proof is reassuring, but I've talked myself into believing geometric pictures before. Let me actually run the loop on a small case and watch the numbers, both to catch a bookkeeping slip in the cut algebra and to see whether the monotone-climb claim really holds step by step. Minimize $f = x_1 - x_2$ over the ellipse $G(x) = 3x_1^2 - 2x_1x_2 + x_2^2 - 1\le 0$. By eye the minimum is at $(0,1)$ with $f^\star=-1$. The gradient ($G$ is smooth here) is $\nabla p(t) = (6t_1 - 2t_2,\ -2t_1 + 2t_2)$, and the cut is $p(x;t)=G(t)+\nabla p(t)(x-t)\le 0$. Box it: $S_0 = \{-2\le x_1,x_2\le 2\}$.

Step $0$ by hand. The LP $\min x_1-x_2$ on the box sits at the corner $t_0=(-2,2)$, $f_0=-4$ — well below the true $-1$, and far outside the ellipse. Query: $G(t_0)=3\cdot4 - 2\cdot(-2)(2) + 4 - 1 = 12+8+4-1=23>0$, and $\nabla p(t_0)=(6(-2)-2(2),\,-2(-2)+2(2)) = (-16,\,8)$. Writing the cut as a row $g^\top x\le g^\top t_0 - G(t_0)$ gives $-16x_1+8x_2\le (-16)(-2)+8\cdot2-23 = 32+16-23 = 25$, i.e. $-16x_1+8x_2-25\le0$ — the form I expected. Feeding the box plus this one row to the LP returns $t_1=(-0.5625,\,2)$, $f_1=-2.5625$, up from $-4$ toward $-1$.

Rather than push the arithmetic further by hand I let the loop run and tabulate $f_k$ and the violation $G(t_k)$:

| $k$ | $t_k$ | $f_k$ | $G(t_k)$ |
|----|---------------------|-----------|-----------|
| 0 | $(-2.000,\ 2.000)$ | $-4.0000$ | $23.0000$ |
| 1 | $(-0.5625,\ 2.000)$ | $-2.5625$ | $6.1992$ |
| 2 | $(\ 0.2781,\ 2.000)$ | $-1.7219$ | $2.1197$ |
| 3 | $(-0.5297,\ 0.8376)$ | $-1.3673$ | $1.4306$ |
| 4 | $(-0.0531,\ 1.1603)$ | $-1.2134$ | $0.4779$ |
| 5 | $(\ 0.4267,\ 1.4851)$ | $-1.0584$ | $0.4844$ |
| 6 | $(\ 0.1707,\ 1.2067)$ | $-1.0360$ | $0.1316$ |
| 7 | $(\ 0.0184,\ 1.0411)$ | $-1.0227$ | $0.0466$ |
| 8 | $(-0.1661,\ 0.8404)$ | $-1.0066$ | $0.0683$ |
| 9 | $(-0.0734,\ 0.9298)$ | $-1.0032$ | $0.0172$ |
| 10 | $(-0.0263,\ 0.9752)$ | $-1.0015$ | $0.0044$ |
| 11 | $(-0.0012,\ 0.9994)$ | $-1.0006$ | $0.0013$ |

Two things in this table are exactly what the proof predicted and one is a useful surprise. The predicted parts: $f_k$ is monotone nondecreasing — $-4.0000, -2.5625, -1.7219, \dots$ — and it stays *below* $f^\star=-1$ at every row (the closest is $-1.0006$), so each $f_k$ is a genuine lower bracket on the optimum, climbing toward it. The iterates $t_k$ visibly close in on $(0,1)$. The surprise is in the $G(t_k)$ column: the violation is *not* monotone. From $k=4$ to $k=5$ it rises, $0.4779\to0.4844$, and from $k=7$ to $k=8$ it rises again, $0.0466\to0.0683$. So a step that improves the objective bracket can move me to a *more* infeasible point. That kills any temptation to use "$G(t_k)$ decreased" as a progress measure or stopping rule — and it matches what the proof actually claims, which is only that the violations cannot stay bounded away from zero forever, never that they fall every step. Good: the run confirms the bookkeeping and sharpens, rather than contradicts, the theorem.

Now the honest trouble is the lack of a finite-step rate. I have no a-priori bound on how *fast* $f_k\to f^\star$ for a general $G$ — only that it does. It is tempting to stop the first time $G(t_k)$ drops below a tolerance, since then $t_k$ is nearly feasible. But that is not a certificate in general: $t_k$ can be nearly feasible yet still far from the minimizer, so a small $G(t_k)$ doesn't certify a small objective gap. What I'd really like is a two-sided certificate.

Let me see if I can extract a true *lower* bound on $f^\star$ from the model itself, not just an upper one from the best feasible point. Go back to the epigraph view and think of the objective $f_0$ and any constraints $f_i$ directly as convex functions I'm querying. At the points $x^{(1)},\dots,x^{(q)}$ I've evaluated, I have a value and a subgradient of each, so for every $z$,

$$ f(z)\ \ge\ f(x^{(i)}) + g^{(i)\top}(z - x^{(i)}),\qquad i=1,\dots,q, $$

and taking the max over $i$,

$$ f(z)\ \ge\ \hat f(z) := \max_{i=1,\dots,q}\Big(f(x^{(i)}) + g^{(i)\top}(z - x^{(i)})\Big). $$

So $\hat f$ — the pointwise max of the support planes — is a convex, piecewise-linear function lying *below* $f$ everywhere: a global underestimator built entirely from first-order data. This is the same family of cuts as before, now read as a lower *model* of the function rather than only as feasibility cuts. Minimizing this PWL lower model (with the polyhedral constraints) is an LP — via the epigraph trick, $\min t$ s.t. $f(x^{(i)})+g^{(i)\top}(z-x^{(i)})\le t$ for all $i$ — and because $\hat f\le f$ everywhere, *the optimal value of this relaxed LP is a lower bound on $f^\star$.* Its minimizer is naturally my next query point: I am minimizing the best lower model I currently have, which is exactly the "vertex of the relaxed problem" the constraint-generation loop was already producing.

So at every step I get both bounds for free: the relaxed-LP value $\hat f^{(k)}$ is a lower bound $\ell_k$ on $f^\star$, the best feasible objective so far is an upper bound $u_k$, and I stop when $u_k - \ell_k\le\epsilon$. The lower bounds increase as cuts are added, the recorded upper bound can only improve when a better feasible point appears, and the gap gives a certificate instead of relying on the unsafe rule "stop when $G(t_k)$ is small."

I should check this two-sided rule does the right thing on a *nonsmooth* objective — the case the smooth machinery couldn't touch — since that's where I want it to earn its keep. Take a small Chebyshev fit, $f(x)=\max\big(\lvert x_1-1\rvert,\ \lvert x_1+x_2\rvert,\ \lvert x_2-2\rvert\big)$ on the box $[-5,5]^2$. The subgradient at any point is the signed coefficient row of whichever piece is active. Running the epigraph master LP with the gap stop at $\epsilon=10^{-7}$, it returns $x=(0,1)$ with $u_k=\ell_k=1.0$ — the bounds meet, gap $0$, in $4$ iterations. A grid sweep of $f$ over $[-3,3]^2$ bottoms out at the same value $1.0$ at $(0,1)$, so the certificate wasn't lying: the method located the true nonsmooth minimum and *proved* it had, with no gradient anywhere in sight. That the lower and upper bounds coincided exactly (rather than merely came within $\epsilon$) is the polyhedral nature of this particular $f$ showing through — its support planes are the actual pieces, so finitely many cuts reproduce $f$ exactly near the optimum.

The piecewise-linear model also tells me where the method strains. The next query is a *vertex* of the relaxed polyhedron — a corner. Corners are extreme; nothing keeps consecutive queries from jumping to opposite faces. Worse, as the cuts accumulate near the solution they become nearly parallel to one another (they're all supporting the same smooth piece of boundary), so the LP that locates $t_k$ gets ill-conditioned: the determination of $t_k$ becomes delicate under finite precision, and the process can degenerate, oscillate, or even drift to the wrong point. And the cut count only grows — every step adds a row, so the LPs get heavier; many old cuts become redundant and ought to be pruned, a cut being truly *needed* only when dropping it would let the relaxed problem run unbounded. So two practical patches: drop redundant cuts to keep the LP light, and accept that the raw method can chatter near the end.

So the method I've landed on: reduce to minimizing a linear form over a convex set; build a piecewise-linear lower model of the convexity by collecting one support plane per query — each a valid cut because convexity makes the support a global underestimator; minimize that model with an LP to get the next iterate and a lower bound; add its cut; and stop on the gap between that lower bound and the best feasible value. The model tightens monotonically, the model values increase to $f^\star$, and compactness plus the Lipschitz bound force the gap to zero. The causal chain is short: convexity gives a support inequality at every point; a support at an infeasible point is a linear cut that keeps all of $R$ and discards that point; an LP over the accumulated cuts is solvable and warm-startable; the nested feasible sets make the model values monotone and bounded by $f^\star$; and the only escape from convergence — iterates that stay infeasible — would force an infinite, uniformly separated set inside a compact region, which cannot exist.

```python
import numpy as np
from scipy.optimize import linprog

def kelley_cutting_plane(f, subgrad, x_lo, x_hi, eps=1e-6, max_iter=500):
    """Minimize a convex (possibly nonsmooth) f over the box [x_lo, x_hi]
    by Kelley's cutting-plane method.

    f(x)        -> scalar value of the convex objective
    subgrad(x)  -> a subgradient g in d f(x)  (the supporting hyperplane's slope)

    Epigraph variable t; decision vector is [x ; t].
    Master LP: min t  s.t.  f(x_i) + g_i^T (x - x_i) <= t   for all queries i,
               and x in [x_lo, x_hi].
    The optimal t is a lower bound on f*; min f(x_i) is an upper bound.
    """
    n = len(x_lo)
    # cuts:  f(x_i) + g_i^T (x - x_i) - t <= 0  ->  [g_i, -1] . [x;t] <= g_i^T x_i - f(x_i)
    A_ub, b_ub = [], []
    c = np.concatenate([np.zeros(n), [1.0]])          # minimize t
    bounds = list(zip(x_lo, x_hi)) + [(None, None)]   # x in box, t free
    best_x, best_ub = None, np.inf
    x_query = 0.5 * (np.asarray(x_lo) + np.asarray(x_hi))  # start at box center

    for _ in range(max_iter):
        fx = f(x_query)                  # oracle: value
        g = np.asarray(subgrad(x_query)) # oracle: supporting hyperplane
        if fx < best_ub:
            best_x, best_ub = x_query.copy(), fx

        # add the support plane as a cut (a valid global lower bound on f)
        A_ub.append(np.concatenate([g, [-1.0]]))
        b_ub.append(g @ x_query - fx)

        # solve the master LP over the piecewise-linear lower model
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(res.message)
        x_query, lb = res.x[:n], res.x[n]   # next query, and lower bound t = f_hat-min

        if best_ub - lb <= eps:             # certified gap -> stop
            return best_x, best_ub, lb

    return best_x, best_ub, lb
```
