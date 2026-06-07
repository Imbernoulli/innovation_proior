OK, let me start from the thing that's actually blocking me. I have a convex problem whose objective is a sum, $\min f(x)+g(z)$ subject to a linear coupling $Ax+Bz=c$, and I want a method that does two things at once: it has to be *robust* — converge when $f$ or $g$ is nonsmooth, or affine, or takes the value $+\infty$ (so I can hide constraints inside them as indicators), with no strict-convexity or full-rank crutch — and it has to *decompose*, because the data and the variable are huge and physically spread across machines, so each iteration's work must split into independent pieces with only a cheap message-pass between them. The trouble is that the two demands pull against each other in every classical tool I know. Let me lay the tools out and watch exactly where each one breaks, because the break points are the design spec for whatever I build.

Take the plain equality-constrained problem first, $\min f(x)$ s.t. $Ax=b$, and the most classical move: dualize. The Lagrangian is $L(x,y)=f(x)+y^\top(Ax-b)$, the dual function is $g(y)=\inf_x L(x,y)$, and if I work it out, $g(y)=-f^*(-A^\top y)-b^\top y$ with $f^*$ the conjugate. The dual is concave, so maximize it by gradient ascent. The beautiful part is that the gradient is free: to evaluate $\nabla g(y)$ I find $x^+=\arg\min_x L(x,y)$, and then $\nabla g(y)=Ax^+-b$, the constraint residual. So dual ascent is

$$ x^{k+1}=\arg\min_x L(x,y^k),\qquad y^{k+1}=y^k+\alpha^k(Ax^{k+1}-b). $$

Minimize over $x$ to get a residual, step the price $y$ up the residual, repeat. And it has a gift hidden in it: if $f$ is *separable*, $f(x)=\sum_i f_i(x_i)$ with $A=[A_1\cdots A_N]$ partitioned to match, then $L(x,y)=\sum_i\big(f_i(x_i)+y^\top A_i x_i\big)-y^\top b$ is itself separable in $x$, so the $x$-minimization shatters into $N$ independent little problems $x_i^{k+1}=\arg\min_{x_i}\{f_i(x_i)+y^\top A_i x_i\}$ that I solve in parallel — broadcast the one price vector $y$, let each worker grind on its own $f_i$, gather the residual contributions $A_i x_i$. That's dual decomposition, and it's exactly the parallel structure I want.

So why don't I just use it? Stare at the $x$-update. It's an *unconstrained* minimization of $f(x)+y^\top Ax$, and that is only well-posed if $f$ curves up hard enough to pin the minimizer down. If $f$ is merely convex but not strictly convex — worse, if $f$ is *affine* in even one coordinate — then for almost every $y$ the linear term $y^\top Ax$ tips the whole thing and $L$ is unbounded below in $x$: the $\arg\min$ doesn't exist, the update returns nothing. And nonsmooth or extended-valued $f$ make it worse still; I'd be forced onto dual *subgradient* steps with finicky step sizes and nonmonotone ascent. So dual decomposition gives me the parallelism but is brittle exactly where my objectives live — at $\ell_1$ norms, indicators, affine losses. Decomposition: yes. Robustness: no.

How did people make the $x$-update robust? The fix is to stop letting the linear term run away by gluing a quadratic penalty onto the constraint. Augment the Lagrangian:

$$ L_\rho(x,y)=f(x)+y^\top(Ax-b)+\tfrac{\rho}{2}\|Ax-b\|_2^2,\qquad \rho>0. $$

The way to *read* this — and the reason it's legitimate — is that $L_\rho$ is just the ordinary Lagrangian of a *different but equivalent* problem, $\min f(x)+\tfrac{\rho}{2}\|Ax-b\|^2$ s.t. $Ax=b$, because on the feasible set the penalty is zero, so the modified problem has the same solution. The penalty doesn't change the answer; it changes the *geometry of the dual*. With that strongly-convexifying $\tfrac{\rho}{2}\|Ax-b\|^2$ in the objective, the inner $\inf_x$ is well-defined and the dual function $g_\rho$ becomes differentiable under mild conditions even when the un-augmented dual wasn't. Now run dual ascent on the augmented problem and you get the method of multipliers:

$$ x^{k+1}=\arg\min_x L_\rho(x,y^k),\qquad y^{k+1}=y^k+\rho(Ax^{k+1}-b). $$

Notice the dual step size is $\rho$ itself, not a free $\alpha$. That's not an aesthetic choice; it's forced, and it's worth seeing why. Since $x^{k+1}$ minimizes $L_\rho(\cdot,y^k)$, set its gradient to zero: $0=\nabla f(x^{k+1})+A^\top y^k+\rho A^\top(Ax^{k+1}-b)=\nabla f(x^{k+1})+A^\top\big(y^k+\rho(Ax^{k+1}-b)\big)$. If I *define* $y^{k+1}$ to be exactly that bracket, $y^{k+1}=y^k+\rho(Ax^{k+1}-b)$, then the line collapses to $0=\nabla f(x^{k+1})+A^\top y^{k+1}$ — which is precisely the dual-feasibility condition. So choosing the dual step equal to $\rho$ makes every single iterate dual feasible; all that's left to reach optimality is to drive the primal residual $Ax^{k+1}-b$ to zero, which the iteration does. And the method of multipliers converges under *far* weaker conditions than dual ascent: $f$ can be nonsmooth, can be $+\infty$, can fail strict convexity. The penalty bought robustness.

But now watch the parallelism evaporate. Suppose $f=\sum_i f_i(x_i)$, the separable case I cared about. The first two terms of $L_\rho$ are still separable. The penalty is not. Write $Ax-b=\sum_i A_i x_i - b$ and square it: $\|\sum_i A_i x_i - b\|^2$ expands into cross terms $\sum_{i\ne j}(A_i x_i)^\top(A_j x_j)$ that *bind every block to every other block*. So $L_\rho$ is not separable; the joint $x$-minimization can no longer be peeled apart across processors. The very penalty that made the method robust is the thing that re-coupled the variables and killed decomposition. So I've traced the pincer exactly: dual decomposition is parallel but fragile; the method of multipliers is robust but serial. The quadratic penalty is simultaneously the source of robustness and the destroyer of separability.

That's the knot. Let me think about what's really doing the damage. It's the *joint* $x$-minimization of an objective whose coupling term mixes the blocks. What if I don't have to minimize jointly?

Here's the reframing that opens it up. My objective genuinely has two natural halves, $f$ on one set of variables and $g$ on another — that's the structure I started with, $f(x)+g(z)$ with $Ax+Bz=c$. (And the separable consensus case is just this with the right copies; I'll come back to that.) Form the augmented Lagrangian of the *two-block* problem,

$$ L_\rho(x,z,y)=f(x)+g(z)+y^\top(Ax+Bz-c)+\tfrac{\rho}{2}\|Ax+Bz-c\|_2^2. $$

The method of multipliers would tell me to minimize this *jointly* over $(x,z)$ and then bump $y$. And joint minimization is exactly where the cross term $\rho\,(Ax)^\top(Bz)$ in the expanded penalty couples $x$ and $z$ and forces me to solve them together. But I don't actually need the joint minimizer to make progress on the dual. What if I minimize over $x$ with $z$ held fixed, then minimize over $z$ with the new $x$ held fixed, then update $y$? One sweep — first $x$, then $z$ — a single Gauss–Seidel pass over the two blocks instead of a joint solve:

$$ x^{k+1}=\arg\min_x L_\rho(x,z^k,y^k),\qquad z^{k+1}=\arg\min_z L_\rho(x^{k+1},z,y^k),\qquad y^{k+1}=y^k+\rho(Ax^{k+1}+Bz^{k+1}-c). $$

Look at what the alternation does to the coupling. In the $x$-update, $z=z^k$ is *frozen*, so the penalty $\tfrac{\rho}{2}\|Ax+Bz^k-c\|^2$ is a clean quadratic in $x$ alone — the offending cross term $\rho(Ax)^\top(Bz)$ becomes a *linear* term in $x$ (because $Bz^k$ is now a constant), not a bilinear coupling. The $x$-update is a self-contained problem in $x$; the $z$-update, symmetrically, is self-contained in $z$. So the alternation keeps the robustness of the augmented Lagrangian — the penalty is still there pinning the inner minimizations down — while *restoring* the ability to handle each block on its own, and when $f$ or $g$ is itself separable, the corresponding block update splits further across processors. I get both. The thing I gave up is the *joint* minimization, replacing it with one Gauss–Seidel sweep, and the bet — which I'll have to verify with a convergence proof — is that one sweep per iteration is enough.

Let me sanity-check that this is more than wishful alternation by checking the dual update is still doing the right thing. The dual step is again $\rho$, the penalty parameter, just as in the method of multipliers, and I'll see in a moment that this makes one of the optimality conditions hold automatically at every iterate — the analogue of the dual-feasibility gift from before. The state I'm carrying forward is $(z^k,y^k)$; $x^{k+1}$ is just an intermediate quantity computed from them. Note the roles of $x$ and $z$ are *almost* symmetric but not quite — the dual update happens after $z$ but before the next $x$ — so the order of the sweep matters a little.

Now, the linear-plus-quadratic bookkeeping in each subproblem is ugly, and there's a standard trick to clean it that also exposes the meaning of the dual variable. The two terms involving the residual $r=Ax+Bz-c$ are $y^\top r+\tfrac{\rho}{2}\|r\|^2$. Complete the square:

$$ y^\top r+\tfrac{\rho}{2}\|r\|^2=\tfrac{\rho}{2}\Big\|r+\tfrac1\rho y\Big\|^2-\tfrac{1}{2\rho}\|y\|^2=\tfrac{\rho}{2}\|r+u\|^2-\tfrac{\rho}{2}\|u\|^2,\qquad u:=\tfrac1\rho y. $$

The leftover $-\tfrac{\rho}{2}\|u\|^2$ doesn't depend on $x$ or $z$, so it falls out of both $\arg\min$s. Define the **scaled dual variable** $u=y/\rho$, and the iterations become

$$ x^{k+1}=\arg\min_x\Big\{f(x)+\tfrac{\rho}{2}\|Ax+Bz^k-c+u^k\|_2^2\Big\}, $$
$$ z^{k+1}=\arg\min_z\Big\{g(z)+\tfrac{\rho}{2}\|Ax^{k+1}+Bz-c+u^k\|_2^2\Big\}, $$
$$ u^{k+1}=u^k+(Ax^{k+1}+Bz^{k+1}-c). $$

Each subproblem is now a single squared term, much cleaner. And the scaled dual update has a transparent reading: $u^{k+1}=u^k+r^{k+1}$, so $u^k=u^0+\sum_{j\le k}r^j$ is the *running sum of all primal residuals* — the accumulated constraint violation, which is exactly what a price should be. I'll carry this scaled form forward; it's shorter and it's what the code will use.

Before I trust any of this I need two things: the optimality conditions it's chasing, and a stopping test built from how far the current iterate is from them. Let me derive the optimality conditions and, in the process, find the residuals. For $\min f(x)+g(z)$ s.t. $Ax+Bz=c$, a point $(x^\star,z^\star,y^\star)$ is optimal iff it's primal feasible,

$$ Ax^\star+Bz^\star-c=0, $$

and dual feasible — stationarity of the Lagrangian in each block (using subdifferentials, since $f,g$ may be nonsmooth):

$$ 0\in\partial f(x^\star)+A^\top y^\star,\qquad 0\in\partial g(z^\star)+B^\top y^\star. $$

Three conditions. Now here's the payoff of the alternation and the $\rho$-step: let me check which of these the ADMM iterates satisfy *for free*. Start with the $z$-update. By definition $z^{k+1}$ minimizes $L_\rho(x^{k+1},z,y^k)$, so its optimality condition (in unscaled $y$) is

$$ 0\in\partial g(z^{k+1})+B^\top y^k+\rho B^\top(Ax^{k+1}+Bz^{k+1}-c)=\partial g(z^{k+1})+B^\top y^k+\rho B^\top r^{k+1}. $$

But $y^{k+1}=y^k+\rho r^{k+1}$, so the last two terms are exactly $B^\top y^{k+1}$, giving $0\in\partial g(z^{k+1})+B^\top y^{k+1}$. That's the *second* dual-feasibility condition, holding at every iterate after the $z$- and $y$-updates. So just like the method of multipliers always being dual feasible, ADMM always satisfies $0\in\partial g(z^{k+1})+B^\top y^{k+1}$. Two conditions left to chase: primal feasibility and the *first* dual-feasibility condition.

The primal one gives the obvious residual: $r^{k+1}=Ax^{k+1}+Bz^{k+1}-c$, the **primal residual**, which is zero exactly at feasibility. For the other, look at the $x$-update. $x^{k+1}$ minimizes $L_\rho(x,z^k,y^k)$ — note it used the *old* $z^k$, not $z^{k+1}$ — so

$$ 0\in\partial f(x^{k+1})+A^\top y^k+\rho A^\top(Ax^{k+1}+Bz^k-c). $$

I want to compare this to the target $0\in\partial f(x^{k+1})+A^\top y^{k+1}$, so I need to manufacture $y^{k+1}=y^k+\rho r^{k+1}$ out of what I have. The bracket has $Bz^k$ where I'd want $Bz^{k+1}$; bridge the gap by adding and subtracting $\rho A^\top B z^{k+1}$:

$$ 0\in\partial f(x^{k+1})+A^\top\big(y^k+\rho(Ax^{k+1}+Bz^{k+1}-c)\big)+\rho A^\top B(z^k-z^{k+1})=\partial f(x^{k+1})+A^\top y^{k+1}+\rho A^\top B(z^k-z^{k+1}). $$

Rearrange: $\rho A^\top B(z^{k+1}-z^k)\in\partial f(x^{k+1})+A^\top y^{k+1}$. The right side is exactly the quantity that *should* be zero at optimality (the first dual-feasibility residual), and it equals the left side. So the residual for the first dual condition is

$$ s^{k+1}:=\rho A^\top B(z^{k+1}-z^k), $$

the **dual residual** — it measures how much $z$ moved over the last step, mapped through $\rho A^\top B$. When $z$ stops changing, $s\to0$ and the first dual condition is met. So the two quantities I need to watch are $r^{k+1}=Ax^{k+1}+Bz^{k+1}-c$ and $s^{k+1}=\rho A^\top B(z^{k+1}-z^k)$; the third optimality condition holds automatically. That's clean and it's the natural stopping signal: stop when both residuals are small.

To make "small" precise I want to know that small residuals actually bound suboptimality. Let me get that bound, because it's what justifies the stopping rule. I'll need it from the convergence analysis, so let me set the analysis up properly now and harvest the bound along the way. Pick a saddle point $(x^\star,z^\star,y^\star)$ of the *unaugmented* Lagrangian $L_0$ (its existence is my only real assumption, beyond $f,g$ closed proper convex; note I assume nothing about $A,B,c$ — no full rank). Define the candidate Lyapunov function

$$ V^k=\tfrac1\rho\|y^k-y^\star\|_2^2+\rho\|B(z^k-z^\star)\|_2^2. $$

It's a combination of how far the dual is from optimal and how far $Bz$ is from optimal — nonnegative, and unknown while running (it depends on the unknowns $z^\star,y^\star$), but a legitimate analysis object. The whole proof rides on three inequalities; let me derive them.

Write $p^k=f(x^k)+g(z^k)$ and $p^\star$ the optimal value. **Inequality (A3):** since $(x^\star,z^\star,y^\star)$ is a saddle point, $L_0(x^\star,z^\star,y^\star)\le L_0(x^{k+1},z^{k+1},y^\star)$. The left side is $f(x^\star)+g(z^\star)+ (y^\star)^\top(Ax^\star+Bz^\star-c)=p^\star$ (the residual vanishes at feasibility). The right side is $p^{k+1}+(y^\star)^\top r^{k+1}$. So

$$ p^\star\le p^{k+1}+(y^\star)^\top r^{k+1}\quad\Longleftrightarrow\quad p^\star-p^{k+1}\le (y^\star)^\top r^{k+1}.\tag{A3} $$

**Inequality (A2):** I use that the updates are exact minimizers. From the $x$-update I derived $0\in\partial f(x^{k+1})+A^\top y^{k+1}-\rho A^\top B(z^{k+1}-z^k)$, i.e. $x^{k+1}$ minimizes $f(x)+\big(y^{k+1}-\rho B(z^{k+1}-z^k)\big)^\top Ax$ — convex, so comparing its value at $x^{k+1}$ against $x^\star$,

$$ f(x^{k+1})+\big(y^{k+1}-\rho B(z^{k+1}-z^k)\big)^\top Ax^{k+1}\le f(x^\star)+\big(y^{k+1}-\rho B(z^{k+1}-z^k)\big)^\top Ax^\star. $$

Likewise $z^{k+1}$ minimizes $g(z)+(y^{k+1})^\top Bz$ (the second dual condition I proved above is exactly this stationarity), so

$$ g(z^{k+1})+(y^{k+1})^\top Bz^{k+1}\le g(z^\star)+(y^{k+1})^\top Bz^\star. $$

Add the two, use $Ax^\star+Bz^\star=c$ to collapse the right-hand combination, and rearrange — the algebra is bookkeeping but it lands at

$$ p^{k+1}-p^\star\le -(y^{k+1})^\top r^{k+1}-\rho\big(B(z^{k+1}-z^k)\big)^\top\big(-r^{k+1}+B(z^{k+1}-z^\star)\big).\tag{A2} $$

Now the suboptimality bound I wanted for stopping. Look at the factor $-r^{k+1}+B(z^{k+1}-z^\star)$ sitting inside (A2); there's a clean identity for it. Because $r^{k+1}=Ax^{k+1}+Bz^{k+1}-c$ and $Ax^\star+Bz^\star=c$, expand $-r^{k+1}+B(z^{k+1}-z^\star) = -Ax^{k+1}-Bz^{k+1}+c+Bz^{k+1}-Bz^\star = -Ax^{k+1}+c-Bz^\star = -Ax^{k+1}+Ax^\star = -A(x^{k+1}-x^\star)$. Plug that into (A2):

$$ p^{k+1}-p^\star\le -(y^{k+1})^\top r^{k+1}-\rho\big(B(z^{k+1}-z^k)\big)^\top\big(-A(x^{k+1}-x^\star)\big)=-(y^{k+1})^\top r^{k+1}+(x^{k+1}-x^\star)^\top\,\rho A^\top B(z^{k+1}-z^k). $$

The dual-residual term reappears because $\rho\big(B(z^{k+1}-z^k)\big)^\top A(x^{k+1}-x^\star)=(x^{k+1}-x^\star)^\top\rho A^\top B(z^{k+1}-z^k)$.

The second term is $(x^{k+1}-x^\star)^\top s^{k+1}$. So

$$ f(x^k)+g(z^k)-p^\star\le -(y^k)^\top r^k+(x^k-x^\star)^\top s^k. $$

There's the suboptimality bound in terms of the two residuals: when $r^k$ and $s^k$ are small, the objective gap is small. I can't use it literally because $x^\star$ is unknown, but if I bound $\|x^k-x^\star\|\le d$ then $f(x^k)+g(z^k)-p^\star\le \|y^k\|\,\|r^k\|+d\,\|s^k\|$. So a sensible stopping rule is exactly "both residuals small": $\|r^k\|_2\le\varepsilon^{\text{pri}}$ and $\|s^k\|_2\le\varepsilon^{\text{dual}}$. In practice I set those tolerances with an absolute and a relative part scaled by the sizes of the quantities involved,

$$ \varepsilon^{\text{pri}}=\sqrt{p}\,\varepsilon^{\text{abs}}+\varepsilon^{\text{rel}}\max\{\|Ax^k\|,\|Bz^k\|,\|c\|\},\qquad \varepsilon^{\text{dual}}=\sqrt{n}\,\varepsilon^{\text{abs}}+\varepsilon^{\text{rel}}\|A^\top y^k\|, $$

the $\sqrt{p},\sqrt{n}$ accounting for the dimensions of the two residual vectors, with $\varepsilon^{\text{rel}}$ around $10^{-3}$–$10^{-4}$.

I still owe myself the *convergence* — that those residuals actually go to zero. That's the third inequality. **Inequality (A1):** I claim $V^{k+1}\le V^k-\rho\|r^{k+1}\|^2-\rho\|B(z^{k+1}-z^k)\|^2$. This comes out of $2\times[(A2)+(A3)]$ after rewriting. Adding (A2) and (A3) and multiplying by $2$ gives, after canceling $p^{k+1},p^\star$,

$$ 2(y^{k+1}-y^\star)^\top r^{k+1}-2\rho\big(B(z^{k+1}-z^k)\big)^\top r^{k+1}+2\rho\big(B(z^{k+1}-z^k)\big)^\top B(z^{k+1}-z^\star)\le 0. $$

The first term I massage with $y^{k+1}=y^k+\rho r^{k+1}$ and $r^{k+1}=\tfrac1\rho(y^{k+1}-y^k)$: it becomes $\tfrac1\rho\big(\|y^{k+1}-y^\star\|^2-\|y^k-y^\star\|^2\big)+\rho\|r^{k+1}\|^2$ — that's the $y$-part of $V^{k+1}-V^k$ falling out. The remaining terms, using $z^{k+1}-z^\star=(z^{k+1}-z^k)+(z^k-z^\star)$, reorganize into $\rho\|r^{k+1}-B(z^{k+1}-z^k)\|^2+\rho\big(\|B(z^{k+1}-z^\star)\|^2-\|B(z^k-z^\star)\|^2\big)$ — and that second piece is the $Bz$-part of $V^{k+1}-V^k$. Collecting, the inequality reads $V^k-V^{k+1}\ge\rho\|r^{k+1}-B(z^{k+1}-z^k)\|^2$. Expand the squared norm: $\rho\|r^{k+1}\|^2-2\rho(r^{k+1})^\top B(z^{k+1}-z^k)+\rho\|B(z^{k+1}-z^k)\|^2$. The middle cross term I need to be nonnegative to get (A1); and it is, because $z^{k+1}$ minimizes $g(z)+(y^{k+1})^\top Bz$ while $z^k$ minimizes $g(z)+(y^k)^\top Bz$, so adding the two minimizer inequalities gives $(y^{k+1}-y^k)^\top B(z^{k+1}-z^k)\le 0$, and $y^{k+1}-y^k=\rho r^{k+1}$ turns that into $\rho(r^{k+1})^\top B(z^{k+1}-z^k)\le 0$, i.e. $-2\rho(r^{k+1})^\top B(z^{k+1}-z^k)\ge 0$. Dropping that nonnegative cross term gives

$$ V^{k+1}\le V^k-\rho\|r^{k+1}\|^2-\rho\|B(z^{k+1}-z^k)\|^2.\tag{A1} $$

So $V^k$ is a genuine Lyapunov function: nonincreasing, hence $y^k$ and $Bz^k$ are bounded, and summing (A1) over all $k$ gives $\rho\sum_{k}\big(\|r^{k+1}\|^2+\|B(z^{k+1}-z^k)\|^2\big)\le V^0<\infty$. A convergent series has terms going to zero, so $r^k\to0$ (primal feasibility) and $B(z^{k+1}-z^k)\to0$; multiplying the latter by $\rho A^\top$ shows the dual residual $s^k=\rho A^\top B(z^{k+1}-z^k)\to0$. Feed $r^k\to0$ into (A3) and the bounded $B(z^{k+1}-z^\star)$, $r^{k+1}$, $B(z^{k+1}-z^k)\to0$ into (A2): both sides squeeze, so $p^k\to p^\star$ — objective convergence. So the alternation *is* enough: one Gauss–Seidel sweep per iteration converges, the residuals vanish, and the stopping rule I built is guaranteed to eventually trigger. The bet paid off.

That settles the abstract algorithm. Now let me make it concrete, because the whole point was decomposition and cheap per-block updates, and I want to see them actually fall out for the problems I care about. The cleanest case, and the one that shows the magic, is splitting a regularized loss. Take the lasso, $\min\tfrac12\|Cx-b\|_2^2+\lambda\|x\|_1$. It doesn't *look* like a two-block problem — there's one variable $x$. But I can manufacture the split by introducing a copy: set $f(x)=\tfrac12\|Cx-b\|^2$ and $g(z)=\lambda\|z\|_1$, and couple them with $x-z=0$. So $A=I$, $B=-I$, $c=0$. Now the two halves of the objective each get their own update and never have to be handled together. The scaled ADMM is

$$ x^{k+1}=\arg\min_x\Big\{\tfrac12\|Cx-b\|^2+\tfrac\rho2\|x-z^k+u^k\|^2\Big\},\quad z^{k+1}=\arg\min_z\Big\{\lambda\|z\|_1+\tfrac\rho2\|x^{k+1}-z+u^k\|^2\Big\},\quad u^{k+1}=u^k+x^{k+1}-z^{k+1}. $$

The $x$-update is a smooth quadratic — set the gradient to zero: $C^\top(Cx-b)+\rho(x-z^k+u^k)=0$, so

$$ x^{k+1}=(C^\top C+\rho I)^{-1}\big(C^\top b+\rho(z^k-u^k)\big). $$

That's just **ridge regression** — quadratically regularized least squares — and $C^\top C+\rho I$ is always invertible since $\rho>0$, regardless of the shape of $C$. Better still, the matrix doesn't change across iterations, so I factor it *once* (a Cholesky), cache the factors, and every subsequent $x$-update is a cheap back-solve. (If $C$ is short-and-fat, $q\ll d$, I'd rather apply the matrix-inversion lemma and factor the smaller $q\times q$ system $I+\tfrac1\rho CC^\top$ instead.) So ADMM has turned the lasso into a sequence of ridge regressions — exactly the kind of robust, well-conditioned solve the augmented penalty promised.

The $z$-update is where the $\ell_1$ lives, and I should solve it carefully because it's the prototype that recurs everywhere. With $A=I$ the penalty is $\tfrac\rho2\|z-v\|^2$ for $v=x^{k+1}+u^k$, so I need $\arg\min_z\{\lambda\|z\|_1+\tfrac\rho2\|z-v\|^2\}$. Because $\|z\|_1=\sum_i|z_i|$ is separable, this splits into $n$ identical scalar problems $\min_{z_i}\lambda|z_i|+\tfrac\rho2(z_i-v_i)^2$. Solve one. For $z_i>0$, $|z_i|=z_i$ is smooth: derivative $\lambda+\rho(z_i-v_i)=0\Rightarrow z_i=v_i-\lambda/\rho$, valid only when $v_i>\lambda/\rho$. For $z_i<0$, derivative $-\lambda+\rho(z_i-v_i)=0\Rightarrow z_i=v_i+\lambda/\rho$, valid when $v_i<-\lambda/\rho$. At $z_i=0$ there's a kink; $0$ is the minimizer exactly when $0$ is in the subdifferential $\lambda[-1,1]+\rho(0-v_i)$, i.e. $|v_i|\le\lambda/\rho$. Stitching the three regimes with $\kappa=\lambda/\rho$:

$$ z_i^{k+1}=\begin{cases}v_i-\kappa,&v_i>\kappa\\ 0,&|v_i|\le\kappa\\ v_i+\kappa,&v_i<-\kappa\end{cases}\;=\;\operatorname{sign}(v_i)\,\max(|v_i|-\kappa,0)\;=\;S_\kappa(v_i). $$

This is **soft-thresholding**, $S_\kappa(a)=(a-\kappa)_+-(-a-\kappa)_+=(1-\kappa/|a|)_+a$ — and in the language of the prox it's nothing but $\operatorname{prox}_{(\lambda/\rho)\|\cdot\|_1}$, the proximity operator of the $\ell_1$ norm. So the whole lasso reduces to: ridge-regress (cached factor), soft-threshold, update the running residual. Each step is dirt cheap, and the kink in $|\cdot|$ that made the problem nonsmooth is exactly the dead-zone that snaps small coordinates to zero. For this $A=I,B=-I,c=0$ instance the residuals specialize to $r^{k+1}=x^{k+1}-z^{k+1}$ and, since $A^\top B=-I$, $s^{k+1}=-\rho(z^{k+1}-z^k)$.

This generalizes immediately and that's the real prize. Any $\min \ell(x)+\lambda\|x\|_1$ for convex $\ell$ splits the same way, $x-z=0$, and ADMM becomes $x^{k+1}=\arg\min_x\{\ell(x)+\tfrac\rho2\|x-z^k+u^k\|^2\}$ (a *prox of $\ell$* — solvable by Newton or L-BFGS if $\ell$ is a smooth logistic loss, by a linear solve if it's quadratic) followed by the same soft-threshold $z$-update. Swap $\ell_1$ for any other simple regularizer and the only thing that changes is the $z$-update's prox: an indicator of a set $\mathcal C$ makes the $z$-update a Euclidean *projection* $\Pi_{\mathcal C}$ (the prox of an indicator is projection); a group/block norm $\sum\|z_g\|_2$ makes it *block* soft-thresholding $S_\kappa(a)=(1-\kappa/\|a\|_2)_+a$. The constrained problem $\min f(x)$ s.t. $x\in\mathcal C$ is just $f(x)+\iota_{\mathcal C}(z)$ with $x-z=0$: an $f$-prox alternating with a projection.

Now the distributed payoff, which is what I built the decomposition for in the first place. Take $\min\sum_{i=1}^N f_i(x)$ — a single global parameter, $N$ data-block losses summed over it. It doesn't split because the *one* $x$ is shared across all the $f_i$. So duplicate: give block $i$ a private copy $x_i$ and force agreement with a global $z$,

$$ \min\ \sum_{i=1}^N f_i(x_i)\quad\text{s.t.}\quad x_i-z=0,\ i=1,\dots,N. $$

This is the **consensus** trick: turn a shared-variable sum, which doesn't separate, into a sum over private copies, which does, paying for it with consensus constraints. Form the augmented Lagrangian and run scaled ADMM. The $x$-update separates completely — each worker solves its own

$$ x_i^{k+1}=\arg\min_{x_i}\Big\{f_i(x_i)+\tfrac\rho2\|x_i-z^k+u_i^k\|_2^2\Big\}, $$

a private regularized fit pulling $x_i$ toward the current consensus $z^k$ minus its price $u_i^k$. The $z$-update minimizes $\tfrac\rho2\sum_i\|x_i^{k+1}-z+u_i^k\|^2$ over the single $z$, which is just an average: $z^{k+1}=\tfrac1N\sum_i(x_i^{k+1}+u_i^k)=\overline{x}^{k+1}+\overline{u}^k$. And then $u_i^{k+1}=u_i^k+x_i^{k+1}-z^{k+1}$. So every iteration is: each processor fits its own block in parallel (no communication), a single *gather* forms the average $z$, a *broadcast* sends it back, and each processor nudges its price by its own disagreement $x_i-z$. The only thing crossing the network is the average — exactly the decentralized structure dual decomposition promised, but now with the augmented-Lagrangian robustness that dual decomposition lacked. (One can check the prices average to zero after the first iteration, so $z^{k+1}=\overline{x}^{k+1}$ and the prices act purely to drive the copies into agreement.) If a regularizer $g$ sits on the global variable, $\min\sum_i f_i(x_i)+g(z)$ s.t. $x_i=z$, the $z$-update becomes the average followed by a single prox of $g$ — e.g. a soft-threshold for $g=\lambda\|z\|_1$, a projection for an indicator.

The dual of consensus is the **sharing** problem $\min\sum_i f_i(x_i)+g(\sum_i x_i)$ — agents each pay a private cost and share a common term that sees only the *sum* of their choices. Copy the variables, $\min\sum_i f_i(x_i)+g(\sum_i z_i)$ s.t. $x_i-z_i=0$; the $x$-updates run in parallel, and the $z$-update — naively over $Nn$ variables — collapses to a problem in only $n$ variables by averaging, because the shared $g$ depends only on $\sum z_i$. The two templates are formally dual: running ADMM on a consensus problem is, up to conjugation, running it on the corresponding sharing problem. The exchange problem ($g$ the indicator of $\{\sum x_i=0\}$, agents trading commodities to net zero) is the headline special case.

One last thing pulls the whole construction together and explains *why* the convergence was so clean. Each block update is a prox/resolvent: the $x$-update is (a generalized) resolvent of $\partial f$, the $z$-update a resolvent of $\partial g$, and the iteration alternates them with a dual correction in between. That is precisely the shape of **Douglas–Rachford splitting** — Douglas and Rachford's 1956 alternating-resolvent scheme, extended by Lions and Mercier (1979) to find a zero of a sum of two maximal monotone operators by alternating their resolvents rather than touching the sum. Applied to the *dual* of my problem (where the two operators are built from $\partial f^*$ and $\partial g^*$), Douglas–Rachford splitting *is* this alternating minimization. And Douglas–Rachford is itself an instance of Rockafellar's proximal point algorithm on a maximal monotone operator, the same umbrella under which the method of multipliers lives. So the method I built by stubbornly insisting on "split the joint minimization into one Gauss–Seidel sweep" is, seen from the operator-splitting side, the canonical way to find a zero of $\partial f+\partial g$ — which is why the powerful proximal-point convergence theory applies to it directly, and why a self-contained Lyapunov proof like the one above exists at all. (The variant that slips an extra dual update *between* the $x$- and $z$-steps corresponds to Peaceman–Rachford splitting instead.)

Now let me land it in code, grounded in how a standard ADMM lasso solver is actually written: cache the factorization for the $x$-update, soft-threshold for the $z$-update, accumulate the scaled dual, and stop on the primal/dual residuals with absolute+relative tolerances.

```python
import numpy as np
from numpy.linalg import cholesky, norm


def shrinkage(a, kappa):
    # prox of kappa * ||.||_1 = soft-threshold: (a-kappa)_+ - (-a-kappa)_+
    return np.maximum(0.0, a - kappa) - np.maximum(0.0, -a - kappa)


def factor(C, rho):
    # Cache a Cholesky for the x-update (C'C + rho I), or its short-fat dual via the
    # matrix-inversion lemma when there are far fewer rows than columns.
    q, d = C.shape
    if q >= d:
        L = cholesky(C.T @ C + rho * np.eye(d))      # x-update matrix is d x d
    else:
        L = cholesky(np.eye(q) + (1.0 / rho) * (C @ C.T))  # solve the q x q system instead
    return L, L.T.conj()                              # lower L and its transpose U


def lasso_admm(C, b, lam, rho=1.0, alpha=1.0, n_iter=1000,
               abstol=1e-4, reltol=1e-2):
    # Solve  min 1/2 ||C x - b||^2 + lam ||x||_1  by ADMM with the split x - z = 0
    # (so A = I, B = -I, c = 0):  f(x) = 1/2||Cx-b||^2,  g(z) = lam||z||_1.
    q, d = C.shape
    Ctb = C.T @ b
    L, U = factor(C, rho)

    x = np.zeros(d)
    z = np.zeros(d)
    u = np.zeros(d)                                   # scaled dual u = y / rho

    for k in range(n_iter):
        # x-update: ridge regression (CtC + rho I) x = Ct b + rho (z - u), via the cached factors.
        rhs = Ctb + rho * (z - u)
        if q >= d:
            x = np.linalg.solve(U, np.linalg.solve(L, rhs))
        else:                                         # matrix-inversion-lemma form
            tmp = np.linalg.solve(U, np.linalg.solve(L, C @ rhs))
            x = rhs / rho - (C.T @ tmp) / (rho ** 2)

        # z-update: soft-threshold the relaxed x plus the scaled dual.
        z_old = z
        x_hat = alpha * x + (1.0 - alpha) * z_old     # (over-)relaxation, alpha in [1, 2)
        z = shrinkage(x_hat + u, lam / rho)

        # scaled dual update: u += A x + B z - c = x_hat - z (running sum of residuals).
        u = u + (x_hat - z)

        # primal residual r = x - z ; dual residual s = -rho (z - z_old)  (since A'B = -I).
        r_norm = norm(x - z)
        s_norm = norm(-rho * (z - z_old))
        eps_pri = np.sqrt(d) * abstol + reltol * max(norm(x), norm(z))
        eps_dual = np.sqrt(d) * abstol + reltol * norm(rho * u)
        if r_norm < eps_pri and s_norm < eps_dual:    # both optimality residuals small
            break

    return x, z


def consensus_admm(prox_fi, N, d, rho=1.0, n_iter=1000, abstol=1e-4, reltol=1e-2):
    # Solve  min sum_i f_i(x)  by global consensus: private copies x_i agree with global z.
    # prox_fi[i](v, rho) returns argmin_xi f_i(xi) + (rho/2)||xi - v||^2.
    X = np.zeros((N, d))                              # local copies
    z = np.zeros(d)                                   # global consensus variable
    U = np.zeros((N, d))                              # scaled duals, one per block

    for k in range(n_iter):
        z_old = z
        for i in range(N):                            # x-updates run independently / in parallel
            X[i] = prox_fi[i](z - U[i], rho)          # fit f_i, pulled toward consensus minus price
        z = (X + U).mean(axis=0)                      # z-update: gather the average (the only coupling)
        U = U + X - z                                 # each price nudged by its own disagreement

        # consensus residuals: primal = stacked (x_i - z), dual = -rho (z - z_old) per block.
        r_norm = np.sqrt(((X - z) ** 2).sum())
        s_norm = np.sqrt(N) * rho * norm(z - z_old)
        eps_pri = np.sqrt(N * d) * abstol + reltol * max(norm(X), np.sqrt(N) * norm(z))
        eps_dual = np.sqrt(N * d) * abstol + reltol * norm(rho * U)
        if r_norm < eps_pri and s_norm < eps_dual:
            break

    return z, X
```

The causal chain, end to end: I need a convex solver that is *both* robust and decomposable, but dual decomposition is parallel-yet-fragile (its $x$-update is unbounded for affine/nonstrictly-convex $f$) while the method of multipliers is robust-yet-serial (its quadratic penalty's cross terms couple the blocks); the damage is done specifically by the *joint* minimization of a coupled augmented Lagrangian, so I split the objective into $f(x)+g(z)$ with $Ax+Bz=c$ and replace the joint solve with a single Gauss–Seidel sweep — minimize over $x$, then $z$, then dual-update with step $\rho$ — which freezes the cross term in each subproblem and so keeps the augmented-Lagrangian robustness while restoring per-block decomposition; completing the square turns the dual into the scaled $u=y/\rho$ (the running sum of residuals); the optimality conditions hand me the primal residual $r=Ax+Bz-c$ and the dual residual $s=\rho A^\top B(z^{k+1}-z^k)$, one optimality condition holds automatically at every iterate, and a Lyapunov argument proves both residuals vanish and the objective converges, which justifies stopping when both are below absolute+relative tolerances; instantiating the split on a regularized loss makes the $x$-update a cached ridge solve and the $z$-update the soft-threshold prox of $\ell_1$ (an indicator's prox is a projection, a block norm's is block soft-thresholding), and the same duplication trick turns a shared-variable sum into the parallel consensus algorithm (private fits + an average) and its dual sharing form — all of which is, viewed abstractly, Douglas–Rachford splitting on the dual, an instance of the proximal point algorithm, which is why the convergence theory comes for free.
