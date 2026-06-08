Let me start from the thing that actually hurts. I have a convex problem that splits — $\min f(x)+g(z)$ subject to $Ax+Bz=c$ — and I want two things at once that, with the tools I have, seem to be at war with each other. I want to update $x$ and $z$ *separately*, because in the problems I care about $f$ is really a sum $\sum_i f_i$ over many terms or many machines, and the whole point is to let each term do its own little optimization and only exchange a price. And I want the method to *converge no matter what*: $f$ and $g$ might be nonsmooth, might be $+\infty$ somewhere (an indicator of a constraint set folded in), and $A,B$ might be rank-deficient. No strict convexity to lean on. So: decomposable, and robust.

Take the two classical hammers and see exactly where each one breaks, because the break is the clue.

Dual ascent. Form the Lagrangian $L(x,y)=f(x)+y^\top(Ax-b)$, the dual function $g_{\mathrm d}(y)=\inf_x L(x,y)=-f^\*(-A^\top y)-b^\top y$, and climb the dual: $x^{k+1}=\arg\min_x L(x,y^k)$, then $y^{k+1}=y^k+\alpha^k(Ax^{k+1}-b)$. The residual $Ax^{k+1}-b$ is exactly $\nabla g_{\mathrm d}(y^k)$ when the dual is differentiable, so this really is gradient ascent on the dual, and the beautiful part is decomposition: if $f(x)=\sum_i f_i(x_i)$ and $A=[A_1\cdots A_N]$, then $L=\sum_i\big(f_i(x_i)+y^\top A_ix_i-\tfrac1N y^\top b\big)$ separates, the $x$-step splits into $N$ independent little problems, and $y$ is just a broadcast price. Gather the residual, broadcast the price. Lovely. But now watch it die: suppose $f$ is affine in even one coordinate. Then $L(\cdot,y)$ is unbounded below in that coordinate for almost every $y$, and $\arg\min_x L$ doesn't exist — the $x$-step is garbage. And if $g_{\mathrm d}$ isn't differentiable, I'm doing a supergradient step on a concave dual function, equivalently a subgradient step on $-g_{\mathrm d}$, which is nonmonotone and needs a fussy diminishing $\alpha^k$. So dual ascent gives me decomposition and almost no robustness.

Method of multipliers. The fix for the fragility is known: regularize the dual by adding a quadratic penalty to the Lagrangian,
$$L_\rho(x,y)=f(x)+y^\top(Ax-b)+\tfrac{\rho}{2}\|Ax-b\|_2^2,\qquad\rho>0.$$
Why does that help, concretely? Because $L_\rho$ is just the *ordinary* Lagrangian of the equivalent problem $\min_x f(x)+\tfrac\rho2\|Ax-b\|^2$ s.t. $Ax=b$ — adding $\tfrac\rho2\|Ax-b\|^2$ changes nothing on the feasible set but bends the objective so the inner $\inf_x$ is now attained and the augmented dual $g_\rho$ is differentiable under mild conditions. No strict convexity needed, no finiteness needed. Iterate $x^{k+1}=\arg\min_x L_\rho(x,y^k)$, $y^{k+1}=y^k+\rho(Ax^{k+1}-b)$. And there's a reason the dual step size is *exactly* $\rho$ and not some free $\alpha$. Look at the stationarity of the $x$-step: $0=\nabla_x L_\rho(x^{k+1},y^k)=\nabla f(x^{k+1})+A^\top\!\big(y^k+\rho(Ax^{k+1}-b)\big)$. If I *define* $y^{k+1}=y^k+\rho(Ax^{k+1}-b)$, that right-hand side collapses to $\nabla f(x^{k+1})+A^\top y^{k+1}$, which is zero — so $(x^{k+1},y^{k+1})$ is automatically *dual feasible* every single iteration. The price update with step $\rho$ is the thing that keeps me on the dual-feasibility manifold; then as the primal residual $\to0$ I'm at the optimum. That's robustness, and it's the step-size choice that buys it.

But the penalty has a cost, and it's the exact opposite cost from before. With $f=\sum_i f_i$, the penalty is $\tfrac\rho2\big\|\sum_i A_ix_i-b\big\|^2$, and when I expand that square the cross terms $\rho\,(A_ix_i)^\top(A_jx_j)$ appear. Those couple the blocks. $L_\rho$ does *not* separate. The $x$-minimization can no longer be done block-by-block in parallel. So the method of multipliers is robust and *not* decomposable. Each hammer gives me precisely the property the other one lacks, and the robust hammer's robustness *is* what destroys the decomposition.

So here's the corner I'm in. The penalty is non-negotiable — it's what makes the inner problem well posed for affine/nonsmooth $f$. But the penalty couples the blocks. I can't separate them inside a single joint minimization. Stare at the joint step the method of multipliers wants:
$$(x^{k+1},z^{k+1})=\arg\min_{x,z}L_\rho(x,z,y^k),\qquad L_\rho(x,z,y)=f(x)+g(z)+y^\top(Ax+Bz-c)+\tfrac\rho2\|Ax+Bz-c\|_2^2.$$
The coupling lives entirely in the cross term $\rho(Ax)^\top(Bz)$ inside the square. But — wait. The cross term only couples $x$ and $z$ *while they're both free*. If I *freeze* $z$ and minimize over $x$ alone, then $Bz$ is a constant vector and the square $\tfrac\rho2\|Ax+(Bz^k-c)\|^2$ is just a quadratic in $x$; the cross term is a constant offset, no longer a coupling. Then freeze $x^{k+1}$ and minimize over $z$ alone — same thing, $Ax^{k+1}$ is now a constant. So I never minimize jointly. I do one sweep:
$$x^{k+1}=\arg\min_x L_\rho(x,z^k,y^k),\qquad z^{k+1}=\arg\min_z L_\rho(x^{k+1},z,y^k),$$
and then the same dual step as before, $y^{k+1}=y^k+\rho(Ax^{k+1}+Bz^{k+1}-c)$. One Gauss–Seidel pass over $(x,z)$ instead of the joint minimization. The alternation is the move that makes the penalty usable: by updating one block with the other held fixed, the penalty's cross term degrades into a constant, each subproblem becomes a single-block penalized minimization (which *does* split when $f$ or $g$ is itself a sum), and yet I'm still minimizing the augmented Lagrangian, so the robustness should survive. The state I carry forward is $(z^k,y^k)$; $x^k$ is just an intermediate quantity I recompute. Note the asymmetry: the dual update happens *after* the $z$-step, between the $z$-step and the next $x$-step. That ordering will matter.

Now the real question, the one I actually care about: does this thing converge, and under what assumptions, and at what rate, and when do I stop? Alternating minimization of a nonconvex-looking coupled function with a dual step in the middle has no business converging just because I want it to. I need a certificate.

Let me first nail the optimality conditions and the right notion of "residual," because the proof and the stopping rule both hang off them. A solution $(x^\*,z^\*,y^\*)$ of the problem is characterized by *primal feasibility* $Ax^\*+Bz^\*-c=0$ and *dual feasibility* $0\in\partial f(x^\*)+A^\top y^\*$ and $0\in\partial g(z^\*)+B^\top y^\*$ (subdifferentials, since $f,g$ are nonsmooth; for smooth $f,g$ read $\nabla$ and $=$).

Watch what the $z$-step already gives me for free. Since $z^{k+1}$ minimizes $L_\rho(x^{k+1},z,y^k)$,
$$0\in\partial g(z^{k+1})+B^\top y^k+\rho B^\top(Ax^{k+1}+Bz^{k+1}-c)=\partial g(z^{k+1})+B^\top\big(y^k+\rho r^{k+1}\big)=\partial g(z^{k+1})+B^\top y^{k+1},$$
where I wrote $r^{k+1}:=Ax^{k+1}+Bz^{k+1}-c$ for the primal residual and used $y^{k+1}=y^k+\rho r^{k+1}$. So $(z^{k+1},y^{k+1})$ *always* satisfies the $z$-block dual-feasibility condition, every iteration, exactly. That's the same phenomenon as the method of multipliers being dual feasible — and it's *because* the dual update sits right after the $z$-step with step size $\rho$. If I'd put the dual update somewhere else or used a different step, this wouldn't hold. So two of the three optimality conditions (primal feasibility, and $z$-dual-feasibility) — one of them is auto-satisfied. The whole job is to drive the other two to zero.

Now the $x$-step. $x^{k+1}$ minimizes $L_\rho(x,z^k,y^k)$, so
$$0\in\partial f(x^{k+1})+A^\top y^k+\rho A^\top(Ax^{k+1}+Bz^k-c).$$
This is *not* yet in terms of $y^{k+1}$, because it has $Bz^k$ where the residual wants $Bz^{k+1}$. So let me force $y^{k+1}$ to appear by adding and subtracting $\rho A^\top Bz^{k+1}$:
$$0\in\partial f(x^{k+1})+A^\top\!\Big(y^k+\rho(Ax^{k+1}+Bz^{k+1}-c)+\rho B(z^k-z^{k+1})\Big)=\partial f(x^{k+1})+A^\top y^{k+1}+\rho A^\top B(z^k-z^{k+1}).$$
Rearranged, $\rho A^\top B(z^{k+1}-z^k)\in\partial f(x^{k+1})+A^\top y^{k+1}$. The left side is exactly how far the $x$-block dual-feasibility condition $0\in\partial f+A^\top y$ is from holding. So the natural **dual residual** is
$$s^{k+1}:=\rho A^\top B(z^{k+1}-z^k),$$
and the **primal residual** is $r^{k+1}=Ax^{k+1}+Bz^{k+1}-c$. The two residuals are exactly the two optimality conditions not automatically satisfied. If both vanish, I'm optimal. Good — now I know what has to go to zero, and the dual residual being proportional to $z^{k+1}-z^k$ tells me that, morally, "$z$ stops moving" is half of convergence.

Before I hunt for a Lyapunov function, let me get the scaled form, because the algebra is cleaner and I'll want it for both the code and the bookkeeping. Combine the linear and quadratic terms in the residual $r=Ax+Bz-c$:
$$y^\top r+\tfrac\rho2\|r\|^2=\tfrac\rho2\big\|r+\tfrac1\rho y\big\|^2-\tfrac1{2\rho}\|y\|^2=\tfrac\rho2\|r+u\|^2-\tfrac\rho2\|u\|^2,\qquad u:=\tfrac1\rho y.$$
The trailing $-\tfrac\rho2\|u\|^2$ has no $x,z$, so it drops out of both minimizations, and the updates become pure proximal steps:
$$x^{k+1}=\arg\min_x\Big(f(x)+\tfrac\rho2\|Ax+Bz^k-c+u^k\|^2\Big),\quad z^{k+1}=\arg\min_z\Big(g(z)+\tfrac\rho2\|Ax^{k+1}+Bz-c+u^k\|^2\Big),$$
$$u^{k+1}=u^k+Ax^{k+1}+Bz^{k+1}-c.$$
And $u^k=u^0+\sum_{j=1}^k r^j$, the running sum of residuals — the scaled dual variable is just the accumulated constraint violation. Nice intuition: the dual variable is the integral of the error, like the integral term in a controller.

Now the certificate. I want a nonnegative quantity that strictly decreases every iteration, and whose decrease forces the residuals to zero. Let me guess it should measure how far the *state* $(z^k,y^k)$ is from a saddle point $(x^\*,z^\*,y^\*)$ of the unaugmented Lagrangian $L_0$ — assume one exists; that's my one robustness-friendly assumption (saddle point of $L_0$ $\Leftrightarrow$ strong duality holds and is attained, with no strict convexity required). The dual error is $\|y^k-y^\*\|$; the primal-block error that the dual residual saw is $\|B(z^k-z^\*)\|$. The dual update has a $1/\rho$ in disguise (since $y$ moves in steps of $\rho r$) and the primal penalty carries a $\rho$, so to make them commensurable — and, I suspect, to make $\rho$ cancel so that *any* $\rho>0$ works — let me weight them oppositely:
$$V^k:=\tfrac1\rho\|y^k-y^\*\|_2^2+\rho\|B(z^k-z^\*)\|_2^2.$$
It's nonnegative. It depends on the unknown $z^\*,y^\*$, so I can't compute it while running — that's fine, it's a proof device. The claim I want is
$$V^{k+1}\le V^k-\rho\|r^{k+1}\|_2^2-\rho\|B(z^{k+1}-z^k)\|_2^2.\tag{A.1}$$
If I can prove (A.1), I'm almost done: $V^k$ decreases by an amount controlled by the primal residual *and* by how much $z$ moved, so summing over all $k$,
$$\rho\sum_{k=0}^\infty\big(\|r^{k+1}\|^2+\|B(z^{k+1}-z^k)\|^2\big)\le V^0,$$
a finite bound, which forces $r^k\to0$ (primal feasibility) and $B(z^{k+1}-z^k)\to0$. Multiply the second by $\rho A^\top$ and the dual residual $s^{k}=\rho A^\top B(z^{k}-z^{k-1})\to0$ too. Both residuals to zero. And $V^k\le V^0$ bounds $y^k$ and $Bz^k$. So (A.1) is the whole game. Let me prove it — and I should not assume it; I have to build it from the two min-steps and the saddle inequality, with the signs checked.

I'll get there through two intermediate inequalities about the objective $p^{k+1}:=f(x^{k+1})+g(z^{k+1})$ relative to the optimal value $p^\*$.

The easy one first. Since $(x^\*,z^\*,y^\*)$ is a saddle point of $L_0$, the multiplier $y^\*$ minimizes nothing but the saddle gives $L_0(x^\*,z^\*,y^\*)\le L_0(x^{k+1},z^{k+1},y^\*)$. The left side: $Ax^\*+Bz^\*-c=0$, so $L_0(x^\*,z^\*,y^\*)=f(x^\*)+g(z^\*)=p^\*$. The right side is $p^{k+1}+ {y^\*}^\top r^{k+1}$. Hence
$$p^\*-p^{k+1}\le {y^\*}^\top r^{k+1}.\tag{A.3}$$
Once the Lyapunov step gives $r^{k+1}\to0$, this right side goes to zero, so $\liminf p^{k+1}\ge p^\*$ — the objective can't beat optimal in the limit. I need the other direction too.

The other one comes from the two min-steps. From the $x$-step optimality I derived $0\in\partial f(x^{k+1})+A^\top y^{k+1}+\rho A^\top B(z^k-z^{k+1})$, i.e. substituting $y^k=y^{k+1}-\rho r^{k+1}$ into $0\in\partial f(x^{k+1})+A^\top y^k+\rho A^\top(Ax^{k+1}+Bz^k-c)$ gives
$$0\in\partial f(x^{k+1})+A^\top\!\big(y^{k+1}-\rho B(z^{k+1}-z^k)\big),$$
which says $x^{k+1}$ minimizes the convex function $f(x)+\big(y^{k+1}-\rho B(z^{k+1}-z^k)\big)^\top Ax$. Minimizer means its value at $x^{k+1}$ is $\le$ its value at $x^\*$:
$$f(x^{k+1})+\big(y^{k+1}-\rho B(z^{k+1}-z^k)\big)^\top Ax^{k+1}\le f(x^\*)+\big(y^{k+1}-\rho B(z^{k+1}-z^k)\big)^\top Ax^\*.$$
Similarly, the $z$-step gave $0\in\partial g(z^{k+1})+B^\top y^{k+1}$, so $z^{k+1}$ minimizes $g(z)+{y^{k+1}}^\top Bz$, hence
$$g(z^{k+1})+{y^{k+1}}^\top Bz^{k+1}\le g(z^\*)+{y^{k+1}}^\top Bz^\*.$$
Add the two. On the right, $f(x^\*)+g(z^\*)=p^\*$ and the multiplier terms combine using $Ax^\*+Bz^\*=c$. Rearranging,
$$p^{k+1}-p^\*\le -{y^{k+1}}^\top r^{k+1}-\rho\big(B(z^{k+1}-z^k)\big)^\top\!\big(-r^{k+1}+B(z^{k+1}-z^\*)\big).\tag{A.2}$$
After the Lyapunov step, $B(z^{k+1}-z^\*)$ is bounded, $r^{k+1}\to0$, and $B(z^{k+1}-z^k)\to0$, so the right side goes to zero; together with (A.3), that will give $p^k\to p^\*$. And (A.2) already hands me the stopping bound. Look at the second slot of its last term, $-r^{k+1}+B(z^{k+1}-z^\*)$: expand $r^{k+1}=Ax^{k+1}+Bz^{k+1}-c$ and use $c-Bz^\*=Ax^\*$ to get $-r^{k+1}+B(z^{k+1}-z^\*)=-Ax^{k+1}+c-Bz^\*=-A(x^{k+1}-x^\*)$. So that term becomes $-\rho\big(B(z^{k+1}-z^k)\big)^\top\big(-A(x^{k+1}-x^\*)\big)=(x^{k+1}-x^\*)^\top\rho A^\top B(z^{k+1}-z^k)=(x^{k+1}-x^\*)^\top s^{k+1}$, and (A.2) reads $p^{k+1}-p^\*\le -{y^{k+1}}^\top r^{k+1}+(x^{k+1}-x^\*)^\top s^{k+1}$ — the suboptimality bound I'll use for the stopping rule.

Now assemble (A.1) from (A.2)+(A.3). Add them and multiply by $2$; the $p^{k+1}$ and $p^\*$ cancel, leaving
$$2(y^{k+1}-y^\*)^\top r^{k+1}-2\rho\big(B(z^{k+1}-z^k)\big)^\top r^{k+1}+2\rho\big(B(z^{k+1}-z^k)\big)^\top\!\big(B(z^{k+1}-z^\*)\big)\le0.\tag{A.4}$$
Now grind this into the telescoping form. Rewrite the first term using $y^{k+1}=y^k+\rho r^{k+1}$:
$$2(y^{k+1}-y^\*)^\top r^{k+1}=2(y^k-y^\*)^\top r^{k+1}+2\rho\|r^{k+1}\|^2.$$
Hmm, that gave $2\rho\|r^{k+1}\|^2$; let me redo carefully — $y^{k+1}-y^\*=(y^k-y^\*)+\rho r^{k+1}$, so $2(y^{k+1}-y^\*)^\top r^{k+1}=2(y^k-y^\*)^\top r^{k+1}+2\rho\|r^{k+1}\|^2$. Now in the *first* of those, substitute $r^{k+1}=\tfrac1\rho(y^{k+1}-y^k)$:
$$2(y^k-y^\*)^\top r^{k+1}=\tfrac2\rho(y^k-y^\*)^\top(y^{k+1}-y^k).$$
And $y^{k+1}-y^k=(y^{k+1}-y^\*)-(y^k-y^\*)$, so with the parallelogram identity $2a^\top(b-a)=\|b\|^2-\|a\|^2-\|b-a\|^2$ on $a=y^k-y^\*$, $b=y^{k+1}-y^\*$:
$$\tfrac2\rho(y^k-y^\*)^\top(y^{k+1}-y^k)=\tfrac1\rho\Big(\|y^{k+1}-y^\*\|^2-\|y^k-y^\*\|^2-\|y^{k+1}-y^k\|^2\Big).$$
But $\|y^{k+1}-y^k\|^2=\rho^2\|r^{k+1}\|^2$, so $-\tfrac1\rho\|y^{k+1}-y^k\|^2=-\rho\|r^{k+1}\|^2$, which cancels one of the $+\rho\|r^{k+1}\|^2$ I picked up. Collecting the $y$-terms:
$$2(y^{k+1}-y^\*)^\top r^{k+1}=\tfrac1\rho\Big(\|y^{k+1}-y^\*\|^2-\|y^k-y^\*\|^2\Big)+\rho\|r^{k+1}\|^2.\tag{A.5}$$
There's the dual half of $V$, already telescoping. Now the remaining terms of (A.4), keeping the $\rho\|r^{k+1}\|^2$ from (A.5):
$$\rho\|r^{k+1}\|^2-2\rho\big(B(z^{k+1}-z^k)\big)^\top r^{k+1}+2\rho\big(B(z^{k+1}-z^k)\big)^\top\!\big(B(z^{k+1}-z^\*)\big).$$
In the last term write $z^{k+1}-z^\*=(z^{k+1}-z^k)+(z^k-z^\*)$:
$$=\rho\|r^{k+1}\|^2-2\rho\big(B\Delta z\big)^\top r^{k+1}+2\rho\|B\Delta z\|^2+2\rho\big(B\Delta z\big)^\top\!\big(B(z^k-z^\*)\big),\quad \Delta z:=z^{k+1}-z^k.$$
The first three terms are $\rho\|r^{k+1}-B\Delta z\|^2+\rho\|B\Delta z\|^2$ (since $\|r-B\Delta z\|^2=\|r\|^2-2(B\Delta z)^\top r+\|B\Delta z\|^2$, and I have an extra $+\rho\|B\Delta z\|^2$). For the last term, write $\Delta z=(z^{k+1}-z^\*)-(z^k-z^\*)$ so that $2\rho\|B\Delta z\|^2+2\rho(B\Delta z)^\top B(z^k-z^\*)=\rho\big(\|B(z^{k+1}-z^\*)\|^2-\|B(z^k-z^\*)\|^2\big)+\rho\|B\Delta z\|^2$... let me just do it by the same parallelogram move: $2(B\Delta z)^\top B(z^k-z^\*)$ with $B\Delta z=B(z^{k+1}-z^\*)-B(z^k-z^\*)$ gives $\|B(z^{k+1}-z^\*)\|^2-\|B(z^k-z^\*)\|^2-\|B\Delta z\|^2$, and adding the leftover $+\rho\|B\Delta z\|^2$ from a moment ago cancels that $-\|B\Delta z\|^2$. The net of the $z$-terms is
$$\rho\|r^{k+1}-B\Delta z\|^2+\rho\Big(\|B(z^{k+1}-z^\*)\|^2-\|B(z^k-z^\*)\|^2\Big).$$
Put (A.5) and this together: (A.4) says the sum is $\le0$, i.e.
$$\tfrac1\rho\Big(\|y^{k+1}-y^\*\|^2-\|y^k-y^\*\|^2\Big)+\rho\Big(\|B(z^{k+1}-z^\*)\|^2-\|B(z^k-z^\*)\|^2\Big)+\rho\|r^{k+1}-B\Delta z\|^2\le0,$$
which is exactly
$$V^{k+1}-V^k+\rho\|r^{k+1}-B(z^{k+1}-z^k)\|^2\le0,\qquad\text{i.e.}\qquad V^k-V^{k+1}\ge\rho\|r^{k+1}-B(z^{k+1}-z^k)\|^2.\tag{A.6}$$
So $V$ already decreases by at least $\rho\|r^{k+1}-B\Delta z\|^2$. That's good but it's not quite (A.1), which wants the decrease to be $\rho\|r^{k+1}\|^2+\rho\|B\Delta z\|^2$. Expanding the square in (A.6),
$$\|r^{k+1}-B\Delta z\|^2=\|r^{k+1}\|^2-2\,{r^{k+1}}^\top B\Delta z+\|B\Delta z\|^2,$$
so I get (A.1) provided the cross term is favorable, i.e. provided $-2{r^{k+1}}^\top B(z^{k+1}-z^k)\ge0$, i.e. ${r^{k+1}}^\top B(z^{k+1}-z^k)\le0$. Is it? Here's where the $z$-step pays off again. $z^{k+1}$ minimizes $g(z)+{y^{k+1}}^\top Bz$ and $z^k$ minimizes $g(z)+{y^k}^\top Bz$. Optimality at each, evaluated at the other point:
$$g(z^{k+1})+{y^{k+1}}^\top Bz^{k+1}\le g(z^k)+{y^{k+1}}^\top Bz^k,\qquad g(z^k)+{y^k}^\top Bz^k\le g(z^{k+1})+{y^k}^\top Bz^{k+1}.$$
Add them; the $g$'s cancel:
$$(y^{k+1}-y^k)^\top B(z^{k+1}-z^k)\le0.$$
This is just monotonicity of the subdifferential of $g$ pulled back through $B$. Substitute $y^{k+1}-y^k=\rho r^{k+1}$, and since $\rho>0$, ${r^{k+1}}^\top B(z^{k+1}-z^k)\le0$. So the cross term is $\ge0$, and (A.6) upgrades to
$$V^{k+1}\le V^k-\rho\|r^{k+1}\|^2-\rho\|B(z^{k+1}-z^k)\|^2.$$
That's (A.1). Done. Let me say out loud what just happened, because it answers the question I cared about most: *why does any $\rho>0$ work?* Look at $V=\tfrac1\rho\|y-y^\*\|^2+\rho\|B(z-z^\*)\|^2$. The dual-error term is divided by $\rho$ exactly because $y$ moves in chunks of size $\rho$; the primal-$z$ term is multiplied by $\rho$. When I converted $2(y^k-y^\*)^\top r^{k+1}$ via $r^{k+1}=\tfrac1\rho(y^{k+1}-y^k)$, the $\tfrac1\rho$ and the $\rho$ in $\|y^{k+1}-y^k\|^2=\rho^2\|r\|^2$ multiplied to give a clean $\rho\|r\|^2$ with no residual dependence on the *magnitude* of $\rho$ — the weights were chosen precisely so $\rho$ never appears in a way that could be too big or too small. The decrease is $\rho(\|r\|^2+\|B\Delta z\|^2)\ge0$ for *every* positive $\rho$. There's no stepsize to tune for convergence; $\rho$ trades off how fast the primal vs. dual residual shrinks, but never breaks convergence. That's the robustness I wanted, and it's structural, not a fluke of some condition number.

Let me cash out the stopping rule from the bound I just read off, because in practice I can't watch $V$ (it needs $z^\*,y^\*$):
$$f(x^k)+g(z^k)-p^\*\le-(y^k)^\top r^k+(x^k-x^\*)^\top s^k,\qquad s^k=\rho A^\top B(z^k-z^{k-1}).$$
The right side is small exactly when the primal residual $r^k$ and dual residual $s^k$ are small. I don't know $x^\*$, but if I'm willing to guess $\|x^k-x^\*\|\le d$, then $f(x^k)+g(z^k)-p^\*\le\|y^k\|\,\|r^k\|+d\,\|s^k\|$. So a sensible termination is: stop when $\|r^k\|_2\le\epsilon^{\mathrm{pri}}$ and $\|s^k\|_2\le\epsilon^{\mathrm{dual}}$, with tolerances built from an absolute and a relative part,
$$\epsilon^{\mathrm{pri}}=\sqrt p\,\epsilon^{\mathrm{abs}}+\epsilon^{\mathrm{rel}}\max\{\|Ax^k\|_2,\|Bz^k\|_2,\|c\|_2\},\qquad \epsilon^{\mathrm{dual}}=\sqrt n\,\epsilon^{\mathrm{abs}}+\epsilon^{\mathrm{rel}}\|A^\top y^k\|_2,$$
the $\sqrt p,\sqrt n$ accounting for the dimensions of the $\ell_2$ norms in $\mathbb R^p$ and $\mathbb R^n$ — so the absolute floor scales with problem size, and the relative part compares each residual to the magnitude of the terms it's made of.

Now I want to understand the convergence at a *deeper* level than "I found a Lyapunov function that happened to telescope," because that proof, slick as it is, doesn't tell me *why* the structure is right or whether it's an accident. Let me put on the operator-splitting glasses. Optimality of $\min f(x)+g(Mx)$ is, dually, finding a zero of the sum of two maximal monotone operators. Write the problem as $\min f(x)+g(w)$ s.t. $Mx=w$, attach multiplier $p$, and the dual is $\max -\big(f^\*(-M^\top p)+g^\*(p)\big)$. Set $\mathcal A:=\partial\big(f^\*\circ(-M^\top)\big)$ and $\mathcal B:=\partial g^\*$; both are maximal monotone, and dual optimality is $0\in\mathcal A p+\mathcal B p$, a zero of $\mathcal A+\mathcal B$.

To find a zero of a maximal monotone $T$ I could run the proximal point algorithm $z^{k+1}=J_{cT}(z^k)=(I+cT)^{-1}(z^k)$. Why does *that* converge for any $c>0$? Because the resolvent $J_{cT}$ is firmly nonexpansive — $\|J_{cT}a-J_{cT}b\|^2\le(J_{cT}a-J_{cT}b)^\top(a-b)$ — and its fixed points are exactly the zeros of $T$. Firm nonexpansiveness means each step is a contraction toward the fixed-point set, with no stepsize restriction: any positive $c$ works, even with summable errors, even with relaxation $z^{k+1}=(1-\rho_k)z^k+\rho_kJ_{cT}(z^k)$, $\rho_k\in(0,2)$. So *if* I can realize my method as a proximal point iteration on some monotone operator, the "any $\rho$" robustness is immediate and not a coincidence — it's inherited from firm nonexpansiveness.

But $J_{c(\mathcal A+\mathcal B)}$ is as hard as the original problem. The splitting idea: use only $J_{\lambda\mathcal A},J_{\lambda\mathcal B}$ via the Douglas–Rachford recursion
$$z^{k+1}=J_{\lambda\mathcal A}\big((2J_{\lambda\mathcal B}-I)(z^k)\big)+(I-J_{\lambda\mathcal B})(z^k)=G_{\lambda,\mathcal A,\mathcal B}(z^k).$$
This came originally from an alternating-direction scheme for the discretized heat equation in the '50s, repurposed by Lions–Mercier for monotone operators. They showed $G_{\lambda,\mathcal A,\mathcal B}$ is firmly nonexpansive directly. But I can see something sharper: define the *splitting operator* $S_{\lambda,\mathcal A,\mathcal B}:=(G_{\lambda,\mathcal A,\mathcal B})^{-1}-I$ as a set-valued operator. If $x=G(v)$, then $v-x\in Sx$ by definition, so $v\in(I+S)x$ and therefore $x=(I+S)^{-1}v=G(v)$; conversely the same implication reverses, so $(I+S)^{-1}=G$. The splitting-operator construction makes $S$ maximal monotone whenever $\mathcal A,\mathcal B$ are. So the Douglas–Rachford iteration $z^{k+1}=G(z^k)=(I+S)^{-1}(z^k)$ is *literally the proximal point algorithm applied to $S$ with proximal-point stepsize $c=1$*. That's the unification: Douglas–Rachford is proximal point in disguise, on a cleverly constructed operator. And the zeros of $S$ map (through $J_{\lambda\mathcal B}$) onto zeros of $\mathcal A+\mathcal B$. So all the proximal-point robustness transfers to Douglas–Rachford, while $\lambda>0$ remains the resolvent parameter inside the two easy resolvents.

And what it specializes to, on the dual operators $\mathcal A=\partial(f^\*\circ(-M^\top))$, $\mathcal B=\partial g^\*$, is — when I unwind the resolvent evaluations through the conjugates — exactly the alternating minimization with the dual update I wrote down:
$$x^{k+1}=\arg\min_x\big(f(x)+(p^k)^\top Mx+\tfrac\lambda2\|Mx-w^k\|^2\big),\ \ w^{k+1}=\arg\min_w\big(g(w)-(p^k)^\top w+\tfrac\lambda2\|Mx^{k+1}-w\|^2\big),\ \ p^{k+1}=p^k+\lambda(Mx^{k+1}-w^{k+1}).$$
That is the same alternating augmented-Lagrangian pattern, and the general constraint $Ax+Bz=c$ follows by absorbing the linear maps and the constant into the two dual monotone pieces. With $\lambda=\rho$, my alternating scheme *is* Douglas–Rachford on the dual *is* proximal point on $S$ with stepsize $1$. Two consequences I care about land at once.

First, the "$\lambda=\rho$ fixed, any positive value, no tuning for convergence" property is no accident. The proximal-point stepsize on $S$ is the fixed value $1$; the penalty parameter is the Douglas–Rachford resolvent parameter $\lambda$. Resolvents of maximal monotone operators are firmly nonexpansive for every positive resolvent parameter, so convergence for all $\rho>0$ is firm nonexpansiveness showing up in the augmented-Lagrangian variables.

Second — and this finally explains the *order* of my updates and why the proximal-point step on $S$ is fixed at $1$ — try to be greedy and evaluate $(I+cS)^{-1}(z)$ with $c\ne1$. The resolvent conditions require finding $(u,b)\in\mathcal B$ and $(v,a)\in\mathcal A$ with $(1-c)v+cu+\lambda b=z$ and $a=\tfrac1\lambda(u-v)-b$ simultaneously; the two operators are tangled together again. But fix $c=1$: now I only need $(u,b)\in\mathcal B$ with $u+\lambda b=z$, which determines $u=J_{\lambda\mathcal B}(z)$ and $b=(z-u)/\lambda$ independently, and then $(v,a)\in\mathcal A$ with $v+\lambda a=u-\lambda b$, which determines $v=J_{\lambda\mathcal A}(u-\lambda b)$. The two resolvents evaluate sequentially: first $\mathcal B$ (my $z$-block, via $g^\*$), then $\mathcal A$ (my $x$-block). Keeping the proximal stepsize at exactly $1$ is precisely what decouples the joint solve into one block then the other. So the alternation — minimize one block, then the other, with the dual update wedged between — is not a heuristic; it is the factorization of the splitting operator's unit-step resolvent. The Gauss–Seidel order and the "$z$-dual-feasibility holds for free" fact are two faces of $c=1$.

There's one more thing I'd like before I'm satisfied: a *rate*, not just "converges." The VI sign is cleaner if I flip the multiplier only for this bookkeeping: write $\lambda=-y$, stack $w=(x,z,\lambda)$, let $u=(x,z)$ and $\theta(u)=f(x)+g(z)$, and set
$$F(w)=\begin{pmatrix}-A^\top\lambda\\-B^\top\lambda\\Ax+Bz-c\end{pmatrix}.$$
Now the KKT conditions are exactly the VI: find $w^\*\in\Omega$ with $\theta(u)-\theta(u^\*)+(w-w^\*)^\top F(w^\*)\ge0$ for all $w$. The linear part of $F$ is skew-symmetric, so $(w-w')^\top(F(w)-F(w'))=0$; this is the monotonicity that makes a rate possible without strong convexity. The seminorm that matches the Lyapunov proof is
$$H=\begin{pmatrix}0&0&0\\0&\rho B^\top B&0\\0&0&\rho^{-1}I\end{pmatrix},\qquad \|w^k-w^{k+1}\|_H^2=\rho\|B(z^{k+1}-z^k)\|^2+\rho\|r^{k+1}\|^2,$$
because $\lambda^{k+1}-\lambda^k=-\rho r^{k+1}$. This is the right error measure: if it is zero, then $r^{k+1}=0$ and $B(z^{k+1}-z^k)=0$; the $z$-dual condition already holds and the $x$-dual violation $s^{k+1}=\rho A^\top B(z^{k+1}-z^k)$ is zero, so $w^{k+1}$ solves the VI.

The first rate ingredient is already in my hands. In this $H$-norm, (A.1) is exactly
$$\|w^{k+1}-w^\*\|_H^2\le\|w^k-w^\*\|_H^2-\|w^k-w^{k+1}\|_H^2.\tag{R.1}$$
The second ingredient is that these $H$-steps do not increase. Let me check that instead of hand-waving it. Define the auxiliary point
$$\widetilde w^k=\big(x^{k+1},\,z^{k+1},\,\lambda^k-\rho(Ax^{k+1}+Bz^k-c)\big),$$
and the matrices
$$M=\begin{pmatrix}I&0&0\\0&I&0\\0&-\rho B&I\end{pmatrix},\qquad Q=HM=\begin{pmatrix}0&0&0\\0&\rho B^\top B&0\\0&-B&\rho^{-1}I\end{pmatrix}.$$
The identity $w^{k+1}=w^k-M(w^k-\widetilde w^k)$ is just the three updates written in one line: the first two components are the new $x,z$, and the last component is $\lambda^k-\rho r^{k+1}$. The optimality conditions of the $x$- and $z$-minimizations combine into, for every $w\in\Omega$,
$$\theta(u)-\theta(\widetilde u^k)+(w-\widetilde w^k)^\top\big(F(\widetilde w^k)+Q(\widetilde w^k-w^k)\big)\ge0.\tag{R.2}$$
Put $w=\widetilde w^{k+1}$ in (R.2), put $w=\widetilde w^k$ in the same inequality one iteration later, and add them. The $\theta$ terms cancel, monotonicity of $F$ removes the $F$ difference with the correct sign, and what remains is
$$
(\widetilde w^k-\widetilde w^{k+1})^\top Q\big((w^k-w^{k+1})-(\widetilde w^k-\widetilde w^{k+1})\big)\ge0.\tag{R.3}
$$
Now the algebra is mechanical but important. Since $Q=HM$ and
$$(Q^\top+Q)-M^\top HM=\begin{pmatrix}0&0&0\\0&0&0\\0&0&\rho^{-1}I\end{pmatrix}\succeq0,$$
(R.3) implies
$$\|M(w^k-\widetilde w^k)\|_H^2-\|M(w^{k+1}-\widetilde w^{k+1})\|_H^2\ge0.$$
Using $M(w^k-\widetilde w^k)=w^k-w^{k+1}$, this is the monotone non-increase I need:
$$\|w^{k+1}-w^{k+2}\|_H^2\le\|w^k-w^{k+1}\|_H^2.\tag{R.4}$$
Now the rate is just the telescope. Summing (R.1) gives
$$\sum_{t=0}^{\infty}\|w^t-w^{t+1}\|_H^2\le\|w^0-w^\*\|_H^2.$$
Because the terms are non-increasing by (R.4), the latest one is no larger than the average of the first $k+1$ terms:
$$
(k+1)\|w^k-w^{k+1}\|_H^2\le\sum_{t=0}^k\|w^t-w^{t+1}\|_H^2\le\|w^0-w^\*\|_H^2,
$$
so
$$\|w^k-w^{k+1}\|_H^2\le\frac{1}{k+1}\,\|w^0-w^\*\|_H^2.$$
That is the non-ergodic $O(1/k)$ rate for the VI error measure, and the same telescope behind averaged iterates gives the usual ergodic $O(1/k)$ VI gap. Since the $H$-step sequence is both summable and non-increasing, the non-ergodic bound even sharpens asymptotically to $o(1/k)$.

Let me close the loop on the causal chain. I needed decomposable *and* robust; dual ascent is decomposable but fragile, the method of multipliers is robust (the penalty makes the inner problem well posed and the step-$\rho$ dual update keeps every iterate dual feasible) but its penalty's cross term couples the blocks. Freezing one block at a time turns that cross term into a constant, so alternating minimization keeps the augmented-Lagrangian robustness while restoring per-block updates — at the price of a single Gauss–Seidel sweep instead of a joint solve. The $z$-step with the dual update right after makes one optimality condition hold for free; the leftover two define the primal residual $r$ and the dual residual $s=\rho A^\top B\,\Delta z$. The certificate $V=\tfrac1\rho\|y-y^\*\|^2+\rho\|B(z-z^\*)\|^2$ decreases by $\rho(\|r\|^2+\|B\Delta z\|^2)$ every step — proved from the two min-step optimality conditions, the saddle inequality, and the monotonicity of $\partial g$ giving the cross-term sign — so both residuals vanish and the objective hits $p^\*$, for *any* $\rho>0$ with no tuning. The deeper reason any $\rho$ works is that the whole scheme is Douglas–Rachford splitting on the dual, which is the proximal point algorithm on a maximal-monotone splitting operator with stepsize fixed at $1$ — firmly nonexpansive, hence convergent for every positive resolvent parameter, and the stepsize-$1$ is exactly what lets the joint resolvent factor into the two sequential block solves. And the variational-inequality view turns the same Lyapunov decrease, in matrix-norm form, into an $O(1/k)$ worst-case rate.

I can now write the iteration and theorem cleanly in one place:

> **ADMM.** For closed proper convex $f,g$, form $L_\rho(x,z,y)=f(x)+g(z)+y^\top(Ax+Bz-c)+\tfrac\rho2\|Ax+Bz-c\|^2$ and iterate, for any $\rho>0$,
> $$x^{k+1}=\arg\min_x L_\rho(x,z^k,y^k),\quad z^{k+1}=\arg\min_z L_\rho(x^{k+1},z,y^k),\quad y^{k+1}=y^k+\rho(Ax^{k+1}+Bz^{k+1}-c).$$
> **Theorem.** If $f,g$ are closed proper convex and $L_0$ has a saddle point $(x^\*,z^\*,y^\*)$, then with $V^k=\tfrac1\rho\|y^k-y^\*\|^2+\rho\|B(z^k-z^\*)\|^2$ one has $V^{k+1}\le V^k-\rho\|r^{k+1}\|^2-\rho\|B(z^{k+1}-z^k)\|^2$, whence the primal residual $r^k=Ax^k+Bz^k-c\to0$, the dual residual $s^k=\rho A^\top B(z^k-z^{k-1})\to0$, and $f(x^k)+g(z^k)\to p^\*$, at worst-case rate $O(1/k)$. No strict convexity, finiteness, or full rank of $A,B$ is needed.

And a small driver that exhibits the residual decrease on a concrete two-block split:

```python
import numpy as np

def admm(prox_f, prox_g, A, B, c, rho,
         x0, z0, y0, eps_abs=1e-6, eps_rel=1e-4, max_iter=10000):
    """
    minimize f(x) + g(z)  s.t.  A x + B z = c.
    prox_f(z, y): returns argmin_x  f(x) + (rho/2)||A x + B z - c + y/rho||^2
    prox_g(x, y): returns argmin_z  g(z) + (rho/2)||A x + B z - c + y/rho||^2
    The augmented Lagrangian couples x,z only through the cross term in the square;
    freezing one block makes that term constant, so each prox is a single-block solve.
    """
    x, z, y = np.array(x0, float), np.array(z0, float), np.array(y0, float)
    p, n = c.size, x.size
    for k in range(max_iter):
        z_prev = z
        # ---- block 1: x-minimization (z, y frozen -> cross term is a constant) ----
        x = prox_f(z, y)
        # ---- block 2: z-minimization (x^{k+1}, y frozen) ----
        z = prox_g(x, y)
        # ---- dual ascent with step exactly rho -> (x,z,y) stays z-dual-feasible ----
        r = A @ x + B @ z - c                 # primal residual
        y = y + rho * r
        s = rho * (A.T @ (B @ (z - z_prev)))  # dual residual = rho A^T B (z^{k+1}-z^k)
        # ---- stopping from the suboptimality bound (the sqrt(p),sqrt(n) scale norms) ----
        eps_pri  = np.sqrt(p) * eps_abs + eps_rel * max(np.linalg.norm(A @ x),
                                                        np.linalg.norm(B @ z),
                                                        np.linalg.norm(c))
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * np.linalg.norm(A.T @ y)
        if np.linalg.norm(r) <= eps_pri and np.linalg.norm(s) <= eps_dual:
            break
    return x, z, y
```
