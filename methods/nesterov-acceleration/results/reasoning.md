I have a convex function $f$ on $\mathbb{R}^n$ whose gradient is Lipschitz with constant $\beta$, and I can only touch it through a first-order oracle: hand it a point, get back $f$ and $\nabla f$ there. I want to drive $f(x_k)-f^*$ to zero in as few oracle calls as possible. The natural thing is gradient descent, and I already know what it gives, so let me write that down carefully, because I want to *feel* exactly where it is slow before I try to beat it.

Smoothness means $f$ is trapped under a moving quadratic: $f(y)\le f(x)+\nabla f(x)^\top(y-x)+\tfrac{\beta}{2}\|y-x\|^2$. If I sit at $x$ and minimize that upper model in $y$, the minimizer is $y=x-\tfrac1\beta\nabla f(x)$, and plugging back gives the one fact I will lean on everywhere: a single step from $x$ buys at least $\tfrac{1}{2\beta}\|\nabla f(x)\|^2$ of descent,
$$f\!\big(x-\tfrac1\beta\nabla f(x)\big)\le f(x)-\tfrac{1}{2\beta}\|\nabla f(x)\|^2.$$
So with $x_{t+1}=x_t-\tfrac1\beta\nabla f(x_t)$ and $\delta_s=f(x_s)-f^*$ I have $\delta_{s+1}\le\delta_s-\tfrac{1}{2\beta}\|\nabla f(x_s)\|^2$. Convexity gives me a handle on that gradient norm: $\delta_s\le\nabla f(x_s)^\top(x_s-x^*)\le\|x_s-x^*\|\,\|\nabla f(x_s)\|$. And the distance $\|x_s-x^*\|$ doesn't grow — that's the standard contraction $\|x_{s+1}-x^*\|^2=\|x_s-x^*\|^2-\tfrac2\beta\nabla f(x_s)^\top(x_s-x^*)+\tfrac1{\beta^2}\|\nabla f(x_s)\|^2$, and coercivity $(\nabla f(x_s))^\top(x_s-x^*)\ge\tfrac1\beta\|\nabla f(x_s)\|^2$ makes the last two terms combine to $-\tfrac1{\beta^2}\|\nabla f(x_s)\|^2\le0$. So $\|x_s-x^*\|\le\|x_1-x^*\|=:R$, and then $\delta_{s+1}\le\delta_s-\tfrac{1}{2\beta R^2}\delta_s^2$. Set $\omega=\tfrac{1}{2\beta R^2}$; dividing through, $\tfrac{1}{\delta_{s+1}}-\tfrac1{\delta_s}\ge\omega$, telescope, and
$$f(x_t)-f^*\le\frac{2\beta R^2}{t-1}.$$
$O(1/t)$. To get $\varepsilon$ accuracy I pay $\Theta(1/\varepsilon)$ gradients. That's the wall I'm staring at. Nothing in that derivation tells me it's the best I can do — it just tells me gradient descent does this.

Before I try to beat it I should know what I'm aiming for. Is $1/t$ the floor, or is there room? Let me try to *prove* a floor, because if I can't beat the floor there's no point. I'll restrict to the honest black-box model: my first query is $x_1=0$ and every later query lies in the span of the gradients I've seen, $x_{t+1}\in\mathrm{Span}(g_1,\dots,g_t)$. Now I get to *design an adversary function* that is hard for any such method.

I need the gradient to reveal information one coordinate at a time. Take the tridiagonal quadratic
$$f(x)=\tfrac{\beta}{8}\,x^\top A_{2t+1}\,x-\tfrac{\beta}{4}\,x^\top e_1,\qquad (A_k)_{ii}=2,\ (A_k)_{i,i\pm1}=-1\ \text{up to row }k.$$
Why tridiagonal? Because $\nabla f(x)$ at a point whose nonzero coordinates are $x(1),\dots,x(s)$ only ever activates coordinate $s+1$ — the off-diagonal $-1$ couples neighbors and nothing further. Start at $0$: the gradient is $\propto e_1$, so $x_2$ can only move coordinate $1$; then $\nabla f(x_2)$ can reach coordinate $2$; by induction after $s$ steps $x_s$ is supported on the first $s-1$ coordinates. Meanwhile the true minimizer of the $k$-block solves $A_k x=e_1$, which is $x_k^*(i)=1-\tfrac{i}{k+1}$ — a *ramp* with mass spread over all $k$ coordinates. So a method that has only reached coordinate $t$ after $t$ steps is still missing all the mass in coordinates $t{+}1,\dots,2t{+}1$. Quantitatively, $f(x_s)-f^*=f_s(x_s)-f_{2t+1}^*\ge f_t^*-f_{2t+1}^*$ where $f_k^*=-\tfrac\beta8(1-\tfrac1{k+1})$, and $\|x_{2t+1}^*\|^2=\sum_{i=1}^{2t+1}(\tfrac{i}{2t+2})^2\le\tfrac{2t+2}{3}$, which gives
$$f(x_s)-f^*\ge f_t^*-f_{2t+1}^*=\tfrac\beta8\Big(\tfrac1{t+1}-\tfrac1{2t+2}\Big)\ge\frac{3\beta}{32}\frac{\|x_{2t+1}^*\|^2}{(t+1)^2}=\Omega\!\Big(\frac{\beta R^2}{t^2}\Big).$$
There it is. No first-order method can beat $\Omega(1/t^2)$ on smooth convex functions. So gradient descent's $1/t$ is a *full order* off the floor. The room I was hoping for is real, and it's exactly one factor of $t$.

Same exercise in the strongly convex case: take the infinite tridiagonal operator $A$ with the same structure and $f(x)=\tfrac{\alpha(\kappa-1)}{8}(\langle Ax,x\rangle-2\langle e_1,x\rangle)+\tfrac\alpha2\|x\|^2$. The same coordinate-leak forces $x_t(i)=0$ for $i\ge t$, while the minimizer is the *geometric* sequence $x^*(i)=\big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^i$ (you can check it satisfies the gradient equations). Strong convexity gives $f(x_t)-f^*\ge\tfrac\alpha2\|x_t-x^*\|^2\ge\tfrac\alpha2\sum_{i\ge t}x^*(i)^2$, so
$$f(x_t)-f^*\ge\frac\alpha2\Big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\Big)^{2(t-1)}\|x_1-x^*\|^2\approx\frac\alpha2\,e^{-4(t-1)/\sqrt\kappa}\,R^2.$$
The floor is $\sqrt\kappa$, and gradient descent sits at $\kappa$. Again the gap is a square root — the *same* quadratic improvement I half-remember momentum buying on quadratics. That can't be a coincidence. The floor is telling me: the right method should depend on $\sqrt\kappa$, not $\kappa$; on $1/t^2$, not $1/t$.

So what does momentum do? Polyak's heavy ball: $x_{k+1}=x_k-\alpha\nabla f(x_k)+\beta_{\!H}(x_k-x_{k-1})$. A ball with inertia rolling downhill, the $(x_k-x_{k-1})$ term carrying it along instead of letting it zig-zag across a narrow valley. On a strongly convex quadratic you diagonalize, the iteration becomes a $2\times2$ linear recurrence per eigenvalue, and tuning $\alpha,\beta_{\!H}$ to minimize the spectral radius gives $\beta_{\!H}=\big(\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}\big)^2$ and the $\sqrt\kappa$ rate. Beautiful — and it hits the floor on quadratics. So momentum is at least the right kind of mechanism.

But it only works on quadratics, and I need the whole smooth convex class. Why does it break? The eigenvalue argument is the giveaway: it lives entirely in the quadratic world where the map is linear. On a general $f$ the gradient at $x_k$ is read at the same point where the inertia is applied, and there's no descent quantity that this combination is guaranteed to improve — heavy ball can settle into a limit cycle and simply fail to converge. I can't certify it on a non-quadratic. So I have a method that hits the floor but only on the easiest functions, and an analysis technique (spectral radius) that can't escape them.

Stuck. Let me back up and think about what kind of *proof* could possibly certify acceleration on a general convex function, because the method should fall out of the proof technique, not the other way around. Gradient descent's proof rested on one quadratic upper model per step — that's a *local*, one-step, relaxation argument (each step decreases $f$). Maybe relaxation is too microscopic. Maybe I should track a *global* object across all iterations.

The thing I keep coming back to is that convexity hands me, at every point $y$ I evaluate, a *global linear lower bound* on $f$: $f(x)\ge f(y)+\nabla f(y)^\top(x-y)$ for all $x$. And if there's strong convexity $\mu$, I even get a *quadratic* lower bound $f(x)\ge f(y)+\nabla f(y)^\top(x-y)+\tfrac\mu2\|x-y\|^2$. Each oracle call gives me one more lower model that is valid *everywhere*. What if, instead of just descending, I *accumulate* these lower models into a running estimate of $f$ — a simple quadratic whose minimum I can track in closed form — and use that minimum as the global certificate for the point I output?

Let me make that precise. Suppose I can build a sequence of functions $\phi_k$ and a sequence of numbers $\lambda_k\to0$ such that
$$\phi_k(x)\le(1-\lambda_k)f(x)+\lambda_k\phi_0(x)\qquad\text{for all }x.$$
Call $\{\phi_k\},\{\lambda_k\}$ an *estimating sequence*. The idea: $\phi_k$ is mostly $f$ (weight $1-\lambda_k\to1$) plus a vanishing remnant of my initial guess $\phi_0$. Why is this any use? Suppose I can keep my iterate $x_k$ at or below the *minimum* of $\phi_k$: $f(x_k)\le\phi_k^*:=\min_x\phi_k(x)$. Then evaluate the estimating-sequence inequality at $x=x^*$:
$$f(x_k)\le\phi_k^*\le\phi_k(x^*)\le(1-\lambda_k)f(x^*)+\lambda_k\phi_0(x^*),$$
so
$$f(x_k)-f^*\le\lambda_k\big(\phi_0(x^*)-f^*\big).$$
The convergence rate of my method is *literally* the rate at which $\lambda_k\to0$. That's the whole game. If I can make $\lambda_k$ decay like $1/k^2$, I'm at the floor. The method I want is whatever keeps $f(x_k)\le\phi_k^*$ holding.

So two problems: how to *build* the estimating sequence, and how to keep $f(x_k)\le\phi_k^*$.

Building it is natural — average in one fresh lower model per step. Pick $\phi_0$ to be a simple quadratic $\phi_0(x)=\phi_0^*+\tfrac{\gamma_0}2\|x-v_0\|^2$, weights $\alpha_k\in(0,1)$, points $y_k$ to be chosen, and recurse
$$\lambda_{k+1}=(1-\alpha_k)\lambda_k,\qquad
\phi_{k+1}(x)=(1-\alpha_k)\phi_k(x)+\alpha_k\Big[f(y_k)+\nabla f(y_k)^\top(x-y_k)+\tfrac\mu2\|x-y_k\|^2\Big].$$
Why does this stay an estimating sequence? Induction. The bracket is a valid lower bound on $f$ by ($\mu$-strong) convexity — when $\mu=0$ it's the plain linear lower bound, when $\mu>0$ it's the quadratic one. So $\phi_{k+1}(x)\le(1-\alpha_k)\phi_k(x)+\alpha_k f(x)$, and feeding in the inductive bound $\phi_k(x)\le(1-\lambda_k)f(x)+\lambda_k\phi_0(x)$ and the definition $\lambda_{k+1}=(1-\alpha_k)\lambda_k$ collapses to $\phi_{k+1}(x)\le(1-\lambda_{k+1})f(x)+\lambda_{k+1}\phi_0(x)$. And $\lambda_k=\prod_{i<k}(1-\alpha_i)\to0$ as long as $\sum\alpha_i=\infty$. So the rate is set by how aggressively I'm allowed to pick the $\alpha_k$.

Now the crucial structural fact: averaging a quadratic $\phi_k$ with a quadratic lower model keeps $\phi_{k+1}$ a quadratic with the *same* curvature shape. The Hessian recurses as $\nabla^2\phi_{k+1}=(1-\alpha_k)\nabla^2\phi_k+\alpha_k\mu I$, so if $\nabla^2\phi_0=\gamma_0 I$ then $\nabla^2\phi_k=\gamma_k I$ with $\gamma_{k+1}=(1-\alpha_k)\gamma_k+\alpha_k\mu$. Therefore $\phi_k$ always has the canonical form
$$\phi_k(x)=\phi_k^*+\tfrac{\gamma_k}2\|x-v_k\|^2,$$
and I only need to track three scalars/vectors: the minimum value $\phi_k^*$, the minimizer $v_k$, and the curvature $\gamma_k$. Differentiating the recurrence and setting the gradient to zero gives the minimizer update
$$v_{k+1}=\tfrac1{\gamma_{k+1}}\big[(1-\alpha_k)\gamma_k v_k+\alpha_k\mu\, y_k-\alpha_k\nabla f(y_k)\big],$$
and evaluating the recurrence at $y_k$ (using the canonical form on both sides) gives, after the algebra,
$$\phi_{k+1}^*=(1-\alpha_k)\phi_k^*+\alpha_k f(y_k)-\tfrac{\alpha_k^2}{2\gamma_{k+1}}\|\nabla f(y_k)\|^2+\tfrac{\alpha_k(1-\alpha_k)\gamma_k}{\gamma_{k+1}}\Big(\tfrac\mu2\|y_k-v_k\|^2+\nabla f(y_k)^\top(v_k-y_k)\Big).$$
Let me not just assert that — let me see the cancellation. The canonical form says $\phi_{k+1}^*+\tfrac{\gamma_{k+1}}2\|y_k-v_{k+1}\|^2=\phi_{k+1}(y_k)=(1-\alpha_k)\big(\phi_k^*+\tfrac{\gamma_k}2\|y_k-v_k\|^2\big)+\alpha_k f(y_k)$. From the $v_{k+1}$ recurrence, $v_{k+1}-y_k=\tfrac1{\gamma_{k+1}}[(1-\alpha_k)\gamma_k(v_k-y_k)-\alpha_k\nabla f(y_k)]$, so
$$\tfrac{\gamma_{k+1}}2\|y_k-v_{k+1}\|^2=\tfrac1{2\gamma_{k+1}}\big[(1-\alpha_k)^2\gamma_k^2\|v_k-y_k\|^2-2\alpha_k(1-\alpha_k)\gamma_k\nabla f(y_k)^\top(v_k-y_k)+\alpha_k^2\|\nabla f(y_k)\|^2\big].$$
Subtract this from the right side; the coefficient on $\|y_k-v_k\|^2$ is $(1-\alpha_k)\tfrac{\gamma_k}2-\tfrac1{2\gamma_{k+1}}(1-\alpha_k)^2\gamma_k^2=(1-\alpha_k)\tfrac{\gamma_k}2\cdot\tfrac{\alpha_k\mu}{\gamma_{k+1}}$ (using $\gamma_{k+1}=(1-\alpha_k)\gamma_k+\alpha_k\mu$), which is exactly the $\tfrac\mu2\|y_k-v_k\|^2$ term above. Good — the bookkeeping is consistent, and the $\phi_{k+1}^*$ recurrence holds.

Now the second requirement: keep $f(x_k)\le\phi_k^*$. Suppose inductively $\phi_k^*\ge f(x_k)$. I want $\phi_{k+1}^*\ge f(x_{k+1})$. Look at the $\phi_{k+1}^*$ recurrence and lower-bound it: drop the strong-convexity term ($\ge0$) and use $\phi_k^*\ge f(x_k)\ge f(y_k)+\nabla f(y_k)^\top(x_k-y_k)$ (convexity, linear lower bound at $y_k$) to get
$$\phi_{k+1}^*\ge f(y_k)-\tfrac{\alpha_k^2}{2\gamma_{k+1}}\|\nabla f(y_k)\|^2+(1-\alpha_k)\nabla f(y_k)^\top\Big(\tfrac{\alpha_k\gamma_k}{\gamma_{k+1}}(v_k-y_k)+x_k-y_k\Big).$$
I want the right side to be $\ge f(x_{k+1})$. I have two free choices left — the step that produces $x_{k+1}$, and the point $y_k$ — and I'll spend each to kill one obstruction.

First the gradient term. I know from smoothness that a gradient step from $y_k$ gives $f(y_k)-\tfrac1{2\beta}\|\nabla f(y_k)\|^2\ge f(x_{k+1})$ when $x_{k+1}=y_k-\tfrac1\beta\nabla f(y_k)$. So if I force $\tfrac{\alpha_k^2}{2\gamma_{k+1}}=\tfrac1{2\beta}$, i.e.
$$\boxed{\ \beta\alpha_k^2=\gamma_{k+1}=(1-\alpha_k)\gamma_k+\alpha_k\mu\ }$$
then the first two terms are exactly $\ge f(x_{k+1})$. That equation *defines* $\alpha_k$ as the positive root of a quadratic — and notice what it's doing: it ties the step's smoothness gain $\tfrac1{2\beta}\|\nabla f\|^2$ to the estimating sequence's quadratic gain. The aggressiveness of $\alpha_k$ (hence the decay of $\lambda_k$) is throttled by exactly how much descent one gradient step can guarantee.

Second, the leftover cross term $(1-\alpha_k)\nabla f(y_k)^\top\big(\tfrac{\alpha_k\gamma_k}{\gamma_{k+1}}(v_k-y_k)+x_k-y_k\big)$. This has no fixed sign, so I can't bound it — I have to make it vanish. I still have $y_k$ free. Choose $y_k$ so the bracket is zero:
$$\tfrac{\alpha_k\gamma_k}{\gamma_{k+1}}(v_k-y_k)+x_k-y_k=0\ \Longrightarrow\ y_k=\frac{\alpha_k\gamma_k\,v_k+\gamma_{k+1}\,x_k}{\gamma_k+\alpha_k\mu}.$$
And *there* is the look-ahead. The point where I take the gradient is not $x_k$ and not $v_k$ — it's a specific convex blend of the current iterate $x_k$ and the running lower-model minimizer $v_k$. The estimate sequence didn't let me read the gradient at $x_k$ (that would leave the cross term and break the bound); it *demanded* I read it at the extrapolated point that cancels the cross term. This is the thing heavy ball got wrong: it applied inertia and read the gradient at the same place. Here the proof forces me to extrapolate *first*, then read the gradient at the extrapolated $y_k$, then step. That's why this one certifies on a general convex $f$ and heavy ball didn't — the $y_k$ choice is exactly what makes the global lower-model accounting close.

So I have a method, parameterized by the three sequences $(\gamma_k,v_k,\alpha_k)$ with $x_{k+1}=y_k-\tfrac1\beta\nabla f(y_k)$. Let me clean it into something I'd actually run, eliminating $v_k$ and $\gamma_k$ in favor of $x_k,y_k$. Since $\gamma_{k+1}=\beta\alpha_k^2$, the gradient step gives $\alpha_k\nabla f(y_k)=\tfrac{\gamma_{k+1}}{\alpha_k}(y_k-x_{k+1})$. The look-ahead relation gives $(\gamma_k+\alpha_k\mu)y_k=\alpha_k\gamma_k v_k+\gamma_{k+1}x_k$, and substituting these two identities into the $v_{k+1}$ recurrence collapses it to
$$v_{k+1}=x_k+\tfrac1{\alpha_k}(x_{k+1}-x_k),$$
and feeding that into the next $y_{k+1}$ gives a pure two-point extrapolation
$$y_{k+1}=x_{k+1}+\beta_k\,(x_{k+1}-x_k),\qquad \beta_k=\frac{\alpha_k(1-\alpha_k)}{\alpha_k^2+\alpha_{k+1}},$$
with $\alpha_{k+1}$ the positive root of $\beta\alpha_{k+1}^2=(1-\alpha_{k+1})\beta\alpha_k^2+\alpha_{k+1}\mu$, i.e. $\alpha_{k+1}^2=(1-\alpha_{k+1})\alpha_k^2+q_f\alpha_{k+1}$ where $q_f=\mu/\beta=1/\kappa$. The whole method is now: gradient step, then extrapolate. No $v_k$, no $\gamma_k$.

Now I get to read off the rates, because the rate *is* $\lambda_k$.

Take the pure convex case $\mu=0$ first, $q_f=0$. The coefficient recurrence becomes $\alpha_{k+1}^2=(1-\alpha_{k+1})\alpha_k^2$. It's cleaner to substitute $a_k=1/\alpha_{k-1}$ up to the index convention; concretely set $a_0=1$ and $a_{k+1}=\tfrac{1+\sqrt{1+4a_k^2}}2$, with the momentum coefficient $\tfrac{a_k-1}{a_{k+1}}$ — this is the same recurrence rewritten. The defining identity is $a_{k+1}^2-a_{k+1}=a_k^2$. From it, $a_{k+1}=\tfrac12+\sqrt{\tfrac14+a_k^2}\ge\tfrac12+a_k$, so $a_k\ge1+\tfrac k2$. In the usual one-based indexing this is the familiar coefficient approaching $(k-1)/(k+2)$, and the product $\lambda_k=\prod_{i<k}(1-\alpha_i)$ falls like $1/k^2$. Choosing $\phi_0$ with $\gamma_0=3\beta$, Lemma 2.2.4 gives $\lambda_k\le\tfrac{4}{3(k+1)^2}$, and Lemma 2.2.1 gives $f(x_k)-f^*\le\tfrac{\lambda_k}2(\beta+\gamma_0)R^2$, so
$$f(x_k)-f^*\le\frac{8\beta\|x_0-x^*\|^2}{3(k+1)^2}=O(1/k^2).$$
That matches the $\Omega(1/k^2)$ floor up to a constant. The momentum is time-varying: it starts at $0$ and creeps toward $1$, with the standard large-$k$ shorthand $(k-1)/(k+2)$ after the indexing is aligned. So the look-ahead isn't a fixed-weight momentum; it ramps up.

Let me sanity-check that $1/k^2$ a second way, without the estimate sequence, because I distrust a rate I've only derived one way. Take the two-point form directly. With $\lambda_0=0$, $\lambda_t=\tfrac{1+\sqrt{1+4\lambda_{t-1}^2}}2$ (so $\lambda_{t-1}^2=\lambda_t^2-\lambda_t$), $\gamma_t=\tfrac{1-\lambda_t}{\lambda_{t+1}}$, and updates $y_{t+1}=x_t-\tfrac1\beta\nabla f(x_t)$, $x_{t+1}=(1-\gamma_t)y_{t+1}+\gamma_t y_t$. Smoothness gives two inequalities at $x_t$: writing $x_t-y_{t+1}=\tfrac1\beta\nabla f(x_t)$,
$$f(y_{t+1})-f(y_t)\le\beta(x_t-y_{t+1})^\top(x_t-y_t)-\tfrac\beta2\|x_t-y_{t+1}\|^2,$$
$$f(y_{t+1})-f^*\le\beta(x_t-y_{t+1})^\top(x_t-x^*)-\tfrac\beta2\|x_t-y_{t+1}\|^2.$$
Multiply the first by $(\lambda_t-1)$, add the second, and let $\delta_s=f(y_s)-f^*$:
$$\lambda_t\delta_{t+1}-(\lambda_t-1)\delta_t\le\beta(x_t-y_{t+1})^\top(\lambda_t x_t-(\lambda_t-1)y_t-x^*)-\tfrac\beta2\lambda_t\|x_t-y_{t+1}\|^2.$$
Multiply by $\lambda_t$, use $\lambda_{t-1}^2=\lambda_t^2-\lambda_t$ and the identity $2a^\top b-\|a\|^2=\|b\|^2-\|b-a\|^2$ with $a=\lambda_t(x_t-y_{t+1})$, $b=\lambda_t x_t-(\lambda_t-1)y_t-x^*$:
$$\lambda_t^2\delta_{t+1}-\lambda_{t-1}^2\delta_t\le\tfrac\beta2\big(\|\lambda_t x_t-(\lambda_t-1)y_t-x^*\|^2-\|\lambda_t y_{t+1}-(\lambda_t-1)y_t-x^*\|^2\big).$$
The point of the $x_{t+1}$ definition is to make these brackets telescope: $x_{t+1}=y_{t+1}+\gamma_t(y_t-y_{t+1})$ rearranges to $\lambda_{t+1}x_{t+1}-(\lambda_{t+1}-1)y_{t+1}=\lambda_t y_{t+1}-(\lambda_t-1)y_t$, so with $u_s=\lambda_s x_s-(\lambda_s-1)y_s-x^*$ I get $\lambda_t^2\delta_{t+1}-\lambda_{t-1}^2\delta_t\le\tfrac\beta2(\|u_t\|^2-\|u_{t+1}\|^2)$. Sum from $s=1$ to $t-1$: $\lambda_{t-1}^2\delta_t\le\tfrac\beta2\|u_1\|^2=\tfrac\beta2 R^2$. And $\lambda_{t-1}\ge t/2$ by the same $a_k$ argument, so $\delta_t=f(y_t)-f^*\le\tfrac{2\beta R^2}{t^2}$. Independent derivation, same $1/k^2$. The potential here is $\lambda_{t-1}^2\delta_t+\tfrac\beta2\|u_t\|^2$ — a Lyapunov function — and the estimate-sequence and Lyapunov views are two readings of the same accounting. Good; I trust the rate now.

Now the strongly convex case $\mu>0$, $q_f=1/\kappa>0$. Here the estimating sequence keeps its $\tfrac\mu2$ quadratic lower model, $\gamma_k$ doesn't collapse to zero, and the coefficient equation $\beta\alpha_k^2=(1-\alpha_k)\gamma_k+\alpha_k\mu$ has a *constant* fixed point. If I start $\gamma_0=\mu$ then $\gamma_k\equiv\mu$ and the equation is $\beta\alpha^2=(1-\alpha)\mu+\alpha\mu=\mu$, so $\alpha\equiv\sqrt{\mu/\beta}=\sqrt{q_f}=1/\sqrt\kappa$ for all $k$. The momentum coefficient becomes constant too: $\beta_k=\tfrac{\alpha(1-\alpha)}{\alpha^2+\alpha}=\tfrac{1-\alpha}{1+\alpha}=\tfrac{1-1/\sqrt\kappa}{1+1/\sqrt\kappa}=\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$. This is the same $q$ that controls the hard quadratic tail and the square root of Polyak's tuned heavy-ball inertia $\beta_{\!H}=q^2$; the shared signal is the $\sqrt\kappa$ scale, but this coefficient is forced here by the estimate-sequence recurrence on the *whole* smooth strongly convex class, with a convergence certificate. And $\lambda_k=(1-\alpha)^k=(1-1/\sqrt\kappa)^k$, so
$$f(x_k)-f^*\le\tfrac{\beta+\mu}2\|x_0-x^*\|^2\big(1-\tfrac1{\sqrt\kappa}\big)^k\approx\tfrac{\beta+\mu}2 R^2\,e^{-k/\sqrt\kappa}.$$
$\sqrt\kappa$ in the exponent — matching the $\Omega(\sqrt\kappa\log\tfrac1\varepsilon)$ floor, a square-root improvement over gradient descent's $\kappa$. The method writes itself: $x_{k+1}=y_k-\tfrac1\beta\nabla f(y_k)$, $y_{k+1}=x_{k+1}+\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}(x_{k+1}-x_k)$. (And the convex $\mu=0$ method is the same thing with the constant momentum replaced by the ramping $\tfrac{a_k-1}{a_{k+1}}$, because at $\mu=0$ there's no fixed point and the coefficient must drift.)

I am still assuming I know $\beta$ to set the step, and in the strongly convex case I am assuming I know $\mu$. If I don't know $\beta$, I can replace the fixed step by a line search. From smoothness, as soon as a trial step $h$ falls below $1/\beta$ the descent inequality $f(y_k)-f(y_k-h\nabla f(y_k))\ge\tfrac h2(2-h\beta)\|\nabla f(y_k)\|^2\ge\tfrac h2\|\nabla f(y_k)\|^2$ holds, so I backtrack $h$ by halving from the previous accepted step until $f(y_k)-f(y_k-h\nabla f(y_k))\ge\tfrac h2\|\nabla f(y_k)\|^2$ is met; the accepted $h_k\ge\tfrac1{2\beta}$, and $\tfrac1{h_k}$ plays the role of the local smoothness constant in the coefficient equation. Because I start each line search from the previous accepted step, the total number of extra halvings over the whole run is logarithmic in the initial overestimate, of order $\log_2(2\beta h_{-1})$. With the line search the convex bound degrades only in the constant: $f(x_k)-f^*\le\tfrac{C}{(k+2)^2}$ with $C=4\beta R^2$ (and $C=2\beta R^2$ if $\beta$ is known and $h_k\equiv1/\beta$).

If instead I know $\mu$ but want to reuse the convex method with line search, there's a simpler route than re-deriving: run the $\mu=0$ accelerated method and *restart* it. The convex bound gives $f(x_N)-f^*\le\tfrac{2R^2}{h_N(N+2)^2}$, and with $h_N\ge\tfrac1{2\beta}$ and the strong-convexity inequality $f(y_0)-f^*\ge\tfrac\mu2\|y_0-x^*\|^2$, after $N=\lceil4\sqrt{\beta/\mu}\rceil-1=O(\sqrt\kappa)$ iterations the residual is at most halved: $f(x_N)-f^*\le\tfrac14\mu R^2\le\tfrac12(f(y_0)-f^*)$. So restart from $x_N$ and repeat; each $O(\sqrt\kappa)$ block halves the gap, giving $O(\sqrt\kappa\log\tfrac1\varepsilon)$ overall — the floor again, by a coarser but trivially-correct mechanism.

A last thing to notice, because it ran against my gradient-descent instincts the whole way: this method is *not relaxational*. $f$ need not decrease along $\{x_k\}$ or $\{y_k\}$ — the extrapolation deliberately overshoots, the iterate can climb on a given step, and that's precisely what buys the acceleration. Gradient descent's proof *needed* monotone descent; the estimate-sequence proof needs only that the iterate value stays below the running estimate minimum, which is a global, not a step-by-step, condition. Letting go of monotonicity is the price and the point.

Let me write the landed method as code. Two sequences $y$ (extrapolated, where the gradient is read) and $x$ (gradient-stepped). For the convex case the coefficient recurrence $a_{k+1}=\tfrac{1+\sqrt{1+4a_k^2}}2$ with momentum $\tfrac{a_k-1}{a_{k+1}}\to\tfrac{k-1}{k+2}$; for the strongly convex case the constant $\tfrac{\sqrt\kappa-1}{\sqrt\kappa+1}$.

```python
import numpy as np

def accelerated_gradient(obj, x0, n_iters):
    """Nesterov's accelerated gradient method.

    obj.beta = gradient-Lipschitz constant; obj.mu = strong-convexity (0 if convex).
    Returns the final iterate. Reads the gradient at the EXTRAPOLATED point y,
    then takes a gradient step -- look-ahead before gradient.
    """
    beta, mu = obj.beta, obj.mu
    x = np.array(x0, dtype=float)
    y = x.copy()                       # x1 = y1
    x_prev = x.copy()

    if mu > 0:
        # strongly convex: constant momentum = (sqrt(kappa)-1)/(sqrt(kappa)+1)
        sqrt_kappa = np.sqrt(beta / mu)
        momentum_const = (sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)
    a = 1.0                            # convex case: a_0 = 1

    for _ in range(n_iters):
        x_prev = x
        x = y - (1.0 / beta) * obj.grad(y)       # gradient step at the look-ahead point
        if mu > 0:
            mom = momentum_const                 # fixed inertia -> exp(-k/sqrt(kappa))
        else:
            a_next = (1.0 + np.sqrt(1.0 + 4.0 * a * a)) / 2.0   # a_{k+1}^2 - a_{k+1} = a_k^2
            mom = (a - 1.0) / a_next             # ramps 0 -> (k-1)/(k+2) -> 1
            a = a_next
        y = x + mom * (x - x_prev)               # EXTRAPOLATE first; gradient is read here next step

    return x
```

The chain in one breath: gradient descent on a smooth convex $f$ converges at $O(1/k)$ because each step only guarantees a local $\tfrac1{2\beta}\|\nabla f\|^2$ of descent; a coordinate-leak adversary proves the floor is $\Omega(1/k^2)$ (and $\Omega(\sqrt\kappa)$ when strongly convex), a full order better, with the square-root signature that matches momentum-on-quadratics; heavy ball reaches that floor but only on quadratics because it reads the gradient where it applies inertia and has no global certificate; switching from a one-step descent argument to a *global* estimating sequence — a running simple quadratic that accumulates the convexity lower bounds and whose minimum I keep above the iterate value — forces the gradient to be read not at $x_k$ but at the convex blend $y_k$ of $x_k$ and the lower-model minimizer (the look-ahead), which cancels the only obstruction in the bound; the step/coefficient relation $\beta\alpha_k^2=\gamma_{k+1}$ ties the smoothness descent to the estimate-sequence gain and makes $\lambda_k\sim1/k^2$ in the convex case and $\lambda_k=(1-1/\sqrt\kappa)^k$ in the strongly convex case, so $f(x_k)-f^*$ rides $\lambda_k$ straight down to $O(1/k^2)$ and $O(e^{-k/\sqrt\kappa})$ — the floor, attained by a method that costs one gradient per step and uses the same $\sqrt\kappa$ geometry as heavy ball without relying on a quadratic spectral proof.
