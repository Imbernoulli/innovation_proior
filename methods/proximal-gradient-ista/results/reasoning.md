I have a deblurring problem in front of me and it will not yield. Minimize $\|Ax-b\|^2+\lambda\|x\|_1$, where $A=RW$ is a blur composed with a wavelet synthesis, $x$ is the vector of wavelet coefficients, and the $\ell_1$ term is there because the true image is sparse in this basis. The thing is large. Image-sized large — hundreds of thousands, millions of unknowns — and $A$ is dense, a convolution that mixes everything with everything. I keep wanting to reach for an interior-point solver, because the $\ell_1$ problem is a second-order cone program and those solvers converge in a handful of Newton steps. But each of those steps wants to form and factor a big dense linear system, and at this size I cannot even hold the system, let alone factor it. So that door is shut. Whatever I do, the per-iteration work has to be cheap: a couple of matrix–vector products with $A$ and $A^\top$, and nothing heavier. That constraint is non-negotiable, and it forces me into first-order methods — methods that only ever look at $f$, $g$, and gradients. The price of that cheapness is that I will need many iterations, so the only thing that will save me is the *rate*: how fast $F(x_k)-F^*$ falls.

Let me be honest about the obvious move first, because I want to know exactly why it is not enough. $F(x)=\|Ax-b\|^2+\lambda\|x\|_1$ is convex. I could just compute a subgradient of the whole thing and step. A subgradient of $\|x\|_1$ is $\lambda\,\mathrm{sign}(x)$ componentwise, with the convention that at a zero coordinate any value in $[-\lambda,\lambda]$ is admissible. Step along the negative subgradient with a diminishing step. It works in the sense that it converges. But the rate for a nonsmooth convex objective with a subgradient method is $O(1/\sqrt{k})$ — to get one more digit of accuracy I pay a hundredfold in iterations. That is hopeless here. And when I stare at *why* it is so slow, the answer is irritating: the method treats $\|Ax-b\|^2$ and $\lambda\|x\|_1$ as one undifferentiated lump. But the data term is smooth — beautifully smooth, a quadratic — and the subgradient method gets none of the benefit of that smoothness because it dilutes everything into a single subgradient direction. The nonsmoothness lives *entirely* in the $\ell_1$ term, and it is a very structured, very simple kind of nonsmoothness. I am paying the nonsmooth price on the smooth part for no reason.

So the real object is not $\|Ax-b\|^2+\lambda\|x\|_1$ specifically; it is the shape

$$F(x)=f(x)+g(x),\qquad f\ \text{smooth convex},\quad g\ \text{convex, possibly nonsmooth}.$$

If I can find a method that gets the smooth rate out of $f$ while paying for $g$ only through whatever cheap structure $g$ has, that would dominate the subgradient approach. The question is how to keep $f$ and $g$ apart while marching downhill.

Let me think about what the smoothness of $f$ actually gives me, concretely. $\nabla f$ is Lipschitz with some constant $L$: $\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$. The fact I always lean on is the consequence of that — the descent lemma. Integrate the gradient along the segment from $y$ to $x$:

$$f(x)=f(y)+\int_0^1\langle \nabla f(y+s(x-y)),\,x-y\rangle\,ds.$$

Subtract $\langle x-y,\nabla f(y)\rangle$ from both sides:

$$f(x)-f(y)-\langle x-y,\nabla f(y)\rangle=\int_0^1\langle \nabla f(y+s(x-y))-\nabla f(y),\,x-y\rangle\,ds,$$

and bound the integrand by Cauchy–Schwarz and Lipschitz: $\langle \nabla f(y+s(x-y))-\nabla f(y),x-y\rangle\le \|\nabla f(y+s(x-y))-\nabla f(y)\|\,\|x-y\|\le L s\|x-y\|^2$. Integrate $\int_0^1 Ls\,ds=L/2$. So

$$f(x)\le f(y)+\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|^2\quad\text{for all }x,y.$$

That is the whole leverage of smoothness: a quadratic with curvature $L$ that sits *above* $f$ everywhere and touches it at $y$. A global upper model, recomputed at whatever point I please.

Now here is the move I keep circling back to. I do not have to model $g$ at all — I can keep it exact. At my current point $y=x_{k-1}$, replace only the smooth part by its upper quadratic and *leave $g$ alone*:

$$x_k=\arg\min_x\ \underbrace{f(y)+\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|^2}_{\text{upper model of }f}\ +\ g(x).$$

Why is this a sound thing to do? Because at $x=y$ the model equals $f(y)+g(y)=F(y)$, and for any $x$ the model is $\ge f(x)+g(x)=F(x)$ (the $f$-part is an upper bound, $g$ is itself). So I am minimizing a function that touches $F$ at my current point and dominates it everywhere. Whatever minimizes the model has model value $\le F(y)$, hence $F$-value $\le$ model value $\le F(y)$. I descend on $F$ by descending on a one-shot majorizer. This is exactly the majorize–minimize idea, and it does not care whether $g$ is smooth — it only needs $g$ to be something I can minimize against a quadratic.

Let me simplify the minimization. The terms in $x$ are $\langle x,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|^2+g(x)$. Complete the square in the quadratic-plus-linear part:

$$\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|^2=\tfrac{L}{2}\Big\|x-\big(y-\tfrac1L\nabla f(y)\big)\Big\|^2-\tfrac{1}{2L}\|\nabla f(y)\|^2.$$

The last term is constant in $x$, so

$$x_k=\arg\min_x\ g(x)+\tfrac{L}{2}\Big\|x-\big(y-\tfrac1L\nabla f(y)\big)\Big\|^2.$$

Look at what fell out. Inside the squared distance is $y-\tfrac1L\nabla f(y)$ — that is just an ordinary gradient step on $f$ with step $t=1/L$. And then I am asked: of all $x$, which one balances being small in $g$ against being close to that gradient-stepped point? Write $v=y-\tfrac1L\nabla f(y)$ and $t=1/L$, and the subproblem is

$$\arg\min_x\ g(x)+\tfrac{1}{2t}\|x-v\|^2.$$

I take a gradient step on the smooth part, then I solve *this* little problem against $g$. So the whole iteration is: gradient step on $f$, then this $g$-step. The structure is clean and it is two clearly separated pieces — an explicit step using $\nabla f$, and an implicit step that only ever sees $g$. The explicit step carries all the dependence on $f$; the implicit step carries all the dependence on $g$. That is the separation I was after.

This object $\arg\min_x g(x)+\tfrac{1}{2t}\|x-v\|^2$ — I have seen it. It is Moreau's proximal map. For a convex $\varphi$,

$$\mathrm{prox}_{t\varphi}(v)=\arg\min_x\ \varphi(x)+\tfrac{1}{2t}\|x-v\|^2,$$

a uniquely defined point (the objective is strongly convex), the implicit/backward partner of an explicit gradient step. When $\varphi$ is the indicator of a convex set it is just Euclidean projection. So my iteration is

$$x_k=\mathrm{prox}_{tg}\big(y-t\nabla f(y)\big),\qquad t=\tfrac1L .$$

Gradient step forward on the smooth $f$, prox backward on the nonsmooth $g$. Forward–backward.

I want to double-check that this is actually solving the right problem — that its fixed points are minimizers of $F$. A point $x^\*$ minimizes $F$ iff $0\in\nabla f(x^\*)+\partial g(x^\*)$ (Fermat plus the sum rule, legal because $f$ is smooth). Rearrange: $-\nabla f(x^\*)\in\partial g(x^\*)$, i.e. $\big(x^\*-t\nabla f(x^\*)\big)-x^\*\in t\,\partial g(x^\*)$. Now recall the optimality of the prox: $u=\mathrm{prox}_{t\varphi}(w)$ holds exactly when $w-u\in t\,\partial\varphi(u)$ (differentiate the strongly convex prox objective). With $\varphi=g$, $w=x^\*-t\nabla f(x^\*)$, $u=x^\*$, that inclusion says precisely $x^\*=\mathrm{prox}_{tg}\big(x^\*-t\nabla f(x^\*)\big)$. The fixed points of my map are exactly the minimizers of $F$. Good — the iteration is not just a descent heuristic, its rest points are the right ones.

Everything now hinges on whether $\mathrm{prox}_{tg}$ is cheap, because that is the per-iteration cost beyond the matvec. For $g(x)=\lambda\|x\|_1$ this had better be cheap or the whole plan is pointless. The $\ell_1$ norm is separable — $\|x\|_1=\sum_i|x_i|$ — and the quadratic penalty $\tfrac{1}{2t}\|x-v\|^2=\sum_i\tfrac{1}{2t}(x_i-v_i)^2$ is separable too, so the prox decouples completely into one scalar problem per coordinate:

$$\min_{x_i}\ \lambda|x_i|+\tfrac{1}{2t}(x_i-v_i)^2.$$

Solve it by hand. Away from $0$ the objective is smooth; set the derivative to zero. If $x_i>0$: $\lambda+\tfrac1t(x_i-v_i)=0\Rightarrow x_i=v_i-\lambda t$, which is consistent ($x_i>0$) only when $v_i>\lambda t$. If $x_i<0$: $-\lambda+\tfrac1t(x_i-v_i)=0\Rightarrow x_i=v_i+\lambda t$, consistent only when $v_i<-\lambda t$. And if neither holds — that is, $|v_i|\le\lambda t$ — then the minimizer cannot be at a nonzero point of either sign, so it must be at the kink $x_i=0$. Check the kink directly with the subgradient: $0$ is optimal iff $0\in\lambda\,\partial|0|+\tfrac1t(0-v_i)=[-\lambda,\lambda]-\tfrac{v_i}{t}$, i.e. iff $\tfrac{v_i}{t}\in[-\lambda,\lambda]$, i.e. $|v_i|\le\lambda t$. Exactly the leftover region, and the cases tile the line with no gap. Stitch them together:

$$x_i=\begin{cases}v_i-\lambda t,& v_i>\lambda t\\[2pt] 0,& |v_i|\le\lambda t\\[2pt] v_i+\lambda t,& v_i<-\lambda t\end{cases}\;=\;\mathrm{sign}(v_i)\,\big(|v_i|-\lambda t\big)_+ .$$

That is soft-thresholding. The prox of the $\ell_1$ norm *is* the shrinkage operator $T_{\lambda t}$. It costs an absolute value, a subtraction, and a clamp per coordinate — $O(n)$, nothing. And there is a small thing worth noticing: a hard cutoff would set everything below the threshold to zero and leave the rest untouched, but the prox does not do that — it pulls every surviving coordinate *toward* zero by exactly $\lambda t$. That bias is not a defect; it is the $\ell_1$ penalty doing its job inside the minimization, the same shrinkage that wavelet denoising has used for years, now arising not as a denoising heuristic but as the exact backward step of my splitting.

So with $f(x)=\|Ax-b\|^2$, $\nabla f(x)=2A^\top(Ax-b)$, my iteration is

$$x_{k+1}=T_{\lambda t}\big(x_k-2t\,A^\top(Ax_k-b)\big),$$

a matvec to form $Ax_k-b$, a matvec with $A^\top$, a soft-threshold. This is exactly the iterative shrinkage-thresholding that people have been running — derived independently as a thresholded Landweber iteration, as an EM-type wavelet restoration, by adding a "surrogate" term $C\|x-a\|^2-\|A(x-a)\|^2$ that decouples the troublesome $A^\top A$ coupling so the minimization separates into per-coordinate soft-thresholds (Daubechies, Defrise & De Mol, 2004), and as proximal forward–backward splitting (Bruck 1977; Passty 1979; analyzed by Combettes & Wajs 2005). I have just walked into ISTA from the majorize-minimize side, and the bonus is that nothing in the derivation used the $\ell_1$ structure until the very last step. Any $g$ with a cheap prox plugs in: a constraint set gives projection, a group norm gives block soft-thresholding, and so on. I will call the general method ISTA too, with $g$ arbitrary and $\mathrm{prox}_{tg}$ in place of $T$.

Now the uncomfortable part. ISTA is simple and the iterations are cheap, but in practice it crawls. I want to know its rate, not just whether it converges, because the rate is what I am living and dying by. Let me write the iteration in the cleanest possible form. Fix $L\ge L(f)$ and define, at any anchor $y$,

$$Q_L(x,y)=f(y)+\langle x-y,\nabla f(y)\rangle+\tfrac{L}{2}\|x-y\|^2+g(x),\qquad p_L(y)=\arg\min_x Q_L(x,y).$$

ISTA is $x_k=p_L(x_{k-1})$. From the descent lemma, $L\ge L(f)$ guarantees $F(p_L(y))\le Q_L(p_L(y),y)$ — the model genuinely dominates, so each step really descends.

I need one workhorse inequality, and I want it general enough to serve the faster method I am about to chase, not just ISTA. Suppose $L$ is such that $F(p_L(y))\le Q_L(p_L(y),y)$. I claim that for *every* $x$,

$$F(x)-F(p_L(y))\ \ge\ \tfrac{L}{2}\|p_L(y)-y\|^2+L\,\langle y-x,\,p_L(y)-y\rangle.$$

Prove it. Start from $F(x)-F(p_L(y))\ge F(x)-Q_L(p_L(y),y)$, using the domination. Lower-bound $F(x)=f(x)+g(x)$ by convexity of each piece: $f(x)\ge f(y)+\langle x-y,\nabla f(y)\rangle$, and $g(x)\ge g(p_L(y))+\langle x-p_L(y),\gamma\rangle$ where $\gamma\in\partial g(p_L(y))$. Which $\gamma$? The one from the prox optimality of $p_L(y)$: minimizing $Q_L(\cdot,y)$ gives $\nabla f(y)+L(p_L(y)-y)+\gamma=0$, so $\gamma=-\nabla f(y)-L(p_L(y)-y)$. Add the two lower bounds:

$$F(x)\ge f(y)+\langle x-y,\nabla f(y)\rangle+g(p_L(y))+\langle x-p_L(y),\gamma\rangle.$$

And $Q_L(p_L(y),y)=f(y)+\langle p_L(y)-y,\nabla f(y)\rangle+\tfrac{L}{2}\|p_L(y)-y\|^2+g(p_L(y))$. Subtract:

$$F(x)-Q_L(p_L(y),y)\ge -\tfrac{L}{2}\|p_L(y)-y\|^2+\langle x-p_L(y),\,\nabla f(y)+\gamma\rangle.$$

Substitute $\nabla f(y)+\gamma=-L(p_L(y)-y)$:

$$=-\tfrac{L}{2}\|p_L(y)-y\|^2+L\,\langle x-p_L(y),\,y-p_L(y)\rangle.$$

Now rewrite $\langle x-p_L(y),y-p_L(y)\rangle$. Set $d=p_L(y)-y$; then $x-p_L(y)=(x-y)-d$ and $y-p_L(y)=-d$, so $\langle x-p_L(y),y-p_L(y)\rangle=\langle (x-y)-d,-d\rangle=-\langle x-y,d\rangle+\|d\|^2=\|d\|^2+\langle y-x,d\rangle$. Therefore

$$F(x)-F(p_L(y))\ge -\tfrac{L}{2}\|d\|^2+L\big(\|d\|^2+\langle y-x,d\rangle\big)=\tfrac{L}{2}\|d\|^2+L\langle y-x,d\rangle,$$

which is the claim. This one inequality is the engine. Two specializations of it do everything.

For the ISTA rate, apply it at $y=x_n$, $p_L(y)=x_{n+1}$, and $L=L_{n+1}$, where in the fixed-step case $L_{n+1}=L(f)$ and in the backtracking case $L_{n+1}\le \alpha L(f)$ with $\alpha=\eta$. First take $x=x^\*$:

$$F(x^\*)-F(x_{n+1})\ge \tfrac{L_{n+1}}{2}\|x_{n+1}-x_n\|^2+L_{n+1}\langle x_n-x^\*,x_{n+1}-x_n\rangle.$$

Multiply by $2/L_{n+1}$ and notice the right side is a perfect difference of squares: $\|x_{n+1}-x_n\|^2+2\langle x_n-x^\*,x_{n+1}-x_n\rangle=\|x^\*-x_{n+1}\|^2-\|x^\*-x_n\|^2$ (expand $\|x^\*-x_{n+1}\|^2=\|(x^\*-x_n)-(x_{n+1}-x_n)\|^2$). Because $F(x^\*)-F(x_{n+1})\le0$ and $L_{n+1}\le\alpha L(f)$, replacing $L_{n+1}$ by the larger denominator $\alpha L(f)$ only makes the left side larger, so

$$\tfrac{2}{\alpha L(f)}\big(F(x^\*)-F(x_{n+1})\big)\ge \|x^\*-x_{n+1}\|^2-\|x^\*-x_n\|^2.$$

Sum over $n=0,\dots,k-1$; the right telescopes to $\|x^\*-x_k\|^2-\|x^\*-x_0\|^2\ge -\|x^\*-x_0\|^2$:

$$\tfrac{2}{\alpha L(f)}\Big(kF(x^\*)-\textstyle\sum_{n=1}^{k}F(x_n)\Big)\ge -\|x^\*-x_0\|^2.$$

Now take the same engine inequality at $x=y=x_n$ to get $F(x_n)-F(x_{n+1})\ge\tfrac{L_{n+1}}{2}\|x_n-x_{n+1}\|^2\ge0$, so the values are nonincreasing. Then every $F(x_n)$ in the sum with $1\le n\le k$ is at least $F(x_k)$, hence $kF(x^\*)-\sum_{n=1}^k F(x_n)\le k(F(x^\*)-F(x_k))$. The telescoped left side is therefore no larger than $\tfrac{2k}{\alpha L(f)}(F(x^\*)-F(x_k))$, and since it is already at least $-\|x^\*-x_0\|^2$, I get $\tfrac{2k}{\alpha L(f)}\big(F(x^\*)-F(x_k)\big)\ge-\|x^\*-x_0\|^2$. Hence

$$F(x_k)-F(x^\*)\le \frac{\alpha L(f)\,\|x_0-x^\*\|^2}{2k}.$$

So ISTA is $O(1/k)$ in function value — exactly the rate of the plain gradient method, which makes sense: when $g\equiv0$, $p_L$ is the gradient step and ISTA *is* gradient descent. I have confirmed both the good news (it matches the smooth gradient rate, far better than the subgradient method's $1/\sqrt{k}$) and the bad news (it is only $1/k$). To squeeze out one more digit I run ten times longer. On a million-pixel deblurring that is the difference between minutes and days.

So: can I do better than $1/k$ with a method that is *just as cheap per iteration*? For a moment I worry the answer is no — that $1/k$ is simply what first-order methods cost. But I recall the smooth case. For minimizing a smooth convex $f$ alone, the gradient method gives $1/k$, yet Nesterov (1983) found a gradient method that achieves $1/k^2$ using no more than one gradient per step plus one extra cheaply-computed point — and $1/k^2$ is provably the best any method can do that only sees first-order information at a sequence of points (the Nemirovsky–Yudin lower bound). So $1/k$ is *not* the floor; the floor is $1/k^2$, and the gap is reachable for smooth problems. The trouble is that this acceleration was built for smooth objectives, and its usual derivation leans on the objective being differentiable. My objective is not — the $\ell_1$ term has corners. The question is whether the accelerating device survives when I swap the gradient step for my prox step.

Let me look at what Nesterov's device actually is, stripped down. The plain method evaluates the gradient at the current iterate. The accelerated method evaluates it instead at an *extrapolated* point — the current iterate pushed a little further in the direction it just moved, a momentum term carrying the last step forward. So my instinct is: keep my prox step $p_L(\cdot)$ exactly as is, but apply it not at $x_{k-1}$ but at a cleverly extrapolated point $y_k$ built from the last two iterates:

$$x_k=p_L(y_k),\qquad y_{k+1}=x_k+(\text{coefficient})\cdot(x_k-x_{k-1}).$$

The whole bet is that one prox evaluation, at an extrapolated anchor, costs essentially the same as ISTA (still one $\nabla f$, one prox per step) yet buys the $1/k^2$ rate. The coefficient and the anchor I will not guess — I will let the convergence proof tell me exactly what they must be.

Here is how I will hunt for them. The ISTA proof telescoped a *first-order* quantity, $\|x^\*-x_n\|^2$, and that gave $1/k$. To get $1/k^2$ I want to telescope something that, when it stays bounded, forces $F(x_k)-F^*$ to decay like $1/k^2$. So I will carry a *weight* $t_k$ that grows like $k$, track $t_k^2\big(F(x_k)-F^\*\big)$, and try to bound it by a constant; then $F(x_k)-F^\*\le \text{const}/t_k^2\sim 1/k^2$. The weights, the anchor, and the momentum coefficient all have to conspire so that a clean telescoping inequality holds. Let me derive that conspiracy.

Write $v_k=F(x_k)-F(x^\*)\ge0$. Apply the engine inequality twice, both at the anchor $y=y_{k+1}$ with $L=L_{k+1}$ and $p_L(y_{k+1})=x_{k+1}$ — once at $x=x_k$ and once at $x=x^\*$:

$$\tfrac{2}{L_{k+1}}(v_k-v_{k+1})\ge\|x_{k+1}-y_{k+1}\|^2+2\langle x_{k+1}-y_{k+1},\,y_{k+1}-x_k\rangle,$$
$$-\tfrac{2}{L_{k+1}}v_{k+1}\ge\|x_{k+1}-y_{k+1}\|^2+2\langle x_{k+1}-y_{k+1},\,y_{k+1}-x^\*\rangle.$$

(The first is $F(x_k)-F(x_{k+1})$, the second $F(x^\*)-F(x_{k+1})$, both from $F(x)-F(p_L(y))\ge\tfrac{L}{2}\|d\|^2+L\langle y-x,d\rangle$ with $d=x_{k+1}-y_{k+1}$ and using $\tfrac{L}{2}\|d\|^2+L\langle y-x,d\rangle=\tfrac{L}{2}\big(\|d\|^2+2\langle d,y-x\rangle\big)$.) I want to combine these so the $v$'s appear with weights that will become $t_k^2$. Multiply the first by $(t_{k+1}-1)$ and add the second:

$$\tfrac{2}{L_{k+1}}\big((t_{k+1}-1)v_k-t_{k+1}v_{k+1}\big)\ge t_{k+1}\|x_{k+1}-y_{k+1}\|^2+2\langle x_{k+1}-y_{k+1},\ t_{k+1}y_{k+1}-(t_{k+1}-1)x_k-x^\*\rangle.$$

Multiply through by $t_{k+1}$, so that the left becomes a difference of the weighted values I care about — and here I impose the relation that makes it telescope:

$$t_k^2=t_{k+1}^2-t_{k+1}.$$

With that, $t_{k+1}(t_{k+1}-1)=t_k^2$, and the left side is $\tfrac{2}{L_{k+1}}\big(t_k^2 v_k-t_{k+1}^2 v_{k+1}\big)$. The right side becomes

$$\|t_{k+1}(x_{k+1}-y_{k+1})\|^2+2t_{k+1}\langle x_{k+1}-y_{k+1},\ t_{k+1}y_{k+1}-(t_{k+1}-1)x_k-x^\*\rangle.$$

This is begging for the polarization identity $\|b-a\|^2+2\langle b-a,a-c\rangle=\|b-c\|^2-\|a-c\|^2$. Match it with $a=t_{k+1}y_{k+1}$, $b=t_{k+1}x_{k+1}$, $c=(t_{k+1}-1)x_k+x^\*$: then $b-a=t_{k+1}(x_{k+1}-y_{k+1})$ and $a-c=t_{k+1}y_{k+1}-(t_{k+1}-1)x_k-x^\*$, exactly the two vectors above. So the right side equals

$$\|t_{k+1}x_{k+1}-(t_{k+1}-1)x_k-x^\*\|^2-\|t_{k+1}y_{k+1}-(t_{k+1}-1)x_k-x^\*\|^2.$$

Now I see what the anchor $y_{k+1}$ must be. I want that second, subtracted norm to be the *previous* version of the first, so the whole thing telescopes. Define

$$u_k=t_k x_k-(t_k-1)x_{k-1}-x^\*.$$

The first norm above is exactly $\|u_{k+1}\|^2$. For the second norm to be $\|u_k\|^2$, I need

$$t_{k+1}y_{k+1}-(t_{k+1}-1)x_k=t_k x_k-(t_k-1)x_{k-1},$$

i.e. $t_{k+1}y_{k+1}=t_{k+1}x_k+(t_k-1)x_k-(t_k-1)x_{k-1}=t_{k+1}x_k+(t_k-1)(x_k-x_{k-1})$, which gives the anchor

$$y_{k+1}=x_k+\frac{t_k-1}{t_{k+1}}\,(x_k-x_{k-1}).$$

There it is — the momentum coefficient was never a free choice; it is forced, to the last symbol, by demanding that the proof telescope. With this $y_{k+1}$ the inequality collapses to

$$\tfrac{2}{L_{k+1}}\big(t_k^2 v_k-t_{k+1}^2 v_{k+1}\big)\ge \|u_{k+1}\|^2-\|u_k\|^2.$$

In the fixed-step case this is already the recursion I want. In the backtracking case the chosen $L_k$ sequence is nondecreasing, so $L_{k+1}\ge L_k$; because $v_k\ge0$, replacing $\tfrac{2}{L_{k+1}}t_k^2v_k$ by the larger $\tfrac{2}{L_k}t_k^2v_k$ keeps the inequality true:

$$\tfrac{2}{L_k}t_k^2 v_k-\tfrac{2}{L_{k+1}}t_{k+1}^2v_{k+1}\ge \|u_{k+1}\|^2-\|u_k\|^2.$$

Set $a_k=\tfrac{2}{L_k}t_k^2 v_k$ and $b_k=\|u_k\|^2$ (with $L_k\equiv L$ in the constant-step case). The recursion is $a_k-a_{k+1}\ge b_{k+1}-b_k$, i.e. $a_{k+1}+b_{k+1}\le a_k+b_k$: the quantity $a_k+b_k$ never increases. So if I can show $a_1+b_1\le c$ for $c=\|x_0-x^\*\|^2$, then $a_k\le a_k+b_k\le c$ for all $k$, giving $\tfrac{2}{L_k}t_k^2 v_k\le\|x_0-x^\*\|^2$, i.e.

$$v_k\le\frac{L_k\,\|x_0-x^\*\|^2}{2\,t_k^2}.$$

Two loose ends: the base case, and how big $t_k$ is.

Base case. Choose $t_1=1$, $y_1=x_0$. Then $a_1=\tfrac{2}{L_1}t_1^2 v_1=\tfrac{2}{L_1}v_1$ and $b_1=\|u_1\|^2=\|t_1 x_1-(t_1-1)x_0-x^\*\|^2=\|x_1-x^\*\|^2$. I need $\tfrac{2}{L_1}v_1+\|x_1-x^\*\|^2\le\|x_0-x^\*\|^2$. Apply the engine inequality at $x=x^\*$, $y=y_1=x_0$, $L=L_1$: $F(x^\*)-F(x_1)\ge\tfrac{L_1}{2}\|x_1-x_0\|^2+L_1\langle x_0-x^\*,x_1-x_0\rangle=\tfrac{L_1}{2}\big(\|x_1-x^\*\|^2-\|x_0-x^\*\|^2\big)$ (same difference-of-squares as before). So $-v_1=F(x^\*)-F(x_1)\ge\tfrac{L_1}{2}\big(\|x_1-x^\*\|^2-\|x_0-x^\*\|^2\big)$, i.e. $\tfrac{2}{L_1}v_1\le\|x_0-x^\*\|^2-\|x_1-x^\*\|^2$, which is exactly $a_1+b_1\le c$. The base case holds.

Size of $t_k$. The relation $t_k^2=t_{k+1}^2-t_{k+1}$ solved forward for $t_{k+1}$ (taking the positive root) is

$$t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2}.$$

From this, $t_{k+1}=\tfrac12+\tfrac12\sqrt{1+4t_k^2}\ge\tfrac12+\tfrac12\sqrt{4t_k^2}=t_k+\tfrac12$, so $t_k\ge t_1+\tfrac{k-1}{2}=1+\tfrac{k-1}{2}=\tfrac{k+1}{2}$. Therefore $t_k^2\ge(k+1)^2/4$ and

$$F(x_k)-F(x^\*)\le\frac{L_k\,\|x_0-x^\*\|^2}{2t_k^2}\le\frac{2\,\alpha L(f)\,\|x_0-x^\*\|^2}{(k+1)^2}.$$

$O(1/k^2)$. The same prox step, the same one gradient and one shrinkage per iteration, evaluated at the extrapolated point $y_k$, and the rate jumps from $1/k$ to $1/k^2$. To gain a digit I now run roughly three times longer instead of ten times — on the deblurring problem that is the qualitative difference between unusable and usable, and it matches the lower bound, so I cannot do asymptotically better with a first-order method. I will call this fast version FISTA.

A few things I want to be careful about. First, this is *not* a descent method — I never proved $F(x_k)$ is monotone, and in fact the extrapolation can overshoot so that the value occasionally ticks up. That is fine; in the constant-step notation the potential $t_k^2 v_k+\tfrac{L}{2}\|u_k\|^2$ is what decreases, not $v_k$ itself, and that is enough for the rate. (If monotonicity is wanted for safety one can guard each step, but it is not needed for the bound.) Second, nothing above used the $\ell_1$ structure — the engine inequality and the telescoping hold for any smooth convex $f$ and any convex $g$; the $\ell_1$ case just supplies the cheap soft-threshold for the prox. When $g\equiv0$ this reduces to an accelerated gradient method for smooth minimization, and the proof I just did is, incidentally, a clean self-contained proof of that too.

Third, the step. I have been assuming $L=L(f)$ is known. For $f(x)=\|Ax-b\|^2$ it is $L(f)=2\lambda_{\max}(A^\top A)$ — in convolutional deblurring there may be transform tricks for this, but in a generic large dense problem I do not want the algorithm to depend on computing it directly. So I make $L$ adaptive by backtracking: start from a guess $L_0$ and a factor $\eta>1$; at each step inflate $\bar L=\eta^{i}L_{k-1}$ by the smallest $i\ge0$ that makes the sufficient-decrease test $F(p_{\bar L}(y_k))\le Q_{\bar L}(p_{\bar L}(y_k),y_k)$ hold, then set $L_k=\bar L$. The descent lemma guarantees the test passes as soon as $\bar L\ge L(f)$, so under the standard initialization the accepted values satisfy the upper envelope $L_k\le\alpha L(f)$ with $\alpha=1$ in the constant-step case and $\alpha=\eta$ for backtracking. The variable-$L$ proof uses the additional fact that this backtracking sequence is nondecreasing. The only effect on the bound is that $L(f)$ is replaced by $\alpha L(f)$ in the constant: $F(x_k)-F(x^\*)\le 2\alpha L(f)\|x_0-x^\*\|^2/(k+1)^2$, and similarly the ISTA bound gets the factor $\alpha$. The rate is untouched; the price of not knowing $L$ is one extra prox evaluation per backtracking trial.

Let me write the landing form, both algorithms, against the lasso/$\ell_1$ deblurring objective $f(x)=\|Ax-b\|^2$, $g(x)=\lambda\|x\|_1$, $\nabla f(x)=2A^\top(Ax-b)$, $L=2\lambda_{\max}(A^\top A)$, step $t=1/L$, shrinkage threshold $\lambda t=\lambda/L$.

```python
import numpy as np

def soft_threshold(v, thr):
    # prox of  thr * ||.||_1 :  argmin_x thr*||x||_1 + (1/2)||x-v||^2
    # derived coordinatewise as sign(v)*max(|v|-thr, 0)
    return np.sign(v) * np.maximum(np.abs(v) - thr, 0.0)

def grad_f(x, A, b):
    # f(x) = ||A x - b||^2  ->  grad = 2 A^T (A x - b)
    return 2.0 * (A.T @ (A @ x - b))

def ista(A, b, lam, x0, n_iter):
    # gradient step on f, then prox of g (soft-threshold): x_{k+1} = T_{lam t}(x_k - t grad f)
    L = 2.0 * np.linalg.eigvalsh(A.T @ A)[-1]   # Lipschitz const of grad f
    t = 1.0 / L
    x = x0.copy()
    for _ in range(n_iter):
        x = soft_threshold(x - t * grad_f(x, A, b), lam * t)
    return x

def fista(A, b, lam, x0, n_iter):
    # same prox step, but evaluated at an extrapolated anchor y carrying momentum
    L = 2.0 * np.linalg.eigvalsh(A.T @ A)[-1]
    t = 1.0 / L
    x = x0.copy()
    y = x0.copy()
    tk = 1.0                                     # t_1 = 1
    for _ in range(n_iter):
        x_new = soft_threshold(y - t * grad_f(y, A, b), lam * t)   # x_k = prox(y_k - t grad f(y_k))
        tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk**2)) / 2.0          # forced by t_k^2 = t_{k+1}^2 - t_{k+1}
        y = x_new + ((tk - 1.0) / tk_new) * (x_new - x)            # momentum from last two iterates
        x, tk = x_new, tk_new
    return x
```

The causal chain, start to finish: the problem is large, dense, and nonsmooth, so interior-point is out and I am confined to cheap first-order steps where the rate is everything; the subgradient method wastes the smoothness of the data term and crawls at $1/\sqrt{k}$, so I split $F=f+g$ and keep them apart; majorizing only $f$ by its Lipschitz upper quadratic and minimizing that plus the exact $g$ produces a gradient step followed by $\mathrm{prox}_{tg}$ — forward then backward — whose fixed points are exactly the minimizers; for $g=\lambda\|\cdot\|_1$ the prox separates and solves coordinatewise to soft-thresholding, recovering ISTA from first principles at $O(n)$ extra cost; a single engine inequality, telescoped over a first-order potential, pins ISTA at $O(1/k)$; recalling that $1/k^2$ is the true first-order floor, I evaluate the *same* prox step at an extrapolated anchor and demand that the proof telescope over a $t_k^2$-weighted potential — which forces the weight recursion $t_{k+1}=(1+\sqrt{1+4t_k^2})/2$ and the momentum coefficient $(t_k-1)/t_{k+1}$ — yielding $t_k\ge(k+1)/2$ and the accelerated $O(1/k^2)$; and backtracking on $L$ removes the need to know the Lipschitz constant at the cost of an extra prox per trial, leaving the rate intact.
