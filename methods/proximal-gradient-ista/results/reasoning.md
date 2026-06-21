I start with the large sparse inverse problem, because it makes the constraints impossible to ignore. I want to minimize

`F(x)=||A x-b||_2^2 + lambda ||x||_1`.

The least-squares part is smooth and coupled through `A`; the absolute-value part is nonsmooth but separable. If I turn the whole sum into a cone program, a Newton-type method asks for large dense linear solves. If I treat the whole sum as one nonsmooth function, a subgradient method converges too slowly and wastes the smoothness of the quadratic data term. The useful method has to keep the two pieces apart.

So I rewrite the problem as `min_x f(x)+g(x)`, where `f` is convex differentiable with Lipschitz gradient and `g` is convex but may be nonsmooth. I do not try to differentiate `g`. I only use smoothness where I actually have it. If `grad f` is `L`-Lipschitz, then the descent lemma gives

`f(x) <= f(y) + <x-y, grad f(y)> + (L/2)||x-y||_2^2`.

This is a global quadratic upper model for `f` at `y`. The important choice is to model only `f` and leave `g` exact:

`Q_L(x,y)=f(y)+<x-y,grad f(y)>+(L/2)||x-y||_2^2+g(x)`.

The next point is the minimizer of this model,

`p_L(y)=argmin_x Q_L(x,y)`.

This is a majorization step: at `x=y`, the model equals `F(y)`, and for `L >= L(f)` the model dominates `F`. Minimizing it gives descent without ever smoothing or linearizing the nonsmooth term. That is the first key split.

Now I complete the square in the terms that depend on `x`:

`<x-y,grad f(y)>+(L/2)||x-y||_2^2 = (L/2)||x-(y-grad f(y)/L)||_2^2 - (1/(2L))||grad f(y)||_2^2`.

The constant drops out, so

`p_L(y)=argmin_x g(x)+(L/2)||x-(y-grad f(y)/L)||_2^2`.

Equivalently, with `t=1/L`,

`p_L(y)=prox_{t g}(y-t grad f(y))`,

where

`prox_{t g}(v)=argmin_x g(x)+(1/(2t))||x-v||_2^2`.

This formula is the method's core. I first move forward using the gradient of the smooth term, and then I solve the backward proximal subproblem for the nonsmooth term. The shrinkage or projection that appears later is not an added heuristic; it is the exact minimizer of the nonsmooth proximal subproblem after the smooth gradient step has already been taken.

I check that the fixed points are the right objects. A minimizer `x*` satisfies

`0 in grad f(x*) + partial g(x*)`.

This is equivalent to `x* - t grad f(x*) - x* in t partial g(x*)`. The optimality condition for `u=prox_{t g}(v)` is `v-u in t partial g(u)`. With `v=x*-t grad f(x*)` and `u=x*`, the two statements match. Therefore the fixed points of `x -> prox_{t g}(x-t grad f(x))` are exactly the minimizers of `F`.

For the lasso or deblurring case, I now compute the proximal map instead of naming it. Let `g(x)=lambda ||x||_1`, and let `v` be the point after the gradient step. The prox separates into scalar problems:

`min_z lambda |z| + (1/(2t))(z-v_i)^2`.

If `z>0`, the derivative condition gives `z=v_i-lambda t`, valid only when `v_i>lambda t`. If `z<0`, it gives `z=v_i+lambda t`, valid only when `v_i<-lambda t`. At `z=0`, the subgradient condition is `0 in lambda[-1,1] - v_i/t`, which is equivalent to `|v_i| <= lambda t`. The three cases combine as

`T_{lambda t}(v_i)=sign(v_i) max(|v_i|-lambda t,0)`.

So the nonsmooth proximal step is soft shrinkage. For `f(x)=||Ax-b||_2^2`, `grad f(x)=2 A^T(Ax-b)` and `L(f)=2 lambda_max(A^T A)`, giving

`x_{k+1}=T_{lambda/L}(x_k - (1/L) 2 A^T(Ax_k-b))`.

This is the basic iterative shrinkage-thresholding step. Its per-iteration cost is exactly what the large inverse problem allows: apply `A`, apply `A^T`, then shrink coordinates. The thresholded-Landweber surrogate view reaches the same kind of update by canceling the coupled quadratic term and leaving separable shrinkage problems. The forward-backward splitting view gives the fixed-point language for the same structure. The two views agree because the smooth term is advanced explicitly, and the nonsmooth term is solved implicitly.

I still need a rate. The analysis turns on one inequality. Suppose the accepted `L` satisfies `F(p_L(y)) <= Q_L(p_L(y),y)`. From convexity of `f`, convexity of `g`, and the optimality condition for `p_L(y)`, there exists `gamma in partial g(p_L(y))` with

`grad f(y)+L(p_L(y)-y)+gamma=0`.

Using this subgradient in the lower bound for `g(x)` and subtracting the model value gives, for every `x`,

`F(x)-F(p_L(y)) >= (L/2)||p_L(y)-y||_2^2 + L<y-x,p_L(y)-y>`.

This is the engine. For the basic method I take `y=x_n`, `p_L(y)=x_{n+1}`, and `x=x*`. The right side becomes a difference of squared distances, so it telescopes across iterations. The same inequality with `x=y=x_n` gives monotonicity of the objective values. Combining the telescoping distance bound with monotonicity yields

`F(x_k)-F(x*) <= Lmax||x_0-x*||_2^2/(2k)`,

where `Lmax` is a uniform upper bound on the curvatures used. If I fix `L=L(f)`, then `Lmax=L(f)`. With standard backtracking initialized at `L_0 <= L(f)`, minimality of the accepted multiple gives `Lmax=eta L(f)`; if I start above that scale, the same proof uses `Lmax=max(L_0, eta L(f))`. Thus the basic method has the same `O(1/k)` function-value rate as ordinary gradient descent, but it applies to the composite nonsmooth objective because the nonsmooth part is handled by the prox.

That still leaves a gap. Smooth convex minimization can reach `O(1/k^2)` with an auxiliary-point acceleration, so the question is whether the same acceleration can survive after replacing the gradient step by the composite `p_L` step. I keep the same proximal minimization, but I stop anchoring it at the current iterate. Instead, I evaluate it at an extrapolated point:

`x_k=p_L(y_k)`.

The anchor has to be chosen so the proof telescopes. I introduce weights `t_k` and try to control `t_k^2(F(x_k)-F(x*))` rather than only `F(x_k)-F(x*)`. Let

`v_k=F(x_k)-F(x*)`

and

`u_k=t_k x_k-(t_k-1)x_{k-1}-x*`.

Applying the engine inequality twice at the same anchor `y_{k+1}`, once with `x=x_k` and once with `x=x*`, and combining the two inequalities forces the identity

`t_k^2=t_{k+1}^2-t_{k+1}`.

Solving this positive recurrence gives

`t_{k+1}=(1+sqrt(1+4t_k^2))/2`.

The same algebra forces the extrapolated anchor:

`y_{k+1}=x_k+((t_k-1)/t_{k+1})(x_k-x_{k-1})`.

With exactly this choice, the combined inequality becomes

`(2/L_k)t_k^2 v_k - (2/L_{k+1})t_{k+1}^2 v_{k+1} >= ||u_{k+1}||_2^2 - ||u_k||_2^2`.

Now the potential telescopes. The left term controls the weighted function error, while the right term is the change in a squared norm. The base case follows from the same engine inequality at `y_1=x_0`, `t_1=1`. Since the recurrence also gives `t_k >= (k+1)/2`, I get

`F(x_k)-F(x*) <= 2 Lmax||x_0-x*||_2^2/(k+1)^2`.

This is the accelerated version. It uses the same primitives as the basic method: one gradient of the smooth part and one proximal solve for the nonsmooth part. The only change is the anchor `y_k`, and the momentum coefficient is not arbitrary; it is the coefficient that makes the weighted potential proof collapse. The objective itself need not decrease every iteration, because the extrapolated point can overshoot. What decreases is the proof potential, and that is enough for the `O(1/k^2)` guarantee.

If `L(f)` is unknown, I use backtracking. Starting from the previous accepted curvature `L_{k-1}`, I test `bar L=eta^i L_{k-1}` for the smallest integer `i>=0`, compute `p_{bar L}(y)`, and accept it when

`F(p_{bar L}(y)) <= Q_{bar L}(p_{bar L}(y),y)`.

The descent lemma guarantees acceptance once the tested curvature is large enough. Under the standard multiplicative update by `eta>1`, this supplies the same uniform `Lmax`: `L(f)` for the fixed-curvature case, `eta L(f)` for standard backtracking from below the Lipschitz scale, and `max(L_0, eta L(f))` if the initial curvature is already larger. The construction is therefore not merely shrinkage. It is a composite first-order construction: split the objective, take the smooth gradient step, solve the nonsmooth proximal shrinkage exactly, then optionally accelerate by moving the anchor while preserving the same proximal subproblem.
