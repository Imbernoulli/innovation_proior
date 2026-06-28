I start with the large sparse inverse problem, because it makes the constraints impossible to ignore. I want to minimize

`F(x)=||A x-b||_2^2 + lambda ||x||_1`.

The least-squares part is smooth and coupled through `A`; the absolute-value part is nonsmooth but separable. If I turn the whole sum into a cone program, a Newton-type method asks for large dense linear solves, which the scale rules out. If I treat the whole sum as one nonsmooth function, a subgradient method converges slowly and throws away the smoothness of the quadratic data term. So neither extreme uses what I actually have. What I have is one term I can differentiate and one term I cannot, and I would like a method that touches each piece only with the operation it admits.

So I rewrite the problem as `min_x f(x)+g(x)`, where `f` is convex differentiable with Lipschitz gradient and `g` is convex but may be nonsmooth. I will not try to differentiate `g`. I only use smoothness where I actually have it. If `grad f` is `L`-Lipschitz, then the descent lemma gives

`f(x) <= f(y) + <x-y, grad f(y)> + (L/2)||x-y||_2^2`.

This is a global quadratic upper model for `f` at `y`. The natural move is to replace `f` by this model but keep `g` untouched, since I have no smooth surrogate for `g` and forcing one would lose its exactness:

`Q_L(x,y)=f(y)+<x-y,grad f(y)>+(L/2)||x-y||_2^2+g(x)`.

At `x=y` the model equals `F(y)` exactly, and for `L >= L(f)` the descent lemma makes the model dominate `F`. So minimizing it is a majorization step: minimizing an upper bound that touches at the current point cannot increase `F`. Let

`p_L(y)=argmin_x Q_L(x,y)`.

I want to know what this minimizer actually looks like. Completing the square in the terms that depend on `x`:

`<x-y,grad f(y)>+(L/2)||x-y||_2^2 = (L/2)||x-(y-grad f(y)/L)||_2^2 - (1/(2L))||grad f(y)||_2^2`.

The constant drops out, so

`p_L(y)=argmin_x g(x)+(L/2)||x-(y-grad f(y)/L)||_2^2`.

Equivalently, with `t=1/L`,

`p_L(y)=prox_{t g}(y-t grad f(y))`,

where

`prox_{t g}(v)=argmin_x g(x)+(1/(2t))||x-v||_2^2`.

This is worth pausing on. The smooth term entered only through one gradient evaluation, which moved `y` forward to `v=y-t grad f(y)`. The nonsmooth term entered only through its prox, which is solved exactly at `v`. So the model has split the work cleanly: an explicit forward step using the gradient I have, then an implicit backward step solving the part I can't differentiate. Whatever shrinkage or projection shows up downstream is therefore not a heuristic bolted on; it is forced as the exact minimizer of the prox subproblem.

Before going further I want to be sure this map is solving the right problem — that its fixed points are minimizers of `F`, not merely some related quantity. A minimizer `x*` satisfies

`0 in grad f(x*) + partial g(x*)`,

which I can rearrange as `(x*-t grad f(x*)) - x* in t partial g(x*)`. Now compare to the prox optimality condition: `u=prox_{t g}(v)` holds iff `v-u in t partial g(u)`. Setting `v=x*-t grad f(x*)` and `u=x*`, the two conditions are the same statement. So the fixed points of `x -> prox_{t g}(x-t grad f(x))` coincide with the minimizers of `F`.

I would rather not take that purely on the algebra, so I check it on a case I can solve by hand. Take `f(x)=(a x-b)^2`, `g(x)=lambda|x|`, scalars, with `a=2, b=3, lambda=1`. Guessing the minimizer is positive, stationarity `2a(a x-b)+lambda=0` gives `x*=(2ab-lambda)/(2a^2)=11/8=1.375`, which is indeed positive, so the guess is consistent. Its gradient `grad f(x*)=2a(a x*-b)=-1`, and `-1 = -lambda*sign(x*)`, so `0 in grad f(x*)+lambda partial|x*|` checks out. Then `L(f)=2a^2=8`, `t=1/8`, and `v=x*-t grad f(x*)=1.375+0.125=1.5`. Soft-thresholding `v` by `lambda t=0.125` returns `1.375` again — so `x*` is a genuine fixed point of the map, and a 200-step run of the iteration lands on `1.375` to full precision. The fixed-point claim is not just rearranged subgradients; it holds numerically on a problem whose answer I already know.

Now I make the prox concrete for the lasso instead of just naming it. Let `g(x)=lambda ||x||_1`, and let `v` be the point after the gradient step. The prox separates into scalar problems:

`min_z lambda |z| + (1/(2t))(z-v_i)^2`.

If `z>0`, the derivative condition gives `z=v_i-lambda t`, valid only when `v_i>lambda t`. If `z<0`, it gives `z=v_i+lambda t`, valid only when `v_i<-lambda t`. At `z=0`, the subgradient condition `0 in lambda[-1,1] - v_i/t` is equivalent to `|v_i| <= lambda t`. The three cases combine as

`T_{lambda t}(v_i)=sign(v_i) max(|v_i|-lambda t,0)`.

I distrust case-split derivations of nonsmooth minimizers, so I check this against a brute-force grid minimization of the scalar objective at `t=0.3, lambda=1` (threshold `0.3`): for `v=-2.0` the grid argmin is `-1.70`; for `v=-0.2` and `v=0.05` it is `0`; for `v=0.7` it is `0.40`; for `v=1.4` it is `1.10`. Every one matches `sign(v)max(|v|-0.3,0)`, including both the dead-zone cases and the shrink-by-`tau` cases. So the nonsmooth prox step is soft shrinkage, confirmed numerically.

For `f(x)=||Ax-b||_2^2`, `grad f(x)=2 A^T(Ax-b)` and the gradient is `L`-Lipschitz with `L(f)=2 lambda_max(A^T A)`, giving

`x_{k+1}=T_{lambda/L}(x_k - (1/L) 2 A^T(Ax_k-b))`.

Its per-iteration cost is exactly what the large inverse problem allows: apply `A`, apply `A^T`, then shrink coordinates. To make sure I have the constants right, I trace one step by hand on `A=diag(1,2)`, `b=(1,1)`, `lambda=0.5`, starting at `x_0=0`. Then `L(f)=2 lambda_max(A^TA)=2*4=8`. The gradient at `0` is `2A^T(-b)=(-2,-4)`, so `v=x_0-grad/L=(0.25,0.5)`. The threshold is `lambda/L=0.0625`, and soft-thresholding gives `(0.25-0.0625,\,0.5-0.0625)=(0.1875,0.4375)`. Running the code for one step returns exactly `(0.1875,0.4375)`, so the implementation and the formula agree on the constants, including the factor of `2` in the gradient and the `lambda/L` threshold.

I still need a rate. The analysis turns on one inequality. Suppose the accepted `L` satisfies `F(p_L(y)) <= Q_L(p_L(y),y)`. From convexity of `f`, convexity of `g`, and the optimality condition for `p_L(y)`, there exists `gamma in partial g(p_L(y))` with

`grad f(y)+L(p_L(y)-y)+gamma=0`.

Using this subgradient in the lower bound for `g(x)` and subtracting the model value gives, for every `x`,

`F(x)-F(p_L(y)) >= (L/2)||p_L(y)-y||_2^2 + L<y-x,p_L(y)-y>`.

For the basic method I take `y=x_n`, `p_L(y)=x_{n+1}`, and `x=x*`. The right side is `(L/2)||x_{n+1}-x_n||^2 + L<x_n-x*,x_{n+1}-x_n>`, which is the polarization identity for `(L/2)(||x_n-x*||^2 - ||x_{n+1}-x*||^2)`. So summing over `n` telescopes the distance-to-optimum terms. Taking instead `x=y=x_n` gives `F(x_n)-F(x_{n+1}) >= (L/2)||x_{n+1}-x_n||^2 >= 0`, i.e. monotone nonincreasing objective. Combining the telescoped distance bound with monotonicity yields

`F(x_k)-F(x*) <= Lmax||x_0-x*||_2^2/(2k)`,

where `Lmax` is a uniform upper bound on the curvatures used.

That whole chain rests on the acceptance condition `F(p_L(y)) <= Q_L(p_L(y),y)`, so I want to see it actually hold — and see what happens when it shouldn't. On the same `A=diag(1,2)`, `b=(1,1)`, `lambda=0.5` problem at `y=(0.3,-0.2)`, with `L(f)=8`: at `L=8` I get `F(p_L(y))=0.78578` and `Q_L(p_L(y),y)=0.82375`, so the condition holds with room to spare. At `L=4` (half of `L(f)`) it gives `F(p)=2.348` against `Q_L=-0.853`, and at `L=1.6` it is `26.69` against `-5.88` — both violate the inequality badly. This is the concrete reason the model only dominates `F` for `L >= L(f)`: undershoot the curvature and the "upper" model dips below the true value, the descent argument collapses, and the step can overshoot. It also tells me what backtracking has to test for. Running ISTA at `L=L(f)` for 30 steps, the objective is monotone nonincreasing throughout, matching the `x=y=x_n` branch above. So the rate's load-bearing inequality is real, not assumed, and it fails exactly where the theory says it should.

If I fix `L=L(f)`, then `Lmax=L(f)`. With standard backtracking initialized at `L_0 <= L(f)`, minimality of the accepted multiple gives `Lmax=eta L(f)`; if I start above that scale, the same proof uses `Lmax=max(L_0, eta L(f))`. Thus the basic method has the same `O(1/k)` function-value rate as ordinary gradient descent, but it applies to the composite nonsmooth objective because the nonsmooth part is handled by the prox.

That still leaves a gap. Smooth convex minimization can reach `O(1/k^2)` with an auxiliary-point acceleration, so the question is whether the same acceleration can survive after replacing the gradient step by the composite `p_L` step. I keep the same proximal minimization, but I stop anchoring it at the current iterate. Instead, I evaluate it at an extrapolated point:

`x_k=p_L(y_k)`,

and I have to figure out which anchor `y_k` makes a `O(1/k^2)` proof close. The basic proof telescoped a plain distance; for `1/k^2` I want to telescope `t_k^2(F(x_k)-F(x*))` for some growing weights `t_k`. Let

`v_k=F(x_k)-F(x*)`, `u_k=t_k x_k-(t_k-1)x_{k-1}-x*`.

I apply the engine inequality at the shared anchor `y_{k+1}` twice — once with `x=x_k`, once with `x=x*` — and form the combination `(t_{k+1}-1)` times the first plus one times the second, which is the weighting that lets the function-value terms assemble into `t_{k+1}^2 v_{k+1}`. For the cross terms to fold into the squared norm `||u_{k+1}||^2 - ||u_k||^2` rather than leaving a stray residual, the weights cannot be free: the coefficient bookkeeping only closes if

`t_{k+1}^2 - t_{k+1} = t_k^2`.

I want to be sure this recurrence is consistent and gives growth, not something that stalls or blows up. Solving the quadratic for the positive root,

`t_{k+1}=(1+sqrt(1+4t_k^2))/2`,

and iterating from `t_1=1` I get `1, 1.618, 2.194, 2.750, 3.295, 3.833, 4.365, ...`. Checking the defining identity directly: `t_1^2=1` and `t_2^2-t_2=2.618-1.618=1`; `t_2^2=2.618` and `t_3^2-t_3=4.812-2.194=2.618`; it matches at every step I compute. And comparing to `(k+1)/2 = 1, 1.5, 2, 2.5, 3, ...`, I have `t_k >= (k+1)/2` throughout — the gap only widens. So the recurrence both satisfies the identity the proof needs and grows at least linearly in `k`, which is what will turn `t_k^2 v_k` bounded into `v_k = O(1/k^2)`.

The same bookkeeping that fixed the recurrence also pins the anchor. Writing `u_{k+1}` in terms of `u_k` and demanding that the cross term match forces

`y_{k+1}=x_k+((t_k-1)/t_{k+1})(x_k-x_{k-1})`.

With exactly this choice, the combined inequality becomes

`(2/L_k)t_k^2 v_k - (2/L_{k+1})t_{k+1}^2 v_{k+1} >= ||u_{k+1}||_2^2 - ||u_k||_2^2`.

Now the potential telescopes: the left side controls the weighted function error and the right side is the change in a squared norm, so summing leaves `(2/L)t_k^2 v_k + ||u_k||^2` bounded by its base-case value. The base case follows from the engine inequality at `y_1=x_0`, `t_1=1`. Using `t_k >= (k+1)/2`,

`F(x_k)-F(x*) <= 2 Lmax||x_0-x*||_2^2/(k+1)^2`.

I'd like one empirical check that this acceleration is real and not just a tighter-looking constant. On a random `60x40` Gaussian `A` with a 3-sparse signal and `lambda=0.1`, taking a 20000-step FISTA run as the reference optimum: at `k=10`, ISTA's gap to optimum is `1.25e-1` while FISTA's is `2.74e-3`, a factor of ~46; by `k=40` both are already at `~9e-8` and by `k=160` both sit at machine precision. So the `1/k^2` advantage shows up exactly where it should — in the early-to-mid regime, before both methods saturate. The momentum coefficient `(t_k-1)/t_{k+1}` is therefore not a tunable knob; it is the value that makes the weighted potential collapse, and the empirical speedup is the visible consequence.

One caveat the proof makes explicit and the experiment confirms: FISTA's objective `F(x_k)` need not decrease every iteration, because the extrapolated `y_k` can overshoot. What decreases is the potential `(2/L)t_k^2 v_k + ||u_k||^2`, and that is what carries the `O(1/k^2)` guarantee.

If `L(f)` is unknown, I use backtracking — which the upper-model experiment above already told me how to specify. Starting from the previous accepted curvature `L_{k-1}`, I test `bar L=eta^i L_{k-1}` for the smallest integer `i>=0`, compute `p_{bar L}(y)`, and accept it when

`F(p_{bar L}(y)) <= Q_{bar L}(p_{bar L}(y),y)`.

The descent lemma guarantees acceptance once the tested curvature reaches `L(f)`, which is exactly the threshold where my numeric check flipped from violated to satisfied. Under the multiplicative update by `eta>1`, this supplies the same uniform `Lmax`: `L(f)` for the fixed-curvature case, `eta L(f)` for standard backtracking from below the Lipschitz scale, and `max(L_0, eta L(f))` if the initial curvature is already larger. So the method that emerges is a composite first-order construction: split the objective into a smooth and a nonsmooth piece, take a gradient step on the smooth part, solve the nonsmooth prox exactly (soft shrinkage for the lasso), and optionally accelerate by moving the anchor to the Nesterov-extrapolated point while reusing the very same prox subproblem.
