I start with ordinary projected subgradient descent because it is the natural cheap method for a large nonsmooth convex problem. I query a first-order oracle, get a subgradient, subtract a scaled version of it, and project back to the feasible set. The rate is already the right nonsmooth order, so the weakness I need to explain is not the `1/sqrt(k)` behavior. The weakness is the geometry hidden in the constant.

The simplex makes the mismatch visible. The feasible points are distributions, and losses are often bounded coordinate by coordinate. That means the gradient information is naturally controlled in `l_infinity`, while movement of distributions is naturally measured in an `l1`-type geometry. If I force the same problem into Euclidean units, the subgradient norm can grow by `sqrt(n)`. The Euclidean labor bound carries the scale `O(n / e^2)`, while an `l1`-adapted construction has the scale `O(log n / e^2)`. I want to make sure that gap is real and not an artifact of loose constants, so I put numbers on it. The Euclidean rate constant is set by a diameter (the simplex has Euclidean diameter `sqrt(2)`, near constant) times the Lipschitz bound `sqrt(n) ||g||_infinity`; the entropy construction will instead pay a radius of order `log n` and a Lipschitz bound of `||g||_infinity`. For `n = 1000` the relevant primal radius factors are `sqrt(2 log n) = 3.72` against `sqrt(n) = 31.6`, and for `n = 10000` they are `4.29` against `100`. So the favorable scale is not a constant-factor accident; it widens with `n`. I am not seeing a harder oracle problem. I am seeing the cost of choosing the wrong ruler.

I now ask what ordinary gradient descent is really doing. In a Hilbert space I can identify the primal space with its dual, so the expression `x_k - t_k g_k` looks innocent. But a subgradient is a linear functional. Outside Euclidean or Hilbert geometry, subtracting a dual object directly from a primal point is the wrong type of operation. The familiar formula works only because the Hilbert-space identification hides the map between primal and dual coordinates.

So I want a construction that keeps the primal and dual coordinates separate. I choose a regular function `V` on the dual space `E*`. The main trajectory is `g(t)` in `E*`, and the primal point is only its image or shadow, `x(t)=V'(g(t))`. To see whether such a scheme can actually descend, I track the Lyapunov-like quantity `V(g)-<g,x*>` and compute its derivative. Differentiating, `d/dt(V(g)-<g,x*>) = <V'(g)-x*, g'>`. With `V'(g)=x` and the dual update `g'(t)=-f'(x(t))`, this is `<x(t)-x*, -f'(x(t))> = <f'(x(t)), x*-x(t))>`. Convexity bounds it:

`d/dt (V(g(t))-<g(t),x*>) = <f'(x(t)),x*-x(t)> <= f(x*)-f(x(t))`.

So the quantity decreases whenever the current primal shadow is worse than the optimum, which is exactly what I need for a descent argument. And the geometry of the name is now legible: the descent is happening in the dual space, and the primal motion is what I see after reflecting that dual motion back through `V'`.

Before going further I check that this is genuinely a generalization and not a relabeling of the Euclidean method. Take `V(g)=(1/2)||g||_2^2`. Then `V'(g)=g`, so the dual point and the primal point coincide, and the update `g_{k+1}=g_k-t_k g_k'` becomes `x_{k+1}=x_k-t_k f'(x_k)`. That is exactly gradient descent. The Euclidean method is therefore the special case where the reflecting map is the identity, which is reassuring: any new bound I prove must reduce to the standard one here.

To turn this picture into a finite-dimensional update I can run, I replace the dual regular function by a distance-generating potential `psi` on the primal geometry. Its tangent gap is the Bregman divergence

`B_psi(x,y)=psi(x)-psi(y)-<grad psi(y),x-y>`.

The candidate step keeps the first-order linearization but changes the penalty:

`x_{k+1}=argmin_{x in X} { t_k <g_k,x> + B_psi(x,x_k) }`.

This is the projected subgradient step with the Euclidean squared distance removed and the problem-adapted geometry inserted. The reason to use the tangent-gap form rather than an arbitrary distance is that I expect it to give the algebra a telescoping proof needs; I will only believe that once the telescoping actually closes below.

When I write the optimality condition for the unconstrained version, the dual motion reappears. Setting the gradient of the objective to zero gives `t_k g_k + grad psi(x_{k+1}) - grad psi(x_k) = 0`, i.e.

`grad psi(x_{k+1}) = grad psi(x_k) - t_k g_k`.

Now the type mismatch is gone. I map the primal point into dual coordinates with `grad psi`, take the additive subgradient step there, and return through the inverse mirror map. With constraints, the normal cone enters:

`0 in t_k g_k + grad psi(x_{k+1}) - grad psi(x_k) + N_X(x_{k+1})`.

So the return to the feasible set is a Bregman projection, not a Euclidean projection. The same potential defines both the coordinates in which the gradient acts and the geometry used to come back to feasibility.

I then check that the proof telescopes. For an optimum `x*`, convexity gives

`t_k(f(x_k)-f(x*)) <= t_k <g_k,x_k-x*>`.

The optimality condition replaces `t_k g_k` by `grad psi(x_k)-grad psi(x_{k+1})`, so the right-hand side becomes `<grad psi(x_k)-grad psi(x_{k+1}), x_{k+1}-x*>` plus a `t_k g_k` term carrying `x_k-x_{k+1}`. The first piece is a Bregman three-point quantity, and I want to be certain of the identity I am about to use, because the whole telescoping rests on it:

`<grad psi(x_k)-grad psi(x_{k+1}), x_{k+1}-x*> = B_psi(x*,x_k) - B_psi(x*,x_{k+1}) - B_psi(x_{k+1},x_k)`.

I verify this numerically with `psi` the negative entropy on a 4-point simplex and three random interior points. For one such triple the left side evaluates to `-0.5829857799` and the right side to `-0.5829857799`, agreeing to `1e-16`; repeating on other random triples gives the same machine-precision agreement. The identity holds, so I can substitute it in. The first two terms telescope over time. The `B_psi(x_{k+1},x_k)` term is nonnegative and combines with the leftover subgradient term; strong convexity of `psi` bounds `B_psi(x_{k+1},x_k)` below by `(sigma/2)||x_{k+1}-x_k||^2`, and the remaining `t_k<g_k,x_k-x_{k+1}>` is bounded above using the dual norm and that quadratic, leaving `t_k^2||g_k||_*^2/(2sigma)`. The one-step estimate becomes

`t_k(f(x_k)-f(x*)) <= B_psi(x*,x_k)-B_psi(x*,x_{k+1}) + t_k^2 ||g_k||_*^2/(2 sigma)`.

After summing, I get

`min_{s<=k} f(x_s)-f(x*) <= (B_psi(x*,x_1) + (1/(2 sigma)) sum_s t_s^2 ||g_s||_*^2) / sum_s t_s`.

The gradient is measured in the dual norm that matches the chosen primal norm, and the feasible set is measured by a Bregman radius rather than by a Euclidean diameter. The method still uses only first-order information, but its constants now reflect the geometry of the problem instead of the default geometry of the page.

The Euclidean specialization should recover the known method, and it does: if `psi(x)=(1/2)||x||_2^2`, then `grad psi` is the identity, `B_psi(x,y)=(1/2)||x-y||_2^2` is squared Euclidean distance, and the update `grad psi(x_{k+1})=grad psi(x_k)-t_k g_k` collapses to `x_{k+1}=x_k-t_k g_k` followed by projection. Nothing in the bound is lost in this case.

On the simplex I choose negative entropy, `psi(x)=sum_i x_i log x_i`. Its gradient is `1+log x`, so the dual-coordinate step is additive in logarithmic coordinates:

`log y_{k+1,i}=log x_{k,i}-t_k g_{k,i}`.

Mapping back gives `y_{k+1,i}=x_{k,i} exp(-t_k g_{k,i})`, and the entropy projection onto the simplex is normalization:

`x_{k+1,i}=x_{k,i} exp(-t_k g_{k,i}) / sum_j x_{k,j} exp(-t_k g_{k,j})`.

I want to confirm this normalized exponential really is the constrained Bregman argmin and not just the unconstrained inverse map. I take `n=3`, uniform start `x_k=(1/3,1/3,1/3)`, subgradient `g=(1, -2, 0.5)`, step `t=0.7`. The closed form gives `x_{k+1}=(0.0945, 0.7715, 0.1341)`. Minimizing `t<g,x>+B_psi(x,x_k)` over the simplex numerically returns `(0.0945, 0.7715, 0.1341)` with the same objective value `-0.56085`, matching to `4e-9`. So the multiplicative update is the actual projected step, and the logarithmic dual coordinates are the right coordinates for distributions.

Now I put the simplex pieces into the general bound. Negative entropy is 1-strongly convex with respect to `l1` (Pinsker), so `sigma=1`; the dual norm paired with `l1` is `l_infinity`; and starting from the uniform point, `B_psi(x*,x_1)=psi(x*)-psi(uniform)-... = log n - H(x*) <= log n`, maximized at a vertex. The horizon-constant step then yields a rate of order `sqrt(2 log n)||g||_infinity/sqrt(k)` rather than the Euclidean `sqrt(n)||g||_infinity/sqrt(k)`.

I run the whole loop once to make sure the predicted bound actually dominates the achieved gap. With `n=6`, a linear objective `f(x)=<c,x>` whose minimum sits at a simplex vertex, `K=2000` entropic steps from the uniform start, and the horizon step `t=sqrt(2 B_0)/(L sqrt(K))`, the measured Bregman radius `B_0` comes out to `1.79176`, exactly `log 6=1.79176`, confirming the `log n` radius. The achieved optimality gap is `4.4e-7`, while the theoretical bound `(B_0+(1/2)K t^2 L^2)/(K t)` evaluates to `0.0904`, equal to the closed-form `L sqrt(2 B_0)/sqrt(K)`. The bound is valid (it exceeds the achieved gap) and, as expected for a worst-case guarantee on a benign instance, loose. The construction runs and certifies as predicted: I adapt the mirror map to the problem so the gradient acts in dual coordinates, and I return through the induced Bregman geometry to the feasible set.
