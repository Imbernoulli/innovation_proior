I start with ordinary projected subgradient descent because it is the natural cheap method for a large nonsmooth convex problem. I query a first-order oracle, get a subgradient, subtract a scaled version of it, and project back to the feasible set. The rate is already the right nonsmooth order, so the weakness I need to explain is not the `1/sqrt(k)` behavior. The weakness is the geometry hidden in the constant.

The simplex makes the mismatch visible. The feasible points are distributions, and losses are often bounded coordinate by coordinate. That means the gradient information is naturally controlled in `l_infinity`, while movement of distributions is naturally measured in an `l1`-type geometry. If I force the same problem into Euclidean units, the subgradient norm can grow by `sqrt(n)`. Nemirovski and Yudin's original example says the Euclidean labor bound has the scale `O(n / e^2)`, while an `l1`-adapted construction has the scale `O(log n / e^2)`. I am not seeing a harder oracle problem. I am seeing the cost of choosing the wrong ruler.

I now ask what ordinary gradient descent is really doing. In a Hilbert space I can identify the primal space with its dual, so the expression `x_k - t_k g_k` looks innocent. But a subgradient is a linear functional. Outside Euclidean or Hilbert geometry, subtracting a dual object directly from a primal point is the wrong type of operation. The familiar formula works only because the Hilbert-space identification hides the map between primal and dual coordinates.

The primary-source construction makes that hidden map explicit. I choose a regular function `V` on the dual space `E*`. The main trajectory is `g(t)` in `E*`, and the primal point is only its image or shadow, `x(t)=V'(g(t))`. The descent argument tracks the Lyapunov-like quantity `V(g)-<g,x*>`. Along the idealized trajectory its derivative is controlled by convexity:

`d/dt (V(g(t))-<g(t),x*>) = <f'(x(t)),x*-x(t)> <= f(x*)-f(x(t))`.

So the quantity decreases whenever the current primal shadow is worse than the optimum. This explains the name. The descent is genuinely happening in the dual space, and the primal motion is what I see after reflecting that dual motion back through `V'`.

The Hilbert case checks the interpretation. If I take `V(g)=(1/2)||g||_2^2`, then `V'(g)=g`, so the dual point and the primal point are identified and the scheme becomes the usual gradient method. The Euclidean method is therefore not a separate idea. It is the special case where the mirror is the identity.

To turn this source-level picture into the modern finite-dimensional update, I replace the dual regular function by a distance-generating potential `psi` on the primal geometry. Its tangent gap is the Bregman divergence

`B_psi(x,y)=psi(x)-psi(y)-<grad psi(y),x-y>`.

The candidate step keeps the first-order linearization but changes the penalty:

`x_{k+1}=argmin_{x in X} { t_k <g_k,x> + B_psi(x,x_k) }`.

This is the projected subgradient step with the Euclidean squared distance removed and the problem-adapted geometry inserted. It is not an arbitrary distance replacement. The tangent-gap form gives the algebra that the proof needs.

When I differentiate the unconstrained version, the dual motion reappears:

`grad psi(x_{k+1}) = grad psi(x_k) - t_k g_k`.

Now the type mismatch is gone. I map the primal point into dual coordinates with `grad psi`, take the additive subgradient step there, and return through the inverse mirror map. With constraints, the normal cone enters:

`0 in t_k g_k + grad psi(x_{k+1}) - grad psi(x_k) + N_X(x_{k+1})`.

So the return to the feasible set is a Bregman projection, not a Euclidean projection. The same potential defines both the coordinates in which the gradient acts and the geometry used to come back to feasibility.

I then check that the proof still telescopes. For an optimum `x*`, convexity gives

`t_k(f(x_k)-f(x*)) <= t_k <g_k,x_k-x*>`.

The optimality condition lets me move the comparator term into differences of `grad psi`, and the Bregman three-point identity turns those differences into

`B_psi(x*,x_k) - B_psi(x*,x_{k+1}) - B_psi(x_{k+1},x_k)`.

The first two terms telescope over time. The last term is controlled by strong convexity, and the remaining subgradient term is bounded with the dual norm. The one-step estimate becomes

`t_k(f(x_k)-f(x*)) <= B_psi(x*,x_k)-B_psi(x*,x_{k+1}) + t_k^2 ||g_k||_*^2/(2 sigma)`.

After summing, I get

`min_{s<=k} f(x_s)-f(x*) <= (B_psi(x*,x_1) + (1/(2 sigma)) sum_s t_s^2 ||g_s||_*^2) / sum_s t_s`.

This is the payoff of the mirror construction. The gradient is measured in the dual norm that matches the chosen primal norm, and the feasible set is measured by a Bregman radius rather than by a Euclidean diameter. The method still uses only first-order information, but its constants now reflect the geometry of the problem instead of the default geometry of the page.

The Euclidean specialization confirms that nothing has been lost. If `psi(x)=(1/2)||x||_2^2`, then `grad psi` is the identity, `B_psi` is squared Euclidean distance, and the update collapses to projected subgradient descent.

On the simplex I choose negative entropy, `psi(x)=sum_i x_i log x_i`. Its gradient is `1+log x`, so the dual-coordinate step is additive in logarithmic coordinates:

`log y_{k+1,i}=log x_{k,i}-t_k g_{k,i}`.

Mapping back gives `y_{k+1,i}=x_{k,i} exp(-t_k g_{k,i})`, and the entropy projection onto the simplex is normalization:

`x_{k+1,i}=x_{k,i} exp(-t_k g_{k,i}) / sum_j x_{k,j} exp(-t_k g_{k,j})`.

The multiplicative update appears because logarithmic dual coordinates are the right coordinates for distributions. Negative entropy is strongly convex in `l1`, the dual gradient norm is `l_infinity`, and the uniform starting point has entropy radius at most `log n`. The rate therefore scales like `sqrt(log n)||g||_infinity/sqrt(k)` instead of the Euclidean `sqrt(n)||g||_infinity/sqrt(k)`. The distinctive insight is exactly this: I adapt the mirror map to the problem so the gradient acts in dual coordinates, then I return through the induced Bregman geometry to the feasible set.
