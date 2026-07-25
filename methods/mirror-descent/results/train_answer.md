I present mirror descent as the canonical first-order method for nonsmooth convex optimization when the problem geometry is not Euclidean. The goal is to minimize a convex function f over a closed convex set X in R^n using only subgradient access. At each queried point x the oracle returns a subgradient g in partial f(x), and we want a point whose objective value is within epsilon of the minimum. The key question is not local quadratic approximation; it is whether each iteration can be cheap and whether the convergence rate can avoid an unnecessary dependence on the ambient dimension. Mirror descent answers this by replacing the Euclidean squared-distance penalty in projected subgradient descent with a Bregman divergence tailored to the feasible set and to the norm in which subgradients are naturally small.

The motivation becomes clear on the probability simplex. Points in the simplex are distributions, and the natural movement of distributions is measured by l1-type quantities and relative entropy. Loss vectors in expert-style problems are often bounded coordinatewise, so their natural bound is in l_infinity, not in l2. If we force such a vector into Euclidean units, its l2 norm can be larger by a factor of sqrt(n). Ordinary projected subgradient descent then pays a dimension penalty that is not intrinsic to the oracle; it is an artifact of measuring a simplex-shaped problem with a Euclidean ruler. The original Nemirovski-Yudin construction showed that the Euclidean labor bound can scale like O(n / epsilon^2), while an l1-adapted construction scales like O(log n / epsilon^2). Mirror descent is the general framework that exposes and removes this mismatch.

The conceptual starting point is the observation that a subgradient is a linear functional, not a primal vector. In a Hilbert space we identify the primal space with its dual, so the formula x_{k+1} = x_k - t_k g_k looks innocent. Outside Hilbert geometry, however, subtracting a dual object directly from a primal point is a type error. The familiar gradient step works only because the Euclidean inner product hides the identification between primal and dual coordinates. Mirror descent makes that identification explicit and replaceable.

In the dual-space formulation we choose a regular convex function V on the dual space E*. The main trajectory is a dual point g(t), and the primal iterate is its image or shadow x(t) = V'(g(t)). The dual variable is updated additively by a subgradient direction, and the primal iterate is recovered through the mirror map V'. Along the idealized trajectory the Lyapunov-like quantity V(g(t)) - <g(t), x*> has derivative controlled by convexity: it decreases whenever the current primal shadow is worse than the optimum. This explains the name: the descent is genuinely happening in the dual space, and the primal motion is what we see after reflecting that dual motion back through V'. In the Hilbert case V(g) = (1/2)||g||_2^2 gives V'(g) = g, so dual and primal points coincide and mirror descent reduces to ordinary gradient descent. Thus the Euclidean method is not a separate idea; it is the special case where the mirror is the identity.

To obtain the modern finite-dimensional update, we replace the dual regular function by a differentiable distance-generating potential psi on the primal geometry. The tangent gap of psi is the Bregman divergence

B_psi(x, y) = psi(x) - psi(y) - <grad psi(y), x - y>.

The mirror descent step keeps the linearized objective but changes the proximal penalty:

x_{k+1} = argmin_{x in X} { t_k <g_k, x> + B_psi(x, x_k) }.

Differentiating the unconstrained version gives the transparent dual-coordinate update

grad psi(x_{k+1}) = grad psi(x_k) - t_k g_k.

Here the type mismatch is gone: we map the primal point into dual coordinates with grad psi, take the additive subgradient step there, and return through the inverse mirror map. With constraints, the optimality condition becomes

0 in t_k g_k + grad psi(x_{k+1}) - grad psi(x_k) + N_X(x_{k+1}),

so the return to the feasible set is a Bregman projection, not a Euclidean projection. The same potential defines both the coordinates in which the gradient acts and the geometry used to come back to feasibility.

The convergence analysis telescopes cleanly. For a minimizer x* and a subgradient g_k in partial f(x_k), convexity gives

t_k (f(x_k) - f(x*)) <= t_k <g_k, x_k - x*>.

Using the optimality condition and the Bregman three-point identity, the right-hand side becomes

B_psi(x*, x_k) - B_psi(x*, x_{k+1}) - B_psi(x_{k+1}, x_k) + t_k <g_k, x_k - x_{k+1}>.

The first two terms telescope over iterations, the third term is controlled by the strong convexity of psi with constant sigma, and the subgradient term is bounded by the dual norm. This yields the one-step estimate

t_k (f(x_k) - f(x*)) <= B_psi(x*, x_k) - B_psi(x*, x_{k+1}) + t_k^2 ||g_k||_*^2 / (2 sigma).

Summing over k iterations and dividing by the total step length gives

min_{s <= k} f(x_s) - f(x*) <= (B_psi(x*, x_1) + (1/(2 sigma)) sum_s t_s^2 ||g_s||_*^2) / sum_s t_s.

This is the payoff: the gradient is measured in the dual norm that matches the chosen primal norm, and the feasible set is measured by a Bregman radius rather than a Euclidean diameter. With constant step size t_s proportional to 1/sqrt(k) and bounded dual subgradient norm L, the best iterate converges at rate O(L sqrt(B_psi(x*, x_1) / sigma) / sqrt(k)).

The Euclidean specialization confirms that nothing is lost. If psi(x) = (1/2)||x||_2^2, then grad psi is the identity, B_psi is squared Euclidean distance, and the update collapses to projected subgradient descent. On the simplex we choose negative entropy psi(x) = sum_i x_i log x_i. Its gradient is 1 + log x, so the dual-coordinate step is additive in logarithmic coordinates:

log y_{k+1, i} = log x_{k, i} - t_k g_{k, i}.

Mapping back gives y_{k+1, i} = x_{k, i} exp(-t_k g_{k, i}), and the entropy projection onto the simplex is simply normalization:

x_{k+1, i} = x_{k, i} exp(-t_k g_{k, i}) / sum_j x_{k, j} exp(-t_k g_{k, j}).

The multiplicative update appears because logarithmic dual coordinates are the right coordinates for distributions. Negative entropy is 1-strongly convex with respect to the l1 norm, the dual norm is l_infinity, and the uniform starting point has entropy radius at most log n. The rate therefore scales like sqrt(2 log n) ||g||_infinity / sqrt(k) instead of the Euclidean sqrt(n) ||g||_infinity / sqrt(k). The distinctive insight is exactly this: we adapt the mirror map to the problem so the gradient acts in dual coordinates, then return through the induced Bregman geometry to the feasible set.

This is the complete method, stated as a protocol rather than a single formula. Fix the feasible set X, a norm ||.|| on the primal space with dual norm ||.||_*, and a differentiable potential psi that is sigma-strongly convex with respect to ||.||. Initialize x_1 in X. At each step k, query a subgradient g_k in partial f(x_k), form the dual-coordinate update

grad psi(x_{k+1}) = grad psi(x_k) - t_k g_k,

and recover the next primal iterate by the Bregman projection that reconciles this dual step with the constraint set,

0 in t_k g_k + grad psi(x_{k+1}) - grad psi(x_k) + N_X(x_{k+1}),

equivalently x_{k+1} = argmin_{x in X} { t_k <g_k, x> + B_psi(x, x_k) }. This protocol carries the guarantee

min_{s <= k} f(x_s) - f(x*) <= (B_psi(x*, x_1) + (1/(2 sigma)) sum_{s=1}^k t_s^2 ||g_s||_*^2) / sum_{s=1}^k t_s,

and with the horizon-optimal step t_s = sqrt(2 sigma B_psi(x*, x_1)) / (L sqrt(k)) for a known bound ||g_s||_* <= L, this collapses to the closed-form rate L sqrt(2 B_psi(x*, x_1) / sigma) / sqrt(k). Choosing psi(x) = (1/2)||x||_2^2 recovers ordinary projected subgradient descent with its Euclidean sqrt(n) dependence. Choosing psi to be negative entropy on the simplex instead makes sigma = 1 in the l1 norm, bounds the Bregman radius by log n, and turns the abstract protocol into the closed-form multiplicative weights update

x_{k+1, i} = x_{k, i} exp(-t_k g_{k, i}) / sum_j x_{k, j} exp(-t_k g_{k, j}),

running at the dimension-free rate sqrt(2 log n) ||g||_infinity / sqrt(k). The mirror map psi is the single design choice that carries all of this: it fixes the dual coordinates in which the subgradient step is additive, the Bregman divergence through which the iterate returns to the feasible set, and the constant sigma that ultimately sets the convergence rate.
