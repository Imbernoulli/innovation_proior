I start with the constrained first-order problem and immediately run into the feasibility wall. If I sit at a feasible point x and take the honest gradient step x - eta grad f(x), I get a point that usually has no reason to remain in D. The usual repair is to project it back, but that repair is not a small algebraic afterthought. It is the optimization problem argmin over D of a squared distance. On the domains I care about, that can be a full singular-value decomposition, a quadratic program over a complicated polytope, or a hard structured convex problem. So the first thing I need to stop doing is stepping out and then paying to clean up the damage.

I ask what operation on D is actually cheap. Projection is quadratic in the candidate point, but a linear objective over D is often much simpler. On a simplex it just picks the smallest coordinate. On an atomic hull it chooses one atom. On a nuclear-norm ball it asks for a leading singular-vector pair instead of a full projection. On an assignment polytope it is a linear assignment problem. So I should try to make the gradient information generate a linear problem over the original feasible set, not a projection problem.

Convexity gives exactly the bridge. At the current x, the tangent model f(x) + <y - x, grad f(x)> lies below f everywhere. If I minimize that tangent model over D, the constants disappear and I am just solving

```text
s = argmin_{v in D} <v, grad f(x)>.
```

This is not yet a full method. The point s is the feasible point where the linear model looks best, but f is not actually linear. If I jump all the way to s, I may trust the tangent plane far outside the region where it is accurate. That is the overshoot wall. The linear model tells me a direction inside the feasible set; it does not license an uncontrolled jump across the whole domain.

The fix is to move toward s but not necessarily all the way. I take

```text
x_next = (1 - gamma) x + gamma s,  gamma in [0,1].
```

Now the feasibility issue disappears for the right reason. Both x and s are in D, and D is convex, so the entire segment between them is in D. I never take the raw gradient step outside the set, and I never project back. I replace the hard projection by a linear minimization over D, then I convex-combine toward the atom returned by that linear problem. That is the real mechanism; the name of the method is secondary.

The same linear minimization also gives a certificate. Since s minimizes <v, grad f(x)> over D,

```text
g(x) = <x - s, grad f(x)> = max_{v in D} <x - v, grad f(x)>.
```

This quantity is nonnegative because v = x is feasible. More importantly, it upper-bounds the true primal error. Convexity at the optimum x* gives

```text
f(x*) >= f(x) + <x* - x, grad f(x)>.
```

Rearranging gives

```text
f(x) - f(x*) <= <x - x*, grad f(x)>.
```

The optimum x* is one feasible point among all v in D, so

```text
<x - x*, grad f(x)> <= max_{v in D} <x - v, grad f(x)> = g(x).
```

Thus g(x) proves how far I might still be from optimality, even though I do not know f(x*). The stopping criterion is not an add-on; it is a by-product of the same linear oracle call that chose the direction.

Now I need a descent proof, because replacing projection by a segment step only matters if the objective really improves. I measure how much f can rise above its tangent model along feasible chords. The curvature constant is

```text
C_f = sup (2/gamma^2) [f(y) - f(x) - <y - x, grad f(x)>],
```

where x and s are in D, gamma is in (0,1], and y = x + gamma(s - x). This is the exact second-order error term the update creates. If f has an L-Lipschitz gradient, then the usual smoothness inequality gives C_f <= L diam(D)^2, but the curvature form is cleaner because it is tied to D and to the segment update itself.

For the actual next point x_next = x + gamma(s - x), the definition of C_f gives

```text
f(x_next) <= f(x) + gamma <s - x, grad f(x)> + (gamma^2/2) C_f.
```

The inner product is -g(x), so

```text
f(x_next) <= f(x) - gamma g(x) + (gamma^2/2) C_f.
```

This is the whole tradeoff in one line. The first-order term wants gamma large because the atom s points in the best linearized feasible direction. The curvature term wants gamma small because the tangent model gets worse along a long chord. Since g(x) >= f(x) - f(x*), with h(x) = f(x) - f(x*) I get

```text
h(x_next) <= (1 - gamma) h(x) + (gamma^2/2) C_f.
```

I want a step-size rule that does not know C_f or f(x*). The harmonic rule gamma_k = 2/(k+2) has the right shape: it starts with gamma_0 = 1, then shrinks slowly enough that total movement continues but quickly enough that the quadratic curvature penalty fades. The constants also fit the recursion. If I write C = C_f/2, the recursion is h_{k+1} <= (1 - gamma_k) h_k + gamma_k^2 C. The induction target is h_k <= 4C/(k+2), which is the same as 2C_f/(k+2). At k = 0, the first step has gamma_0 = 1, so h_1 <= C <= 4C/3. For k >= 1, using gamma_k = 2/(k+2) and h_k <= 4C/(k+2),

```text
h_{k+1} <= (1 - 2/(k+2)) 4C/(k+2) + 4C/(k+2)^2
          = 4C(k+1)/(k+2)^2
          <= 4C/(k+3),
```

because (k+1)(k+3) <= (k+2)^2. So the primal error falls like 2C_f/(k+2).

The certificate also has to become small, not just the primal error. I cannot demand every gap be small at every iteration, because the gap can oscillate. But it cannot stay large over a long late block. Rearranging the descent inequality gives gamma_k g(x_k) <= h_k - h_{k+1} + (gamma_k^2/2)C_f. If the gaps over a late block all stayed above a threshold proportional to C_f/(K+2), summing these inequalities would telescope the h terms and force the nonnegative error below zero. A late-block summation makes this precise: for K >= 2, some iterate k_hat in {1,...,K} has

```text
g(x_{k_hat}) <= 2 beta C_f/(K+2),  beta = 27/8.
```

So the same quantity I compute for stopping is guaranteed to certify progress at the same O(1/K) scale.

The update also explains the sparse and low-rank behavior. Each iteration introduces one new feasible atom s and forms a convex combination with the old point. If I start at an atom, then after k iterations the point is a convex combination of at most k+1 atoms. On the simplex that means a sparse vector. On a nuclear-norm ball it means a sum of a few rank-one matrices. This is not a decoration on top of the convergence proof; it is a direct consequence of using a linear minimization oracle and a convex-combination step instead of a projection.

The candidate method is therefore forced by the bottleneck. I do not repair infeasible gradient steps. I use the gradient only to build a linear objective, minimize that objective over the feasible set, and move partway toward the returned atom. Convexity of D gives feasibility. Convexity of f gives the gap certificate. Curvature gives the one-step descent inequality and the O(C_f/k) rate. The whole construction is projection-free because it never asks for the nearest feasible point in the first place.
