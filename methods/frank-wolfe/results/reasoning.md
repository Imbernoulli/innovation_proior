I start with the constrained first-order problem and immediately run into the feasibility wall. If I sit at a feasible point x and take the honest gradient step x - eta grad f(x), I get a point that usually has no reason to remain in D. The usual repair is to project it back, but that repair is not a small algebraic afterthought. It is the optimization problem argmin over D of a squared distance. On the domains I care about, that can be a full singular-value decomposition, a quadratic program over a complicated polytope, or a hard structured convex problem. So the first thing I want to stop doing is stepping out and then paying to clean up the damage.

I ask what operation on D is actually cheap. Projection is quadratic in the candidate point, but a linear objective over D is often much simpler. On a simplex it just picks the smallest coordinate. On an atomic hull it chooses one atom. On a nuclear-norm ball it asks for a leading singular-vector pair instead of a full projection. On an assignment polytope it is a linear assignment problem. So I want to see whether the gradient information can be made to generate a linear problem over the original feasible set, rather than a projection problem.

Convexity gives a bridge worth trying. At the current x, the tangent model f(x) + <y - x, grad f(x)> lies below f everywhere. If I minimize that tangent model over D, the constants disappear and I am just solving

```text
s = argmin_{v in D} <v, grad f(x)>.
```

That is a linear minimization over D, which is exactly the cheap primitive I was hoping to use. So the gradient buys me a feasible point s without ever asking for a nearest point.

This is not yet a usable iteration. The point s is the feasible point where the linear model looks best, but f is not actually linear. My first instinct is to just set x_next = s, since s is the linearly-best feasible point and it is certainly in D. Let me see what that does on a concrete case before trusting it. Take f(x) = (1/2)||x - b||^2 on the probability simplex in R^3 with b = (0.5, 0.3, -0.4), starting at the vertex x = (1,0,0). The gradient is g = x - b = (0.5, -0.3, 0.4), so its smallest coordinate is index 1 and s = (0,1,0). Jumping to s gives x_next = (0,1,0), with f = (1/2)||(0,1,0)-(0.5,0.3,-0.4)||^2 = (1/2)(0.25 + 0.49 + 0.16) = 0.45. But the starting value was f(1,0,0) = (1/2)(0.25 + 0.09 + 0.16) = 0.25. The objective went up, from 0.25 to 0.45. So jumping all the way to s is not even a descent step. The linear model pointed in a sensible direction, but it licensed an uncontrolled jump across the whole domain, far outside the region where the tangent plane is accurate. That rules out "move to s".

The repair is to move toward s but not all the way:

```text
x_next = (1 - gamma) x + gamma s,  gamma in [0,1].
```

Both x and s are in D, and D is convex, so the entire segment between them is in D for every gamma in [0,1]. I never take the raw gradient step outside the set, and I never project. The hard projection is replaced by a linear minimization over D followed by a convex combination toward the returned atom. Feasibility now holds for a structural reason, not by repair.

The same linear minimization also seems to hand me a certificate, which I want to pin down rather than assume. Since s minimizes <v, grad f(x)> over D,

```text
g(x) = <x - s, grad f(x)> = max_{v in D} <x - v, grad f(x)>.
```

This is nonnegative because v = x is feasible, so the max is at least <x - x, grad f(x)> = 0. The question is whether it bounds the true primal error. Convexity at the optimum x* gives

```text
f(x*) >= f(x) + <x* - x, grad f(x)>,
```

and rearranging,

```text
f(x) - f(x*) <= <x - x*, grad f(x)>.
```

The optimum x* is one feasible point among all v in D, so it cannot beat the maximizer:

```text
<x - x*, grad f(x)> <= max_{v in D} <x - v, grad f(x)> = g(x).
```

So g(x) >= f(x) - f(x*). On the example above, at x = (1,0,0) the gap is g = <(1,0,0)-(0,1,0), (0.5,-0.3,0.4)> = <(1,-1,0),(0.5,-0.3,0.4)> = 0.8. The true optimum is the simplex projection of b, which is x* = (0.6, 0.4, 0) with f* = 0.09, so the actual primal error is 0.25 - 0.09 = 0.16. And indeed 0.16 <= 0.8, with comfortable slack. So the gap I get for free from the same oracle call is a genuine upper bound on how far I am from optimal, computable without knowing f*.

Now I need a descent statement, because replacing projection by a segment step only matters if the objective really improves with the right gamma. I measure how much f can rise above its tangent model along feasible chords. Define

```text
C_f = sup (2/gamma^2) [f(y) - f(x) - <y - x, grad f(x)>],
```

over x, s in D, gamma in (0,1], and y = x + gamma(s - x). This is the exact second-order error the segment update creates. For the quadratic f = (1/2)||y - b||^2 it is concrete: f(y) - f(x) - <y - x, y - b> = (1/2)||y - x||^2, and with y - x = gamma(s - x) this is (1/2)gamma^2||s - x||^2, so (2/gamma^2) times it is just ||s - x||^2. The supremum over the simplex is attained between two vertices, ||e_i - e_j||^2 = 2, so C_f = 2 here. The Lipschitz bound C_f <= L diam(D)^2 with L = 1 and diam(D)^2 = 2 gives the same number 2, so for this problem the bound is tight. Good, the curvature form and the smoothness bound agree where I can check both.

By the definition of C_f, the actual next point x_next = x + gamma(s - x) satisfies

```text
f(x_next) <= f(x) + gamma <s - x, grad f(x)> + (gamma^2/2) C_f.
```

The inner product is -g(x), so

```text
f(x_next) <= f(x) - gamma g(x) + (gamma^2/2) C_f.
```

Let me confirm this is the inequality that broke when I jumped to s, and that it now selects a sane step. At x = (1,0,0) with g(x) = 0.8 and C_f = 2: the full jump gamma = 1 gives f(x_next) <= 0.25 - 0.8 + 1.0 = 0.45, which matches the 0.45 I computed directly — the curvature term exactly cancels the descent there, which is why the full jump failed. A short step does better: the bound -gamma(0.8) + (gamma^2/2)(2) = -0.8 gamma + gamma^2 is minimized at gamma = 0.4, giving 0.25 - 0.16 = 0.09. So the inequality is doing its job; it penalizes the long chord and rewards a moderate step.

This is the whole tradeoff in one line. The first-order term wants gamma large because the atom s points in the best linearized feasible direction. The curvature term wants gamma small because the tangent model gets worse along a long chord. Since g(x) >= f(x) - f(x*), writing h(x) = f(x) - f(x*),

```text
h(x_next) <= (1 - gamma) h(x) + (gamma^2/2) C_f.
```

I want a step-size rule that does not know C_f or f(x*). A rule of the form gamma_k = 2/(k+2) has the right shape: it starts with gamma_0 = 1, then shrinks slowly enough that total movement continues but quickly enough that the quadratic curvature penalty fades. I check that the constants actually fit the recursion. Writing C = C_f/2, the recursion is h_{k+1} <= (1 - gamma_k) h_k + gamma_k^2 C. Try the induction target h_k <= 4C/(k+2), i.e. 2C_f/(k+2). Base case: at k = 0, gamma_0 = 1, so h_1 <= 0 + C = C, and the target at k = 1 is 4C/3, and C <= 4C/3 holds. Inductive step for k >= 1, using gamma_k = 2/(k+2) and h_k <= 4C/(k+2):

```text
h_{k+1} <= (1 - 2/(k+2)) 4C/(k+2) + 4C/(k+2)^2
          = 4C(k+1)/(k+2)^2
          <= 4C/(k+3),
```

where the last step needs (k+1)(k+3) <= (k+2)^2, i.e. k^2 + 4k + 3 <= k^2 + 4k + 4, which is 3 <= 4 — true for every k. So the primal error falls like 2C_f/(k+2).

I want to see this survive contact with the actual iteration, not just the bound, because the per-step inequality used the worst-case curvature and might be loose. Running the segment update with gamma_k = 2/(k+2) from x_0 = (1,0,0) on the same problem:

```text
k    x_k                       f(x_k)    gap     f-f*      2C_f/(k+2)
0    (1, 0, 0)                 0.25000   0.800   0.16000   2.00000
1    (0, 1, 0)                 0.45000   1.200   0.36000   1.33333
2    (0.6667, 0.3333, 0)       0.09444   0.089   0.00444   1.00000
3    (0.3333, 0.6667, 0)       0.16111   0.356   0.07111   0.80000
4    (0.6, 0.4, 0)             0.09000   0.000   0.00000   0.66667
```

The primal error stays under the 2C_f/(k+2) bound at every row, and at k = 4 the iterate lands exactly on x* = (0.6, 0.4, 0) with gap 0. Two things stand out. First, k = 1 actually went up to 0.45 — the gamma_0 = 1 step is the full jump to a vertex, which can be uphill — yet the bound 4C/3 = 1.333 still holds because the bound only promises 2C_f/(k+2), not monotone descent. Second, the gap column is not monotone at all: 0.8, 1.2, 0.089, 0.356, 0.0. It oscillates, even spiking after a good step.

That oscillation tells me the certificate needs separate handling. I cannot promise the gap is small at every iteration, because the run just showed it jumping back up to 0.356 right after hitting 0.089. The honest claim is weaker: the gap cannot stay large across a long late block. Rearranging the descent inequality, gamma_k g(x_k) <= h_k - h_{k+1} + (gamma_k^2/2) C_f. If every gap over a late block of iterations stayed above some threshold, summing these inequalities telescopes the h_k - h_{k+1} terms, and since the h's are nonnegative the left side cannot be too large; that caps the threshold. Carrying the summation over k from a fraction of K up to K and using gamma_k = 2/(k+2), the cleanest bound I can extract is that for K >= 2 some iterate k_hat in {1,...,K} has

```text
g(x_{k_hat}) <= 2 beta C_f/(K+2),  beta = 27/8.
```
