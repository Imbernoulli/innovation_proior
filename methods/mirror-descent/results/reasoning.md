I have a convex function f on a closed convex set X in R^n, possibly nonsmooth, Lipschitz with constant L_f, and I can get a subgradient g ∈ ∂f(x) anywhere. n is huge — think a million variables, a tomography reconstruction. I just want an x with f(x) − min_X f ≤ ε, and I want the number of iterations to get there to barely grow with n, because anything that solves a linear system per step is hopeless at this scale. So I'm committed to a first-order method: one subgradient, one cheap projection, repeat.

The workhorse is projected subgradient: x_{k+1} = Π_X(x_k − t_k g_k), where Π_X is the Euclidean projection onto X. It's beautifully cheap when X is simple — a ball, a box, the simplex. And its rate, with a horizon-tuned stepsize of order Diam(X)/√k, is min_{s≤k} f(x^s) − f* ≤ O(1)·L_f·Diam(X)/√k. Slow — only 1/√k, so only a couple of digits of accuracy — but the constant is almost independent of dimension, which is exactly what I need at scale. Good enough in principle.

But let me actually look at that constant on a concrete domain, because I keep hearing that on something like the simplex this method is wasteful, and I want to see *where* the waste is. Take X = Δ_n = {x ≥ 0, Σ x_j = 1}, the probability simplex. The Euclidean diameter of Δ_n is just √2 — a constant, no dimension in it, great. So Diam(X) isn't the problem. The problem must be hiding in L_f. L_f is the Lipschitz constant in the *Euclidean* norm, which is the same as sup ‖g‖_2 over the subgradients. Now on the simplex the natural way a gradient is bounded is in ℓ_∞ — for example if f is a worst-case loss over n experts and the per-expert losses live in [0,1], then ‖g‖_∞ ≤ 1, that's the honest, dimension-free bound on the gradient. But ‖g‖_2 can be as large as √n · ‖g‖_∞. So a gradient that is genuinely O(1) in the right units becomes √n in the Euclidean units the method insists on using. The rate carries a hidden √n: on a million-dimensional simplex, a factor of a thousand in the error constant, and squared in the worst-case iteration count needed to hit a fixed accuracy.

So the slowness isn't intrinsic; it's a units mismatch. The method measures everything — the gradient's size, the distance moved — with the Euclidean ruler, and the simplex is not a Euclidean object. The simplex is an ℓ_1-flavored thing; its natural notion of "how far did I move" is entropy / KL, not squared distance. I want to measure progress in a geometry adapted to X. The question is how to bend the method to do that without losing its cheapness or its dimension-mildness.

Let me stare at the update and ask what's really Euclidean about it. x_{k+1} = Π_X(x_k − t_k g_k). The projection is one Euclidean thing. But there's a cleaner way to see the whole step. Consider the linearized-proximal problem
  x_{k+1} = argmin_{x∈X} { ⟨g_k, x⟩ + (1/2t_k)‖x − x_k‖_2² }.
Differentiate the unconstrained version: g_k + (1/t_k)(x − x_k) = 0 ⇒ x = x_k − t_k g_k, then the constraint pulls it back via Euclidean projection. So this *is* the projected subgradient step. And read this way it has a story: replace f locally by its linearization ⟨g_k, x⟩ (the only first-order information I have), but don't trust the linearization too far — penalize moving away from x_k. The penalty is ½‖x − x_k‖_2². And now it's glaring: that squared-Euclidean penalty is a *choice*. There's no law saying the "don't move too far" term has to be Euclidean. It's the only place the Euclidean ruler enters as a free parameter, and it's exactly the place where the geometry of X should be entering.

So replace it. Let D(x, y) be a general "distance-like" function — I'll demand D(x,y) ≥ 0 with equality iff x = y, but nothing else yet — and consider
  x_{k+1} = argmin_{x∈X} { ⟨g_k, x⟩ + (1/t_k) D(x, x_k) }.
For D = ½‖·‖² this is the old method back. For other D, this is a *nonlinear* projected subgradient step, and the hope is that a D matched to X's geometry kills the √n. The whole game is now: which D?

I can't take D totally arbitrary — I need the convergence proof to go through, and the proof for projected subgradient leans hard on a Euclidean identity (the cosine rule, ‖a − c‖² = ‖a − b‖² + ‖b − c‖² − 2⟨a−b, b−c⟩, which is what makes the telescoping work). So I need a family of D that comes with an analogous identity. What generates a clean second-order-ish penalty with a cosine-rule cousin? A convex potential. Take a strongly convex, differentiable ψ on X, and define
  B_ψ(x, y) = ψ(x) − ψ(y) − ⟨∇ψ(y), x − y⟩,
the gap between ψ at x and its first-order (tangent) approximation built at y. By convexity B_ψ ≥ 0; by strict convexity it's 0 only when x = y. And for ψ = ½‖·‖_2² it's literally ψ(x) − ψ(y) − ⟨y, x−y⟩ = ½‖x‖² − ½‖y‖² − ⟨y, x−y⟩ = ½‖x − y‖_2². So B_ψ with ψ = ½‖·‖² gives back exactly the Euclidean penalty. This is the family. The Bregman divergence of ψ. The choice of ψ *is* the choice of geometry, and ½‖·‖² sits inside it as one point. (Bregman built these in 1967 for convex feasibility; here the role is to be the penalty.)

So now my method is x_{k+1} = argmin_{x∈X} { ⟨g_k, x⟩ + (1/t_k) B_ψ(x, x_k) } and I get to pick ψ. Before I optimize ψ, I want to understand the *shape* of this step, because right now it's an argmin I don't know how to compute, and there's a deeper reason the Euclidean version felt so natural that I want to recover.

Here's the thing that's been nagging me about ⟨g_k, x⟩ and x − t_k g_k. The subgradient g is a *linear form* — it eats a direction and returns a number, f(y) ≥ f(x) + ⟨g, y−x⟩. It lives in the dual space. In a Euclidean space I get away with writing x − t g because Riesz representation silently identifies the space with its dual; primal and dual are isometric and I never notice. But the moment my geometry isn't Euclidean — say I'm really working in ℓ_1 on the simplex — there *is* no such identification, and "x minus a dual vector" is a type error. I'm subtracting an apple from an orange and projecting the result. No wonder the units are wrong: the additive step x − tg only makes sense in the one geometry where it's accidentally fine.

So how do I take a gradient step when the gradient lives in the dual? I need a bridge from primal to dual. And I have one sitting right here: ∇ψ. The gradient of my potential maps a primal point x to a dual vector ∇ψ(x). So the move is: take x_k, push it to the dual as ∇ψ(x_k); *there*, in the dual, the gradient step is honest — both ∇ψ(x_k) and g_k are dual objects, so ∇ψ(x_k) − t_k g_k is a legitimate dual vector; then pull the result back to the primal. Pulling back means inverting ∇ψ. Conjugate duality hands me the inverse for free: for the conjugate ψ*(z) = max_x{⟨z,x⟩ − ψ(x)}, we have ∇ψ* = (∂ψ)^{-1}. So the back-map is ∇ψ*. The whole step is three lines:
  x_k = ∇ψ*(y_k),                         (read the current primal point off its dual coordinate)
  y_{k+1} = ∇ψ(x_k) − t_k g_k,            (gradient step, done in the dual where it's well-typed)
  x_{k+1} = ∇ψ*(y_{k+1}).                 (mirror back to the primal)
The point gets reflected into the dual through the "mirror" ∇ψ, stepped, and reflected back. For ψ = ½‖·‖² the mirror is the identity, ∇ψ(x) = x, ∇ψ*(z) = z, and the three lines collapse to y_{k+1} = x_k − t_k g_k, x_{k+1} = y_{k+1} — plain gradient descent. So the additive step is the degenerate case where the mirror is flat. That's deeply satisfying: ordinary GD isn't wrong, it's just the special geometry where primal and dual coincide.

Now the constraint set. The clean three-line version ignored X; with a constraint, the argmin form is the truth, and I should check it really does reduce to "step in the dual, then come back into X." Write the optimality condition for x_{k+1} = argmin_{x∈X}{ t_k⟨g_k,x⟩ + B_ψ(x,x_k) }. The gradient in x of B_ψ(x, x_k) is ∇ψ(x) − ∇ψ(x_k). So stationarity with the constraint, using the normal cone N_X, is
  0 ∈ t_k g_k + ∇ψ(x_{k+1}) − ∇ψ(x_k) + N_X(x_{k+1}),
i.e. (∇ψ + N_X)(x_{k+1}) ∋ ∇ψ(x_k) − t_k g_k, so
  x_{k+1} = (∇ψ + N_X)^{-1}( ∇ψ(x_k) − t_k g_k ).
That's exactly the same dual step ∇ψ(x_k) − t_k g_k, with the back-map now being (∇ψ + N_X)^{-1} instead of ∇ψ*. And there's a cleaner way to say "(∇ψ + N_X)^{-1} applied to a dual point": it's the Bregman projection. Define Π_X^ψ(y) = argmin_{x∈X} B_ψ(x, y). Then the unconstrained back-map ∇ψ* lands me at some y_{k+1}^raw = ∇ψ*(∇ψ(x_k) − t_k g_k), possibly outside X, and Π_X^ψ pulls it back onto X measuring distance with B_ψ rather than Euclidean distance. So the method is exactly: mirror to the dual, gradient step, mirror back, Bregman-project onto X. The two pictures — the argmin with a Bregman penalty, and the explicit dual-step-then-project — are the same algorithm; one is the implicit/proximal face and the other is the explicit/operational face. (I needed ∇ψ* = (∂ψ)^{-1} to glue them; that's the conjugate identity, and it's also why I want ψ strongly convex — strong convexity is exactly what makes ∇ψ* single-valued and Lipschitz, so the back-map is a genuine function and not a set.)

Good. I have a method with a knob ψ. Now the part that decides everything: the convergence bound, and where ψ enters it. I want to see the bound first, *then* choose ψ to make it small — that's the whole point.

The Euclidean proof telescopes ‖x* − x_k‖² via the cosine rule. I need the Bregman cosine rule. Let me derive it. For any a, b, c, just expand the three Bregman terms from the definition:
  B_ψ(c, a) = ψ(c) − ψ(a) − ⟨∇ψ(a), c − a⟩,
  B_ψ(a, b) = ψ(a) − ψ(b) − ⟨∇ψ(b), a − b⟩,
  B_ψ(c, b) = ψ(c) − ψ(b) − ⟨∇ψ(b), c − b⟩.
Compute B_ψ(c,a) + B_ψ(a,b) − B_ψ(c,b). The ψ(c), ψ(a), ψ(b) terms cancel pairwise. The linear terms:
  −⟨∇ψ(a), c−a⟩ − ⟨∇ψ(b), a−b⟩ + ⟨∇ψ(b), c−b⟩ = −⟨∇ψ(a), c−a⟩ + ⟨∇ψ(b), (c−b) − (a−b)⟩ = −⟨∇ψ(a), c−a⟩ + ⟨∇ψ(b), c−a⟩.
So
  B_ψ(c, a) + B_ψ(a, b) − B_ψ(c, b) = ⟨∇ψ(b) − ∇ψ(a), c − a⟩.
There's the three-point identity. For ψ = ½‖·‖² it's the ordinary cosine rule. This is the generalization I needed; everything in the convergence proof rides on it.

Now run the proof. Let x* be an optimum (offline; for the online/regret reading replace x* by any comparator u, the algebra is identical). The step's optimality condition, for any x ∈ X, is ⟨x − x_{k+1}, t_k g_k + ∇ψ(x_{k+1}) − ∇ψ(x_k)⟩ ≥ 0. Put x = x*:
  ⟨x* − x_{k+1}, t_k g_k − (∇ψ(x_k) − ∇ψ(x_{k+1}))⟩ ≥ 0.   (⋆)
Equivalently,
  ⟨x* − x_{k+1}, t_k g_k⟩ ≥ ⟨x* − x_{k+1}, ∇ψ(x_k) − ∇ψ(x_{k+1})⟩.
Start from convexity of f, which is the only handle I have on f(x_k) − f(x*):
  0 ≤ t_k(f(x_k) − f(x*)) ≤ t_k ⟨g_k, x_k − x*⟩ = ⟨x_k − x*, t_k g_k⟩.
I'll split x_k − x* = (x_k − x_{k+1}) + (x_{k+1} − x*) and route the second piece through (⋆). Write
  ⟨x_k − x*, t_k g_k⟩ = ⟨x_k − x*, ∇ψ(x_k) − ∇ψ(x_{k+1})⟩  + ⟨x_k − x*, t_k g_k − ∇ψ(x_k) + ∇ψ(x_{k+1})⟩.
Hmm, let me instead organize it the clean way the three-point identity wants. The quantity I can telescope is built from ⟨∇ψ(x_k) − ∇ψ(x_{k+1}), x* − x_{k+1}⟩, because by the identity with (c,a,b) = (x*, x_{k+1}, x_k):
  ⟨∇ψ(x_k) − ∇ψ(x_{k+1}), x* − x_{k+1}⟩ = B_ψ(x*, x_{k+1}) + B_ψ(x_{k+1}, x_k) − B_ψ(x*, x_k).
Let me get the suboptimality into exactly this shape. Using convexity then (⋆):
  t_k(f(x_k) − f(x*)) ≤ ⟨x_k − x*, t_k g_k⟩.
From (⋆), along the direction x* − x_{k+1},
  ⟨x* − x_{k+1}, t_k g_k⟩ ≥ ⟨x* − x_{k+1}, ∇ψ(x_k) − ∇ψ(x_{k+1})⟩.
Decompose the left side I want to bound as
  ⟨x_k − x*, t_k g_k⟩ = ⟨x_k − x_{k+1}, t_k g_k⟩ + ⟨x_{k+1} − x*, t_k g_k⟩
                       = ⟨x_k − x_{k+1}, t_k g_k⟩ − ⟨x* − x_{k+1}, t_k g_k⟩
                       ≤ ⟨x_k − x_{k+1}, t_k g_k⟩ − ⟨x* − x_{k+1}, ∇ψ(x_k) − ∇ψ(x_{k+1})⟩.
Now the second term is exactly the three-point quantity with a sign flip:
  −⟨x* − x_{k+1}, ∇ψ(x_k) − ∇ψ(x_{k+1})⟩ = −[ B_ψ(x*, x_{k+1}) + B_ψ(x_{k+1}, x_k) − B_ψ(x*, x_k) ]
                                          = B_ψ(x*, x_k) − B_ψ(x*, x_{k+1}) − B_ψ(x_{k+1}, x_k).
So
  t_k(f(x_k) − f(x*)) ≤ B_ψ(x*, x_k) − B_ψ(x*, x_{k+1}) + [ ⟨x_k − x_{k+1}, t_k g_k⟩ − B_ψ(x_{k+1}, x_k) ].
The first two terms telescope when I sum over k — that's the whole point, B_ψ(x*, ·) plays the role ‖x* − ·‖² played in the Euclidean proof. I just need to control the bracket. Here strong convexity earns its keep. B_ψ(x_{k+1}, x_k) ≥ (σ/2)‖x_{k+1} − x_k‖², so
  ⟨x_k − x_{k+1}, t_k g_k⟩ − B_ψ(x_{k+1}, x_k) ≤ t_k ‖g_k‖_* ‖x_k − x_{k+1}‖ − (σ/2)‖x_k − x_{k+1}‖²
by generalized Cauchy–Schwarz (gradient measured in the dual norm — and *that's* the norm the bound will use, which is exactly the lever I wanted). Maximize the right side over the scalar r = ‖x_k − x_{k+1}‖: it's t_k‖g_k‖_* · r − (σ/2) r², a downward parabola, max value (t_k‖g_k‖_*)² / (2σ). So the bracket ≤ t_k² ‖g_k‖_*² / (2σ). Therefore
  t_k(f(x_k) − f(x*)) ≤ B_ψ(x*, x_k) − B_ψ(x*, x_{k+1}) + t_k² ‖g_k‖_*² / (2σ).
Sum k = 1..s. The Bregman terms telescope to B_ψ(x*, x_1) − B_ψ(x*, x_{s+1}) ≤ B_ψ(x*, x_1) since B_ψ ≥ 0:
  Σ_{k=1}^s t_k (f(x_k) − f(x*)) ≤ B_ψ(x*, x_1) + (1/2σ) Σ_{k=1}^s t_k² ‖g_k‖_*².
Divide by Σ t_k and use that the min is below the weighted average:
  min_{1≤s≤k} f(x^s) − f* ≤ [ B_ψ(x*, x_1) + (2σ)^{-1} Σ_{s=1}^k t_s² ‖g_s‖_*² ] / Σ_{s=1}^k t_s.
There's the efficiency estimate. Stare at the constant. The gradient enters as ‖g‖_*, the *dual* norm — not the Euclidean norm, the dual of whatever ‖·‖ I chose. And the "size of the domain" enters as B_ψ(x*, x_1)/σ — the Bregman radius of X under my potential, divided by its strong-convexity modulus, *not* the Euclidean diameter. Those two quantities are the entire dependence on geometry, and both are mine to shape through (ψ, ‖·‖). If I diminish t_k → 0 with Σ t_k → ∞ the right side → 0, so it converges; and the bound is a clean product (gradient dual-norm) × (Bregman radius) that I can now minimize over the geometry.

Let me actually optimize the stepsize to see the rate cleanly. Bound ‖g_s‖_* ≤ L_f (the Lipschitz constant in *my* norm) and write c = B_ψ(x*, x_1). The right side is [c + (2σ)^{-1} L_f² Σ t_s²]/Σ t_s. For a run whose horizon k is known, take the same step t_s = t for s = 1,...,k; then the bound becomes c/(kt) + L_f² t/(2σ). Minimizing over t gives t = √(2σ c)/(L_f √k), with value L_f √(2c/σ)/√k. So
  min_{1≤s≤k} f(x^s) − f* ≤ L_f · √( 2 B_ψ(x*, x_1) / σ ) · (1/√k),
with t_s = √(2σ B_ψ(x*,x_1))/(L_f√k) throughout that k-step run. In the online reading with a fixed step η, the same one-step inequality gives Regret_T ≤ B_ψ(x*,x_1)/η + η Σ‖g_t‖_*²/(2σ); with B_ψ(x*,x_1) ≤ R² and ‖g_t‖_* ≤ L_f, η = R√(2σ)/(L_f√T) gives Regret_T ≤ R L_f √(2T/σ), or average regret R L_f √(2/(σT)). Now everything hinges on choosing (ψ, ‖·‖) to make L_f² · B_ψ(x*,x_1)/σ small for the X at hand.

So, the two cases. First the sanity check: ψ = ½‖·‖_2², ‖·‖ = ℓ_2. Then σ = 1, B_ψ(x*, x_1) = ½‖x* − x_1‖² ≤ ½ Diam(X)², ‖g‖_* = ‖g‖_2 = L_f, and the mirror map is the identity so the update is y_{k+1} = x_k − t_k g_k, x_{k+1} = Π_X(y_{k+1}) — projected subgradient, with the classical Euclidean rate ≈ L_f Diam(X)/√k. Nothing gained, nothing lost; the old method is exactly the flat-mirror corner of the new one.

Now the simplex, where I started, where the √n was bleeding out. I want a potential whose geometry *is* the simplex. The simplex's effective domain is the nonnegative orthant; its natural distance is KL; its natural norm is ℓ_1 (so the dual, where gradients should be measured, is ℓ_∞ — exactly the norm in which simplex gradients are honestly O(1)). The potential whose Bregman divergence is KL and whose effective domain is the orthant is the negative entropy:
  ψ_e(x) = Σ_j x_j ln x_j  on Δ_n.
Let me just compute the update with this ψ_e and watch what falls out. ∇ψ_e(x) has j-th component ∂/∂x_j (x_j ln x_j) = ln x_j + 1. So the dual step ∇ψ_e(x_{k+1}^raw) = ∇ψ_e(x_k) − t_k g_k reads componentwise
  ln x_{k+1,j}^raw + 1 = ln x_{k,j} + 1 − t_k g_{k,j}  ⇒  ln x_{k+1,j}^raw = ln x_{k,j} − t_k g_{k,j}  ⇒  x_{k+1,j}^raw = x_{k,j} · e^{−t_k g_{k,j}}.
The mirror map turned the additive dual step into a *multiplicative* primal update. Then the Bregman projection onto Δ_n: minimize KL(x ‖ x^raw) subject to Σ x_j = 1, which is just renormalization — divide by the sum. So
  x_{k+1,j} = x_{k,j} e^{−t_k g_{k,j}} / Σ_i x_{k,i} e^{−t_k g_{k,i}}.
That's the multiplicative-weights update — Hedge, exponentiated gradient. It wasn't put in by hand; it *is* mirror descent with the entropy potential. The weighted-majority / Hedge / EG rule that learning theorists derived with bespoke exponential-potential or relative-entropy arguments is the same algorithm as projected subgradient, viewed in entropy geometry instead of Euclidean geometry. The exponential update and the additive update are two faces of one template, separated only by the choice of mirror.

Now does the bound actually deliver the √(log n) I was chasing? Two quantities to pin down: the strong-convexity modulus σ of ψ_e in ℓ_1, and the Bregman radius B_ψ(x*, x_1).

Strong convexity. I claim ψ_e is 1-strongly convex w.r.t. ‖·‖_1 on Δ_n, i.e. for x, y interior to Δ_n,
  ⟨∇ψ_e(x) − ∇ψ_e(y), x − y⟩ = Σ_j (x_j − y_j) ln(x_j / y_j) ≥ ‖x − y‖_1².
This is essentially Pinsker's inequality, but let me prove it directly so I'm sure of the constant. Define, for t > 0, φ(t) = (t − 1) ln t − 2 (t − 1)² / (t + 1). Then φ(1) = 0. Differentiate twice:
  φ''(t) = (t − 1)²(t² + 6t + 1) / [t²(t + 1)³] ≥ 0.
So φ is convex on (0,∞). Since φ'(1) = 0, t = 1 is a global minimizer, and φ(t) ≥ φ(1) = 0. Therefore (t − 1) ln t ≥ 2(t − 1)²/(t + 1) for all t > 0. Set t = x_j / y_j:
  (x_j − y_j) ln(x_j/y_j) = y_j (t − 1) ln t ≥ y_j · 2(t−1)²/(t+1) = 2 (x_j − y_j)² / (x_j + y_j).
Sum over j:
  Σ_j (x_j − y_j) ln(x_j/y_j) ≥ Σ_j 2(x_j − y_j)²/(x_j + y_j) = Σ_j [ (x_j+y_j)/2 ] · [ (x_j − y_j)/((x_j+y_j)/2) ]².
Now Cauchy–Schwarz, weighting by the probabilities w_j = (x_j+y_j)/2 (they sum to 1 since x, y ∈ Δ_n):
  ( Σ_j w_j |u_j| )² ≤ ( Σ_j w_j ) ( Σ_j w_j u_j² ) = Σ_j w_j u_j²,  with u_j = (x_j − y_j)/w_j.
So Σ_j w_j u_j² ≥ ( Σ_j w_j |u_j| )² = ( Σ_j |x_j − y_j| )² = ‖x − y‖_1². Therefore
  Σ_j (x_j − y_j) ln(x_j/y_j) ≥ ‖x − y‖_1²,
which is the σ = 1 strong convexity of ψ_e in ℓ_1. If exactly one of x_j,y_j is zero, the corresponding logarithmic term is +∞ by limit, so the inequality only gets easier; if both are zero, the convention 0 ln 0 = 0 gives no contribution. Good — σ = 1, and the dual norm is ℓ_∞, so ‖g‖_* = ‖g‖_∞, the honest dimension-free gradient bound.

The Bregman radius. Start uniform: x_1 = (1/n, …, 1/n). For any x* ∈ Δ_n,
  B_{ψ_e}(x*, x_1) = Σ_j x*_j ln(x*_j / x_{1,j}) = Σ_j x*_j ln x*_j + ln n = ln n − H(x*),
where H(x*) = −Σ x*_j ln x*_j ≥ 0 is the entropy. So B_{ψ_e}(x*, x_1) = ln n − H(x*) ≤ ln n. The Bregman radius of the *whole* simplex, started from the center, is at most ln n. Just log n.

Drop these into the bound: σ = 1, B_ψ(x*,x_1) ≤ ln n, ‖g‖_* = ‖g‖_∞ ≤ ‖f'‖_∞ =: L_f. For a k-step run, use t_s = √(2 ln n)/(L_f√k) for s = 1,...,k, giving
  min_{1≤s≤k} f(x^s) − f* ≤ √(2 ln n) · ‖f'‖_∞ · (1/√k).
Compare what plain projected subgradient gives on Δ_n: the Bregman radius is the Euclidean diameter ~ O(1) but the gradient must be measured in ℓ_2, costing ‖g‖_2 ~ √n ‖g‖_∞, so its rate is ~ √n · ‖f'‖_∞/√k. The entropy mirror map traded √n for √(ln n) — exponentially better dependence on the dimension, and that is precisely because it measured the gradient in ℓ_∞ (where it's O(1)) and measured the domain in entropy (where its radius is ln n, not √n). The Euclidean method was paying √n for the privilege of using the wrong ruler. Same per-step structure — a subgradient and a simple projection-like operation — but a much smaller worst-case constant on a large simplex. That settles it.

Let me also make sure I see *why* entropy in particular, not just any ℓ_1-strongly-convex potential. Three reasons converge. Its effective domain is the nonnegative orthant and ∇ψ_e blows up at the boundary (ln x_j → −∞ as x_j → 0), so the iterates are pinned to the interior automatically — the multiplicative form can never produce a negative coordinate, and the only "projection" needed is renormalization, no boundary handling. Its Bregman divergence is exactly KL, the canonical geometry of distributions, which is what makes the radius come out as the clean ln n. And the mirror ∇ψ_e is the logarithm, whose inverse is the exponential, which is what turns the dual step into the simple closed-form multiplicative update. A different ℓ_1-strongly-convex ψ might give some σ and some radius, but it wouldn't give a closed-form interior-preserving update with the KL radius. Entropy is the natural one for the simplex for the same reason ½‖·‖² is the natural one for a Euclidean ball.

So the picture is complete and it's one template. Pick a norm ‖·‖ matched to the domain X, and a potential ψ that is strongly convex in that norm. Reflect the current point into the dual through the mirror ∇ψ, take the honest gradient step there (the gradient is a dual object and now it's well-typed), reflect back through ∇ψ*, and Bregman-project onto X. Equivalently — the implicit face — minimize the linearized objective plus a Bregman penalty B_ψ(x, x_k). The convergence is one proof: convexity of f, the optimality condition of the step, the Bregman three-point identity to telescope B_ψ(x*, ·), and strong convexity to cap the leftover term by ‖g‖_*²/(2σ); summing gives min f − f* ≤ [B_ψ(x*,x_1) + (2σ)^{-1}Σ t_s²‖g_s‖_*²]/Σ t_s, i.e. L_f√(2B_ψ(x*,x_1)/σ)/√k at the optimal step. The geometry shows up only through the dual-norm gradient size and the Bregman radius over the strong-convexity modulus, and you tune it to the domain. Squared-Euclidean potential gives plain gradient descent and the Euclidean rate; negative entropy on the simplex gives the multiplicative-weights update and a rate that depends on the dimension only as √(log n). The "wrongness" of Euclidean geometry on the simplex was never about the method — it was about insisting on a ruler the domain doesn't carry, and the fix is to let a strongly convex potential supply the right one.

```python
import numpy as np

# Mirror descent: x_{k+1} = argmin_{x in X} { t_k <g_k, x> + B_psi(x, x_k) }
# realized as  mirror to dual (grad psi) -> gradient step in dual -> mirror back (grad psi*) -> Bregman-project.
# Two mirror maps below recover (a) gradient descent and (b) multiplicative weights.

# (a) EUCLIDEAN mirror: psi = 1/2 ||x||_2^2.  grad psi = identity, Bregman = 1/2||.||_2^2.
#     The dual step is additive; the back-map + Bregman projection is the Euclidean projection.
def gradient_descent_step(x, g, t, project_euclidean):
    y = x - t * g                       # gradient step in the dual (= primal here; mirror is flat)
    return project_euclidean(y)         # Bregman projection w.r.t. 1/2||.||^2 = Euclidean projection

# (b) ENTROPY mirror on the simplex: psi_e(x) = sum_j x_j ln x_j on Delta_n.
#     grad psi_e(x)_j = 1 + ln x_j  => dual step is additive in log-space
#     => primal update is multiplicative; Bregman projection onto the simplex = renormalization.
def entropy_mirror_step(x, g, t):
    log_y = np.log(x) - t * g           # additive gradient step in the DUAL (log) coordinates
    y = np.exp(log_y)                   # mirror back via grad psi*_e = exp  -> x_j * exp(-t g_j)
    # x_{k+1,j} = x_{k,j} exp(-t g_j) / sum_i x_{k,i} exp(-t g_i)
    return y / y.sum()                  # Bregman (KL) projection onto Delta_n = l1 renormalization

def mirror_descent(f_oracle, mirror_step, x1, num_iters, L_f, bregman_radius, sigma=1.0):
    # horizon-tuned step matching the bound for a num_iters-step run
    t = np.sqrt(2.0 * sigma * bregman_radius) / (L_f * np.sqrt(num_iters))
    x = x1
    best_x, best_val = x1, f_oracle.value(x1)
    for k in range(1, num_iters + 1):
        g = f_oracle.subgrad(x)
        x = mirror_step(x, g, t)        # the geometry lives entirely in this one call
        val = f_oracle.value(x)
        if val < best_val:
            best_x, best_val = x, val
    return best_x

# On Delta_n with entropy: sigma=1, bregman_radius <= ln n, ||g||_* = ||g||_inf
#   => min_s f(x^s) - f* <= sqrt(2 ln n) * ||f'||_inf / sqrt(k)   (vs ~ sqrt(n) for Euclidean).
```
