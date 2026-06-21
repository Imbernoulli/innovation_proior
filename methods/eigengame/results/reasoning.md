I have a matrix `X ∈ R^{n×d}` and I want its top-k principal components — the eigenvectors of `M = XᵀX`, ordered by eigenvalue. On a small problem I would just call SVD and walk away. But the problems I actually care about have `n` around a million and `d` in the tens of millions: flattened activations of a deep network over a whole image dataset. SVD costs `O(min{nd², n²d})` time and `O(nd)` space, and I cannot even form `M`, which has `d²` entries. So the question is not "what is the answer" — linear algebra settled that a century ago — it's "what iterative procedure recovers the answer when the data is enormous and the only compute I have is a farm of accelerators with fast interconnects between them."

That last clause is doing more work than it looks. The hardware is parallel. The natural thing would be to give each accelerator one eigenvector to learn, feed it a shard of the data, and let it talk to the others as little as possible. So I am really after a procedure that *decentralizes*: per-vector updates, minimal communication, no step that forces all `k` vectors to synchronize at once.

Let me list the tools. The top-1 eigenvector falls out of Hebb's rule, `v ← v + ηMv` — repeatedly multiply by `M`, the leading direction dominates. The norm runs away, so Oja's rule corrects it, `v ← v + η(I − vvᵀ)Mv`, where the `−vvᵀMv` quietly pins `v` near the unit sphere; or, in ML practice, just renormalize, `v ← v/‖v‖`, which as `η→∞` is the plain power method. Krasulina is Oja-with-renormalization. These are all top-1.

To get `k` of them, what does everyone do? They run a top-1 rule and then **re-orthonormalize** the whole bundle — a `QR` of the `d×k` iterate every step, with sign bookkeeping (the usual top-k "Oja's algorithm"); or project onto the Stiefel manifold of orthonormal matrices and `QR` (Matrix Krasulina); or deflate, `X_{(i)} ← X(I − Σ_{j<i} v̂_jv̂_jᵀ)`, and run a top-1 rule on the deflated data. Sanger's Generalized Hebbian Algorithm folds the deflation into a single update,
`Δv̂_i = 2[Mv̂_i − (v̂_iᵀMv̂_i)v̂_i − Σ_{j<i}(v̂_iᵀMv̂_j)v̂_j]`,
and it does converge to the actual components, and it can be distributed.

So GHA basically does what I want. Why am I not done? Stare at that update. The `−(v̂_iᵀMv̂_i)v̂_i` term is Oja-style self-normalization; the `−Σ_{j<i}(v̂_iᵀMv̂_j)v̂_j` is a Gram–Schmidt-style projecting-out of the parents. It is a recipe — a sequence of operations someone wrote down because it works. But what is each vector *trying to do*? Is there a payoff function `u_i` such that this update is its gradient? Let me just check, because if there is, I could reason about the whole thing as an optimization and borrow convergence theory; and if there isn't, I should be suspicious of the recipe. The update is a vector field `v̂_i ↦ Δv̂_i`; for it to be `∇u_i` of some scalar, its Jacobian must be a Hessian, hence symmetric. Differentiate: the term `−Σ_{j<i}(v̂_iᵀMv̂_j)v̂_j` contributes `−Σ_{j<i}v̂_jv̂_jᵀM` to the Jacobian. That is a product `v̂_jv̂_jᵀM` of a rank-one projector with `M`, and in general `v̂_jv̂_jᵀM ≠ (v̂_jv̂_jᵀM)ᵀ = Mv̂_jv̂_jᵀ`. The Jacobian is **not symmetric**, so it is not the Hessian of anything, so the GHA update is **not the gradient of any function**. There is no objective hiding behind it. It works, but blindly; I can't ask "what does player `i` want," can't import optimization guarantees cleanly, can't reason about equilibria.

That bothers me enough to start over from objectives. What scalar should each vector maximize? Let me get the centralized objective right first, then worry about decentralizing.

Define `R(V̂) := V̂ᵀMV̂`. Its diagonal `R_ii = ⟨v̂_i, Mv̂_i⟩` is the Rayleigh quotient — the variance captured along `v̂_i`, since `‖v̂_i‖=1`. The off-diagonal `R_ij = ⟨v̂_i, Mv̂_j⟩` is the alignment of `v̂_i` and `v̂_j` measured through `M`. The obvious objective is "maximize the variance," i.e. maximize the trace:
`max_{V̂ᵀV̂=I} Tr(R) = Tr(V̂ᵀMV̂) = Tr(V̂V̂ᵀM)`.
And here is a wall I should hit on purpose. Take `k=d`. Then `V̂` is square orthonormal, `V̂V̂ᵀ = I`, and `Tr(V̂V̂ᵀM) = Tr(M)`. The objective is `Tr(M)` — a **constant**, completely independent of `V̂`. Maximizing the trace cannot recover all the eigenvectors; *any* orthonormal basis maximizes it. Even for `k<d`, maximizing the trace pulls `V̂` toward the top-k subspace but expresses no preference for *which* directions inside it the columns point — it learns the **subspace**, not the **components**. That is exactly the trap a lot of fast methods fall into: they nail the subspace and quietly hand off the rotation to a downstream SVD. But the components are what I want — the first principal axis, the second — each aligned to a specific direction of variance.

So what part of the problem distinguishes components from subspace? Go back to the eigenvalue equation: orthonormal `V` solving `MV = VΛ` satisfies, after left-multiplying by `Vᵀ`, the identity `VᵀMV = Λ`, a **diagonal** matrix. So `V̂` is the true eigenvector set exactly when `R(V̂)` is diagonal — when all the off-diagonals vanish. That is the missing half:
`min_{V̂ᵀV̂=I} Σ_{i≠j} R_ij²`.
Trace-max wants the big-variance subspace but is indifferent to rotation; off-diagonal-min wants the columns to actually be eigenvectors but is indifferent to *which* eigenvalues (large or small) they capture. Each alone is incomplete; together they should pin down the ordered components. The cleanest reading: trace = reward (capture variance), off-diagonals = penalty (don't be aligned with the others).

The naive combination is
`max_{V̂ᵀV̂=I} Σ_i R_ii − Σ_{i≠j} R_ij²`.
Let me poke at it. The penalty is **symmetric**: `v̂_1` is penalized for aligning with `v̂_k` and vice versa. But think about what `v̂_1` should be doing. It is the estimate of the *largest* eigenvector; it should be free to chase the single direction of maximum variance, with no regard whatsoever for where the other vectors happen to be. Forcing `v̂_1` to avoid `v̂_k` is wrong — it should ignore `v̂_k` entirely. The symmetric penalty contradicts the natural hierarchy of the components. There's an ordering baked into PCA — first component, second component — and the objective is fighting it.

Let me rebuild respecting that hierarchy, one vector at a time. Top-1: there are no off-diagonals, `R` is the `1×1` matrix `[⟨v̂_1, Mv̂_1⟩]`, and the sensible objective is just the Rayleigh quotient,
`max_{‖v̂_1‖=1} ⟨v̂_1, Mv̂_1⟩`,
which is maximized at the leading eigenvector — standard. Good. `v̂_1` answers to no one.

Now top-2. `v̂_1` keeps its utility unchanged; `v̂_2` gets a new one. The off-diagonal penalty is now live, so `v̂_2` wants `⟨v̂_2, Mv̂_2⟩` large and `⟨v̂_2, Mv̂_1⟩²` small. Write
`max ⟨v̂_2, Mv̂_2⟩ − ⟨v̂_2, Mv̂_1⟩²` subject to `‖v̂_2‖=1` (and, if I like, `v̂_1ᵀv̂_2=0`).
And here is the asymmetry I was missing: `v̂_2` is penalized for aligning with `v̂_1`, but `v̂_1` is *not* penalized for aligning with `v̂_2`. The penalty only points up the hierarchy, from child to parent, never back down. Each vector `i` cares only about its **parents** `j<i`. The moment I make the objectives asymmetric across the vectors, I am no longer doing a single joint optimization — different vectors optimize *different* functions. That is a **game**: `k` players, player `i` controls `v̂_i`, maximizing its own payoff `u_i` given the others. The reward/penalty structure was hiding a game all along, and it's the hierarchy that reveals it.

Before I commit, two refinements on the top-2 penalty. First, scale. The reward `⟨v̂_2,Mv̂_2⟩` is on the order of an eigenvalue, possibly thousands; the bare penalty `⟨v̂_2,Mv̂_1⟩²` is on a totally different scale, and worse, when `v̂_1` happens to point along a huge-eigenvalue direction, the reward term can simply drown the penalty out. (This is the failure mode of the *truly* naive version that subtracts `Σ_{j<i}⟨v̂_i, v̂_j⟩` — a plain Euclidean overlap. Big eigenvalues swamp it; including `M` in the penalty's inner product is what gives it the right boost to balance against the reward.) To put the two terms on a comparable footing I divide the penalty by the parent's own Rayleigh quotient:
`u_2(v̂_2 | v̂_1) = ⟨v̂_2, Mv̂_2⟩ − ⟨v̂_2, Mv̂_1⟩² / ⟨v̂_1, Mv̂_1⟩`.
Second, the explicit orthogonality constraint `v̂_1ᵀv̂_2=0` — do I even need it? At the optimum `v̂_1*=v_1, v̂_2*=v_2`, the penalty term is `⟨v_2, Mv_1⟩² = (Λ_{11}⟨v_2, v_1⟩)² = Λ_{11}²⟨v_2,v_1⟩²`, which already blows up the instant `v̂_2` drifts toward `v_1`. The penalty itself enforces the orthogonality I'd otherwise impose by hand, so the explicit constraint is redundant at the solution; I can drop it and keep only `‖v̂_2‖=1`.

Generalizing to all `k` players, player `i` maximizes, subject to `‖v̂_i‖=1`,
`u_i(v̂_i | v̂_{j<i}) = v̂_iᵀMv̂_i − Σ_{j<i} (v̂_iᵀMv̂_j)² / (v̂_jᵀMv̂_j) = ‖Xv̂_i‖² − Σ_{j<i} ⟨Xv̂_i, Xv̂_j⟩² / ⟨Xv̂_j, Xv̂_j⟩`.
Reward = variance along `v̂_i`. Penalty = squared generalized inner product with each parent, normalized by the parent's variance. Pure, per-vector, depends only on parents — exactly the decentralized structure I wanted: arrange the players on a directed acyclic graph, each parent broadcasts its current vector down to its children, and that's all the communication there is.

Now I should *prove* this game's solution is the thing I want. The right notion is a Nash equilibrium: a choice of vector for each player from which no player can unilaterally improve. `V̂` is a strict Nash if for every `i`, `u_i(v̂_i | v̂_{j<i}) > u_i(z | v̂_{j<i})` for all other unit `z`. Claim: the unique strict Nash is the top-k eigenvectors. Let me grind it out.

Diagonalize `M = UΛUᵀ`, `U` unitary. For any unit vector `v̂_i`, the rotated vector `Uᵀv̂_i` is also unit-length (since `U` preserves inner products), and `R = V̂ᵀMV̂ = (UᵀV̂)ᵀΛ(UᵀV̂)`. So I can analyze the action of an arbitrary unit-column matrix on the *diagonal* `Λ` instead of on `M` — `V` becomes the identity without loss of generality. Cleaner. I'll prove player `i`'s best response, given perfect parents `v_{j<i}`, is `v_i`, by induction. Base case: `u_1 = ⟨v̂_1, Mv̂_1⟩/⟨v̂_1, v̂_1⟩` is the Rayleigh quotient, maximized at `Λ_{11}` by `v_1`. Done.

Inductive step. Write the candidate as `v̂_i = Σ_p w_p v_p` with `‖w‖=1` (to keep `‖v̂_i‖=1`). Plug into the utility with true parents:
`u_i = v̂_iᵀMv̂_i − Σ_{j<i} (v̂_iᵀMv_j)²/Λ_{jj}`.
Expand each piece. The reward: `v̂_iᵀMv̂_i = Σ_p Σ_q w_p w_q v_pᵀMv_q = Σ_p Σ_q w_p w_q Λ_{qq} v_pᵀv_q = Σ_q w_q² Λ_{qq}`, using `v_pᵀv_q = δ_{pq}`. The penalty numerator for parent `j`: `v̂_iᵀMv_j = Σ_p w_p v_pᵀMv_j = Σ_p w_p Λ_{jj} v_pᵀv_j = Λ_{jj} w_j`, so `(v̂_iᵀMv_j)²/Λ_{jj} = Λ_{jj} w_j²`. Therefore
`u_i = Σ_q w_q²Λ_{qq} − Σ_{j<i}Λ_{jj}w_j² = Σ_{p≥i} Λ_{pp} w_p²`.
Set `z_p = w_p²`, so `z ∈ Δ^{d-1}` (nonnegative, sums to one). Then `u_i = Σ_{p≥i} Λ_{pp} z_p`, a **linear function over the simplex**, restricted to indices `p ≥ i` (the first `i−1` got cancelled exactly by the penalty — the normalization is what made that cancellation clean). A linear objective on a simplex is maximized at a vertex; with the `Λ_{pp}` distinct and `Λ_{ii}>0`, the unique maximizer is `z* = e_i`, i.e. `w_i² = 1` — meaning `v̂_i = ±v_i`. So given perfect parents, player `i`'s unique best response is its true eigenvector. By induction every player's best response is its eigenvector, so the top-k eigenvectors are the unique strict Nash, up to sign (and `±v_i` are the same principal component, as expected). The hierarchy is what made this a tower of clean simplex LPs instead of a tangled coupled system.

Was the hierarchy really necessary, or just convenient? Let me check the symmetric version where each player penalizes *all* the others, `u_i = v̂_iᵀMv̂_i − Σ_{j≠i}(v̂_iᵀMv̂_j)²/(v̂_jᵀMv̂_j)`. This game is symmetric across players. Is there a symmetric Nash, all `v̂_i` equal? If `v̂_i = v̂` for everyone, then summing one reward and `k−1` penalties gives `u_i = (1 − (k−1))(v̂ᵀMv̂) = (2−k)(v̂ᵀMv̂) ≤ 0` for `k ≥ 2`. But any single player could deviate to a direction `v̂_⊥ ⟂ v̂` and get `u_i = v̂_⊥ᵀMv̂_⊥ > 0` whenever `rank(M) ≥ 2`. So no symmetric Nash exists. The PCA solution is *a* Nash of the symmetric game (same w-expansion gives `u_i = Λ_{ii}z_i`, uniquely maxed at `z_i=1`), but I cannot prove it is the *only* one — and certifying that no second Nash exists is NP-hard in general. The hierarchy isn't a cosmetic choice; it's what buys me **uniqueness** of the equilibrium and a clean inductive proof. (And solving for a Nash is PPAD-complete in the abstract, but a *hierarchical* game where each player depends only on its parents can be solved by sweeping the DAG in order — another reason the asymmetry pays off.)

Now the update. Player `i` does gradient ascent on `u_i`. Differentiate. The reward `v̂_iᵀMv̂_i` gives `2Mv̂_i`. For one penalty term `(v̂_iᵀMv̂_j)²/(v̂_jᵀMv̂_j)`, the denominator doesn't depend on `v̂_i`, and `∂/∂v̂_i (v̂_iᵀMv̂_j)² = 2(v̂_iᵀMv̂_j)·Mv̂_j`, so the gradient of the penalty is `2(v̂_iᵀMv̂_j)/(v̂_jᵀMv̂_j)·Mv̂_j`. Collect:
`∇_{v̂_i} u_i = 2Mv̂_i − 2Σ_{j<i} (v̂_iᵀMv̂_j)/(v̂_jᵀMv̂_j) Mv̂_j = 2M[ v̂_i − Σ_{j<i} (v̂_iᵀMv̂_j)/(v̂_jᵀMv̂_j) v̂_j ]`.
And there is the payoff for normalizing the penalty by the parent's Rayleigh quotient: the `M` factors *all the way out* to the front. The bracketed thing is `v̂_i` minus its projection onto each parent under the `M`-inner-product — a single step of **generalized Gram–Schmidt**, orthogonalizing `v̂_i` against the parents in the `⟨·,·⟩_M` geometry. The outer `2M` is exactly the matrix multiply at the heart of Oja's rule and power iteration. So the gradient of this hand-built utility *is* "Oja's rule applied after one generalized Gram–Schmidt step." I didn't put either ingredient in by hand; they fell out of differentiating a single clean objective. Sanity check on the picture: set `M = I` and iterate the fixed point `v̂_i ← ½∇_{v̂_i}u_i = v̂_i − Σ_{j<i}(v̂_iᵀv̂_j)/(v̂_jᵀv̂_j)v̂_j` in sequence — that is literally classical Gram–Schmidt orthogonalization. The method contains ordinary orthogonalization as the `M=I` special case.

In data form, `M = XᵀX` never appears explicitly:
`∇_{v̂_i} u_i = 2Xᵀ[ Xv̂_i − Σ_{j<i} ⟨Xv̂_i, Xv̂_j⟩/⟨Xv̂_j, Xv̂_j⟩ Xv̂_j ]`.
Two passes through the data per step (`X·`, then `Xᵀ·`), no `d×d` matrix, and the only thing a child needs from a parent is the vector `v̂_j` (or its projection `Xv̂_j`). That broadcast is the entire interconnect cost.

And now I can finally place GHA precisely. Suppose the first `i−1` parents are exact, `v̂_{j<i}=v_{j<i}`. GHA's update is `2[Mv̂_i − (v̂_iᵀMv̂_i)v̂_i − Σ_{j<i}(v̂_iᵀMv_j)v_j]`, and using `Mv_j=Λ_{jj}v_j` the last sum is `Σ_{j<i}Λ_{jj}(v̂_iᵀv_j)v_j`. My gradient, with only the *first* term projected onto the sphere's tangent space, is `2[(I−v̂_iv̂_iᵀ)Mv̂_i − MΣ_{j<i}(v̂_iᵀMv_j/v_jᵀMv_j)v_j]`. The penalty's ratio simplifies with exact parents to `(v̂_iᵀMv_j)/Λ_{jj} = v̂_iᵀv_j`, so `M·Σ(v̂_iᵀv_j)v_j = Σ Λ_{jj}(v̂_iᵀv_j)v_j`, and expanding the projector gives `2[Mv̂_i − (v̂_iᵀMv̂_i)v̂_i − Σ_{j<i}Λ_{jj}(v̂_iᵀv_j)v_j]` — *identical* to GHA. So GHA is precisely my gradient with the reward projected onto the sphere but the **penalty left unprojected**. That asymmetric projection is exactly why GHA fails to be the gradient of anything: it isn't projecting a single coherent vector field. My version is principled because every term comes from one `u_i`.

Optimization detail: `v̂_i` lives on the unit sphere `S^{d-1}`, a Riemannian manifold, not in flat space. The clean way to do constrained ascent is Riemannian: project the ambient gradient onto the tangent space, `∇^R_{v̂_i} = ∇_{v̂_i} − ⟨∇_{v̂_i}, v̂_i⟩v̂_i` (subtract the radial component — `(I − v̂_iv̂_iᵀ)∇`), step, then **retract** back to the sphere by renormalizing, `v̂_i ← v̂_i'/‖v̂_i'‖`. The projector and the renormalization are exactly the Riemannian operations the literature names; nice that they're already familiar from Oja's rule.

But let me think about whether to always project. Near the optimum the ambient gradient points almost radially outward from the sphere — nearly orthogonal to the tangent plane. When the gradient is nearly radial, the *tangential* component `∇^R` is small, yet the retraction (the renormalization after stepping) can still produce a surprisingly large net displacement, because the step plus the projection back can swing the point a long way around the sphere. That can overshoot and destabilize right where I want to settle. If instead I drop the projection and just step with the *ambient* gradient and renormalize, the radial part of the step gets eaten by the renormalization in a way that effectively *shrinks* the step as I approach equilibrium — a built-in learning-rate decay near the optimum. So I'll keep two variants: the Riemannian-projected one (call it the `R` variant) for which I can state the cleanest theory, and the unprojected one (step with ambient `∇`, then renormalize) which is often more stable and faster in practice. Same code modulo one line.

Sequential vs parallel. The convergence story is cleanest if I learn the parents *first* and freeze them, so each child maximizes a stationary objective — learn `v̂_1` to tolerance, stop, learn `v̂_2`, and so on down the DAG. But that throws away the parallelism that motivated everything. In practice I want every player updating at once, each on its own device. Is that safe? As a parent nears its optimum it becomes quasi-stationary — its vector stops moving — so a child maximizing against a slowly-drifting parent is close to maximizing against a fixed one. So I'll run them in parallel and broadcast after each step, even though I can only *prove* the sequential version. (One honest caveat to note for later: stochastic minibatch gradients of this utility are biased, because the utility contains products and ratios of inner products of the same batch; larger batches reduce the bias.)

Now nail the convergence of the sequential version, because the proof is what tells me how many iterations each player needs and how accurately the parents must be learned. I'll lean on a nonconvex Riemannian rate (Boumal–Absil–Cartis 2019): generic Riemannian descent with constant step reaches `‖∇^R f(x)‖ ≤ ρ` in `⌈(f(x₀)−f*)/ξ · 1/ρ²⌉` iterations, given `f` bounded below and a sufficient-decrease condition. I'm maximizing, so I flip signs.

First, what does the utility landscape even look like along a deviation? Parameterize `v̂_i` on the sphere by an angle from its true eigenvector: `v̂_i = cos(θ_i)v_i + sin(θ_i)Δ_i`, with `Δ_i` a unit vector orthogonal to `v_i` giving the deviation *direction* and `θ_i` the deviation *magnitude*. Suppose for now the parents are exact. Then
`u_i(v̂_i, v_{j<i}) = ⟨v̂_i, Λv̂_i⟩ − Σ_{j<i}Λ_{jj}⟨v̂_i, v_j⟩²`.
Substitute. The reward: `⟨v̂_i, Λv̂_i⟩ = cos²(θ_i)Λ_{ii} + sin²(θ_i)⟨Δ_i, ΛΔ_i⟩` (the cross term `2cos·sin·⟨v_i, ΛΔ_i⟩` vanishes because `ΛΔ_i` keeps `Δ_i⟂v_i` once `Δ_i` is expanded in eigenvectors and `⟨Δ_i,v_i⟩=0`). The penalty: `⟨v̂_i, v_j⟩ = cos(θ_i)⟨v_i,v_j⟩ + sin(θ_i)⟨Δ_i,v_j⟩ = sin(θ_i)⟨Δ_i,v_j⟩` for `j<i` (since `⟨v_i,v_j⟩=0`), so `Σ_{j<i}Λ_{jj}⟨v̂_i,v_j⟩² = sin²(θ_i)Σ_{j<i}Λ_{jj}⟨Δ_i,v_j⟩²`. Collect using `cos²=1−sin²`:
`u_i = Λ_{ii} − sin²(θ_i)Λ_{ii} + sin²(θ_i)[⟨Δ_i,ΛΔ_i⟩ − Σ_{j<i}Λ_{jj}⟨Δ_i,v_j⟩²]`.
The bracket is just `u_i(Δ_i, v_{j<i})`, and since `Δ_i⟂v_i` the simplex argument from the Nash proof gives `u_i(Δ_i, v_{j<i}) = Σ_{l>i} z_l Λ_{ll}` (component `i` is absent because `⟨Δ_i,v_i⟩=0` forces `z_i=0`). So
`u_i(v̂_i, v_{j<i}) = Λ_{ii} − sin²(θ_i)(Λ_{ii} − Σ_{l>i} z_l Λ_{ll})`.
A **sinusoid** in `θ_i`: peak `Λ_{ii}` at `θ_i=0`, period `π` (not `2π`, because `v_i` and `−v_i` are the same component). The amplitude depends on the deviation direction through `z`, but along any fixed direction it's a clean sinusoid — which means it is non-concave, yet *every local max is a global max*. That's the property that lets me bound iterations: no spurious local optima to get stuck in, only the trough.

But parents are *not* exact in general; the parent error mis-specifies the child's utility, and I need to know how much that perturbs the child's maximizer. Let both child and parents deviate, `v̂_j = cos(θ_j)v_j + sin(θ_j)Δ_j` for all `j ≤ i`. Substituting into `u_i = ⟨v̂_i, Λv̂_i⟩ − Σ_{j<i}⟨v̂_i, Λv̂_j⟩²/⟨v̂_j, Λv̂_j⟩` and grinding through the algebra (expanding each `⟨cos·v + sin·Δ, Λ(cos·v + sin·Δ)⟩`, using `2sin·cos = sin(2θ)`), the whole thing collapses to the form
`u_i(v̂_i, v̂_{j<i}) = A·sin²(θ_i) − B·(sin(2θ_i)/2) + C`,
where `A, B, C` are messy functions of the *parents'* deviations `(θ_j, Δ_j)` and the child's deviation direction `Δ_i`, but crucially **not** of `θ_i`. Using `sin²θ = (1−cos2θ)/2`, rewrite as a single phase-shifted cosine:
`u_i = ½[ −A cos(2θ_i) − B sin(2θ_i) + A + 2C ] = ½[ √(A²+B²) cos(2θ_i + φ) + A + 2C ]`, with `φ = tan⁻¹(B/A)`.
Still sinusoidal in `θ_i`, period `π`, single global max. Where is the maximizer? Set `∂u_i/∂θ_i = A sin(2θ_i) − B cos(2θ_i) = 0`, i.e. `tan(2θ_i) = B/A`. The second derivative's sign is `sign(cos 2θ_i)·sign(A)`, so for a maximum I need that negative; working it through, when `A<0` the maximizer magnitude is `|θ_i*| = ½ tan⁻¹|B/A|`, when `A>0` it is `½[π − tan⁻¹|B/A|]`, and `π/4` when `A=0`. The `arctan` makes this a **soft step**: while the parents are accurate the argument `|B/A|` stays small and the child's maximizer sits near `θ_i*≈0` (near the true eigenvector); once the parents degrade past a threshold, `A` flips sign and the maximizer jumps away. So there is a sharp accuracy threshold on the parents below which the child can recover its eigenvector and above which it cannot.

Make the threshold quantitative. Assume `|θ_j| ≤ ε` for all parents `j<i`. Bounding `A` (with `g_i = Λ_{ii} − Λ_{i+1,i+1}` the eigenvalue gap, `κ_i = Λ_{11}/Λ_{ii}` the condition number) gives, after the trig and Cauchy–Schwarz work,
`A ≤ −g_i + (i−1)(Λ_{11}+Λ_{ii})·ε²/(1−ε²) + 2(i−1)Λ_{11}·ε/√(1−ε²)`.
The leading `−g_i` is the "good" term (negative, so the child has a max near `θ_i=0`); the others are error injected by the parents and grow with `ε`. To keep `A < 0` and the argument `|B/A| ≤ ½` (so that `tan⁻¹` is in its near-linear regime), it suffices to demand the parents be learned to within a *fraction* of a canonical threshold:
`|θ_j| ≤ c_i · g_i/((i−1)Λ_{11})` with `c_i ≤ 1/16`.
Then `A < 0`, and `|θ_i*| = ½ tan⁻¹|B/A| ≤ ½|B/A| ≤ 8c_i`. So the child's maximizer is within `8c_i` of its true eigenvector — the parent error propagates to the child *amplified* by the relation above. The `1/16` is just the conservative constant that keeps everything in the linear regime of the `arctan`.

Two more ingredients to satisfy Boumal's assumptions. (1) Boundedness: bound the ambient gradient norm `‖∇_{v̂_i}u_i‖ ≤ 2‖Mv̂_i‖ + 2Σ_{j<i}|ratio_j|‖Mv̂_j‖`. Each ratio `⟨v̂_i,Λv̂_j⟩/⟨v̂_j,Λv̂_j⟩` is bounded (expand both in the angle parameterization) by `(1+(1+κ_j)ε)/√(1−ε²)`, giving `‖∇_{v̂_i}u_i‖ ≤ 2Λ_{11}[1 + (i−1)(1+(1+κ_{i-1})ε)/√(1−ε²)]`, and with accurate parents this tightens to `‖∇‖ ≤ 4[Λ_{11}i + (1+κ_{i-1})c_i g_i] =: L_i`. Since `u_i` is degree-2 homogeneous in `v̂_i` (both reward and penalty are quadratic in `v̂_i`), Euler gives `v̂_iᵀ∇_{v̂_i}u_i = 2u_i`, so `|u_i| = ½|v̂_iᵀ∇_{v̂_i}u_i| ≤ ½‖v̂_i‖‖∇_{v̂_i}u_i‖ ≤ L_i` (since `‖v̂_i‖=1`) — boundedness done. (2) Sufficient decrease: with step `α = 1/(2L_i)`, one ascent step satisfies `u_i(v̂_i') − u_i(v̂_i) ≥ (2αz²/(1+α²z²))(1−αL_i) > 0` (working in the 2-D plane spanned by `v̂_i` and `∇`, since the iterate stays in it), and at `α=1/(2L_i)` this gives `≥ (8/5)L_i‖η‖²·(…)` so the descent-Lipschitz holds with `ξ = ξ' = (8/5)L_i`. Both assumptions met.

Plug in. To get `‖∇^R‖ ≤ ρ_i`, Boumal needs `⌈(5/4)·1/ρ_i²⌉` iterations (the `5/4` is `1/ξ` after the constants), and a small lemma relates Riemannian-gradient norm to angular error, `|θ_i − θ_i*| ≤ (π/g_i)‖∇^R‖`, so after that many iterations the child is within `(π/g_i)ρ_i + 8c_i` of its eigenvector — convergence error plus inherited parent error. Now chain it up the DAG. Each parent must be learned accurately enough for its child, which strengthens the requirement hop by hop: recursing `c_{i-1} ≤ c_i g_i/(16(i−1)Λ_{11})` down to the root gives
`c_i ≤ [(i−1)! Π_{j=i+1}^{k} g_j] / [(16Λ_{11})^{k-i}(k−1)!] · c_k`,
so the *first* eigenvector has to be learned to extreme accuracy to enable the `k`-th — the `(k−1)!` blowup. Summing the per-player iteration counts, the total to get all `k` within tolerance `θ_tol` is
`T_k = ⌈ O( k · [ (16Λ_{11})^k (k−1)! / (Π_{j=1}^{k} g_j) · 1/θ_tol ]² ) ⌉`.
Read it off: the outer `k` is from naively summing `k` worst-case per-player bounds; the `1/θ_tol²` is the `1/ρ²` rate; each `Λ_{11}/g_j` says small gaps relative to the spectral scale need more iterations to resolve; the `(k−1)!` is the up-the-chain accuracy amplification; the `16` is the conservative error-propagation constant. And this is **global** — independent of initialization — because the per-direction utility is a single-basin sinusoid: even from a bad start there are no spurious maxima to trap the iterate, only a possible trough where the gradient is small. The algorithm handles the trough by sizing its iteration budget from the *observed* initial gradient norm — `t_i = ⌈(5/4)·min(‖∇_{v̂_i⁰}u_i‖/2, ρ_i)⁻²⌉` — so if it starts near a flat trough it simply runs longer. (In high dimension a random init is, with overwhelming probability, near `π/2` from the true eigenvector — all points are far apart on a high-dimensional sphere, by the incomplete-beta cap-probability `I_{sin²φ}((d−1)/2, 1/2) → 0` — which is exactly why the global rate, not just the lucky-init rate, matters.)

Let me write the algorithm concretely, mirroring exactly what I derived. Per player `i`: compute `Xv̂_i` (the rewards signal), compute the penalty `Σ_{j<i} (⟨Xv̂_i,Xv̂_j⟩/⟨Xv̂_j,Xv̂_j⟩)Xv̂_j`, form `∇ = 2Xᵀ[rewards − penalties]`, optionally project to the tangent space, step, renormalize, broadcast. In full-batch form I can vectorize across all `k` at once. Let `V̂ ∈ R^{d×k}`. Then `R = (XV̂)ᵀ(XV̂)` is the `k×k` Gram of the projected vectors (`R_{ij} = ⟨Xv̂_i,Xv̂_j⟩`). Normalize each column of `R` by its diagonal, `R_norm = R / diag(R)` (so entry `(i,j)` is `⟨Xv̂_i,Xv̂_j⟩/⟨Xv̂_j,Xv̂_j⟩`, and the diagonal is all ones). I want, for child `i`, the bracket `v̂_i − Σ_{j<i}(…)v̂_j`: keep `+v̂_i` from the diagonal and subtract the parent terms `j<i`. The lower-triangular mask `mask = LT(2I_k − 1_k)` does exactly this — `2I_k − 1_k` is `+1` on the diagonal and `−1` off it, and taking the lower triangle (diagonal included) gives `+1` on the diagonal (the reward) and `−1` on the strict lower triangle (the parents), zero above. Then `G_s = V̂(R_norm ⊙ mask)ᵀ` assembles, for each `i`, the vector `v̂_i − Σ_{j<i}(…)v̂_j` (the generalized-Gram–Schmidt bracket). The full ambient gradient is `∇ = Xᵀ(XG_s)`. Riemannian projection: subtract each column's radial part, `∇^R = ∇ − V̂·sum(∇⊙V̂, axis=0)`. Step `V̂ ← V̂ + α∇^R`, then column-normalize `V̂ ← V̂/‖V̂‖_col`. That's it — no `QR`, no SVD, no `d×d` matrix; the only `k×k` object is `R`, and the only coupling between vectors is the masked `R_norm`, which is exactly the parent broadcasts.

```python
import numpy as np

def reference_components(X, k):
    """Exact eigenvectors of M = X^T X, descending — for validation only."""
    w, U = np.linalg.eigh(X.T @ X)
    order = np.argsort(w)[::-1]
    return U[:, order[:k]], w[order[:k]]

def normalize_columns(V):
    return V / np.linalg.norm(V, axis=0, keepdims=True)

def utility(X, V):
    """u_i = ||X v_i||^2 - sum_{j<i} <Xv_i,Xv_j>^2 / <Xv_j,Xv_j>  (per column)."""
    XV = X @ V                                  # (n, k)
    R = XV.T @ XV                               # R_ij = <Xv_i, Xv_j>
    rewards = np.diag(R)                        # variance along each v_i
    k = V.shape[1]
    pen = np.array([sum(R[i, j] ** 2 / R[j, j] for j in range(i))
                    for i in range(k)])
    return rewards - pen

# --- one synchronous (parallel) sweep: every player takes one ascent step ----
def eigengame_step(X, V, lr, riemannian=True):
    """Each column v_i ascends its own utility u_i(v_i | v_{j<i}).
    Gradient = 2 X^T [ X v_i  -  sum_{j<i} (<Xv_i,Xv_j>/<Xv_j,Xv_j>) X v_j ].
    The bracket is one generalized Gram-Schmidt step; the outer X^T X is Oja.
    """
    k = V.shape[1]
    XV = X @ V                                  # (n, k)  -- the "rewards" signal
    R = XV.T @ XV                               # (k, k)  <Xv_i, Xv_j>
    R_norm = R / np.diag(R)                     # divide each col by parent's Rayleigh
    mask = np.tril(2 * np.eye(k) - np.ones((k, k)))   # diag +1 (reward), strict-lower -1 (parents)
    G_s = V @ (R_norm * mask).T                 # column i: v_i - sum_{j<i} (...) v_j
    grad = 2.0 * (X.T @ (X @ G_s))             # 2 X^T X [ v_i - generalized-GS ]
    if riemannian:                              # project onto sphere tangent space
        grad = grad - V * np.sum(grad * V, axis=0, keepdims=True)
    V = V + lr * grad                           # ascent step
    return normalize_columns(V)                 # retract back to the sphere

def eigengame(X, k, lr=1e-4, iters=5000, riemannian=True, V0=None):
    d = X.shape[1]
    V = normalize_columns(np.random.randn(d, k)) if V0 is None else V0.copy()
    for _ in range(iters):                      # in parallel: one device per column
        V = eigengame_step(X, V, lr, riemannian)
    return V

# --- streaming variant: same step, fresh minibatch each iteration ------------
def eigengame_streaming(stream, d, k, lr=1e-4, riemannian=False, V0=None):
    V = normalize_columns(np.random.randn(d, k)) if V0 is None else V0.copy()
    for Xt in stream:                           # Xt : (m, d) minibatch
        V = eigengame_step(Xt, V, lr, riemannian)   # broadcast(v_i) is the only comm
    return V

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 30))
    k = 6
    V = eigengame(X, k, lr=3e-4, iters=8000, riemannian=False)
    V_true, w_true = reference_components(X, k)
    cos = np.abs(np.sum(V * V_true, axis=0))    # |cos angle| to true component
    print("alignment per component:", np.round(cos, 4))
    print("recovered Rayleigh quotients:", np.round(np.diag((X @ V).T @ (X @ V)), 3))
    print("true eigenvalues:           ", np.round(w_true, 3))
```

The whole chain: PCA wants the ordered *components*, not just the subspace; trace-maximization is blind to rotation (it equals `Tr(M)`), so I add an off-diagonal penalty to force `R` diagonal; but a symmetric penalty fights the natural ordering, so I make each vector penalize only its parents — which turns one optimization into a `k`-player game with asymmetric utilities; the top-k eigenvectors are that game's unique strict Nash (a tower of simplex LPs); the gradient of each utility is automatically Oja's rule after one generalized Gram–Schmidt step, with `M=I` recovering ordinary Gram–Schmidt; the per-direction utility is a single-basin sinusoid, so Riemannian gradient ascent converges globally, with parent accuracy propagating to children through an `arctan` soft-step threshold that fixes how hard the top vectors must be learned; and because every player's update depends only on its parents' broadcast vectors — no `QR`, no SVD, no `d×d` matrix — the procedure runs one eigenvector per device and scales to data that no SVD could touch.
