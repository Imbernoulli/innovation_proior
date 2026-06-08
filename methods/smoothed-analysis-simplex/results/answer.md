# Smoothed analysis of the simplex method

## The problem

The simplex method for linear programming runs in roughly linear time in practice but takes
exponentially many pivots on contrived inputs (Klee–Minty cubes). Worst-case analysis brands it
intractable because its `max` over inputs locks onto those brittle instances; average-case
analysis (Borgwardt, Smale, Megiddo) proves it fast only on rotationally symmetric random inputs
that do not resemble real linear programs. Neither explains the everyday speed.

## The key idea: smoothed complexity

Analyze the **maximum over inputs of the expected running time under a small Gaussian
perturbation of the input.** This keeps an adversary (the `max`) but forbids it from being
infinitely precise (the perturbation). It interpolates between the two classical analyses: as the
perturbation vanishes it becomes worst-case, as the perturbation swamps the input it becomes
average-case.

**Definition (smoothed complexity).** For an algorithm `A` with complexity measure `C_A`, input
domain `X` with size-`n` inputs `X_n`, and Gaussian perturbations of standard deviation
`σ‖x‖` (with `g` a vector of independent standard Gaussians),

    Smoothed_A(n, σ) = max_{x ∈ X_n}  E_g [ C_A( x + (σ‖x‖) g ) ].

`A` has **polynomial smoothed complexity** if this is bounded by a polynomial in `n` and `1/σ`.
(Polynomial in `1/σ`, not `log(1/σ)`: a `poly(d,n,log 1/σ)` bound would convert into a worst-case
`poly(d,n,L)` bound, collapsing the notion back to worst-case-polynomial.)

## The shadow-vertex pivot rule

For a linear program `max zᵀx s.t. Ax ≤ y` (A is n×d), pick a second objective `t` optimized by a
known start vertex. Project the feasible polyhedron onto the plane `span(t, z)`; the projection
(the **shadow**) is a convex polygon whose vertices/edges are images of polyhedron vertices/edges,
and which contains the images of both the start vertex and the optimum. Rotating the objective
`q_λ = (1−λ)t + λz` from `λ=0` to `λ=1` and tracking `optVert(q_λ)` walks the shadow boundary. The
number of pivots is at most the number of shadow edges. Among pivot rules, this is the only one
whose visited-vertex sequence is a closed-form geometric object (a projection) rather than an
iteration — which is what makes a probabilistic analysis tractable.

## The shadow-size theorem (geometric heart)

**Theorem (shadow size).** Let `d ≥ 3`, `n > d`, let `z, t` be independent, and let `a_1,…,a_n`
be Gaussian in ℝᵈ of standard deviation `σ` centered at points of norm at most 1. Then

    E[ |Shadow_{t,z}(a_1,…,a_n)| ]  ≤  58,888,678 · n d³ / min(σ, 1/(3√(d ln n)))⁶.

So the expected shadow size is **O(n d³ / σ⁶)** — polynomial in `n`, `d`, `1/σ`.

**Proof architecture.**
1. WLOG `σ ≤ 1/(3√(d ln n))` (scale data down) and `z ⊥ t`. Parametrize the plane by angle:
   `q_θ = z sinθ + t cosθ`.
2. Discretize the circle into `m` angles; in expectation, the shadow edge count is the limit of
   the sum of the adjacent-angle basis-change indicators (the limit is clean: bases optimal on a
   zero-length angular interval form a measure-zero, codimension-1 set).
3. A basis change at `θ` ⟹ `q_θ` is within angle `2π/m` of the boundary of its optimal simplex.
   So `E[|shadow|] ≤ lim_m Σ_i P[ ang_{q_θ} < 2π/m ]`. If `P[ang_q < ε] ≤ K ε` then the `m`'s
   cancel and `E[|shadow|] ≤ 2πK`. The whole bound reduces to a **linear-in-ε angle bound**.
4. Restrict to the high-probability event `P = {all ‖a_i‖ ≤ 2}` (`P[¬P] ≤ binom(n,d)^{-1}`,
   costing `+1`). Union over bases `I` and over the `d` facets of each, then Bayes: since only one
   basis is optimal at a time, `Σ_I P[I optimal] ≤ 1`, so it suffices to bound, conditioned on `I`
   optimal, the probability that the angle from `q` to the facet `simp(A_{I−j})` is `< ε`.
5. **Blaschke change of variables**: describe the `d` base points by `(ω, s, b_1,…,b_d)` (plane
   normal/offset + in-plane positions after rotating the plane normal to `q`). The Jacobian is
   `(d−1)! ⟨ω,q⟩ vol(simp(b))`, and after conditioning on optimality the density also contains the
   `d` basis-point Gaussian densities and the `n−d` half-space probabilities for the non-basis
   constraints. The angle splits as
   `ang(q, facet) ≥ dist(0, aff(b_2,…,b_d)) · ⟨ω,q⟩ / (2 + 4√2)`, so the small-angle event is a
   small-product event.
6. **Combination lemma**: if `P[F ≤ ε] ≤ αε` and `max_x P[G ≤ ε] ≤ (βε)²`, then `P[FG ≤ ε] ≤ 4αβε`
   (dyadic decomposition; the square on `G` makes the geometric series converge). Need a *linear*
   distance bound and a *quadratic* angle bound:
   - **Distance** (`F`): `P[ dist(0, aff(b_2,…,b_d)) < ε ] ≤ 900 e^{2/3} d² ε / σ⁴`. Route:
     perturbed simplices are fat — `P[dist(b_1, aff(b_2..b_d)) < ε] ≤ (3 e^{2/3} d ε/σ²)³` (cubic,
     reinforced by the `vol(simp)` Jacobian weight) — and the ratio
     `dist(0, aff(b_2..b_d)) / dist(b_1, aff(b_2..b_d))` is small with probability at most
     `75dε/σ²` by contracting the simplex toward `b_1` and using Gaussian smoothness.
   - **Angle of incidence** (`G`): `max P_ω[ ⟨ω,q⟩ < ε ] ≤ (340 n ε/σ²)²`. Route: latitude/
     longitude change of variables `c = ⟨ω,q⟩`, Jacobian `(1−c²)^{(d−3)/2}`; on `[0, c_0]`,
     `c_0 = σ²/(240n)`, the non-`c` part of the density varies by ≤2, so
     `P[c < ε] ≤ 2(ε/c_0)²`. The `n` enters because each of the `n−d` side constraints'
     half-space-cut probability shifts as `ω` tilts.
7. Combine: conditional angle prob `≤ const · n d²/σ⁶ · ε`; union over the `d` facets and the `2π`
   discretization give `E[|shadow|] = O(n d³/σ⁶)`. The exponent is accounted for as
   `σ⁶ = σ⁴·σ²`: the distance factor contributes `σ⁴`, while the quadratic incidence bound
   contributes a single `σ²` after the combination lemma uses its square root rate.

**Extensions** (rescaling + Gaussian-additivity splitting): drop `‖ā_i‖ ≤ 1` (replace `σ` by
`σ/max‖ā_i‖`), allow general covariances (eigenvalues in `[σ², 1/(9d ln n)]`), and general
positive `y` (rescale `a_i/y_i`) — each costing a constant or `+1`.

## The main theorem (running time)

**Theorem (main).** There is a polynomial `P` and a constant `σ_0` such that for all `n > d ≥ 3`,
all `ā ∈ ℝ^{n×d}`, `ȳ ∈ ℝⁿ`, `z ∈ ℝᵈ`, and `σ > 0`, the two-phase shadow-vertex simplex method
takes expected pivots

    E_{A,y}[ C(A, y, z) ]  ≤  min( P(d, n, 1/min(σ, σ_0)),  binom(n,d) + binom(n,d+1) + 2 ),

where `A`, `y` are Gaussian perturbations of `ā`, `ȳ` of standard deviation
`σ·max_i‖(ȳ_i, ā_i)‖`. So the simplex method has **polynomial smoothed complexity**.

**Why the bare shadow bound doesn't suffice, and the fixes.**
- The projection plane depends on the data. *Fix:* split the perturbation into two (Gaussians
  add); choose the plane from the first perturbation, show it is well-conditioned w.h.p., so the
  small second perturbation barely moves it; apply the fixed-plane shadow bound to the second.
- Need a feasible start, infeasibility detection, and `y_i > 0`. *Fix:* a two-phase construction —
  relax to `LP'` with a known feasible basis `I` and positive right-hand sides, solve it, then
  solve `LP⁺` (one extra variable interpolating `LP' ↔ LP`) to transform the solution.
- If `s_min(A_I)` is tiny the construction's constants `M, κ` blow up. *Fix:* **many-good-choices
  lemma** — it is exponentially unlikely that almost all `d`-minors are ill-conditioned, so the
  best of `3nd ln n` random `d`-sets has `s_min(A_I) > κ_0 = σ min(1,σ)/(12 d² n⁷ √ln n)` with
  probability `≥ 1 − 0.417 binom(n,d)^{-1}`.

The optimized constants are not pursued; the concrete pivot polynomial is huge (on the order of
`d^55 n^86 σ^{-30} + d^70 n^86`), but the intuition lives in the clean shadow bound `O(n d³/σ⁶)`.

## Final algorithmic artifact

Given `A, y, z`, the two-phase shadow-vertex method is:

1. Draw `3nd ln n` random `d`-subsets of `[n]`; choose `I` maximizing `s_min(A_I)`.
2. Set `M = 2^{ceil(lg max_i ||(y_i,a_i)||)+2}` and `κ = 2^{floor(lg s_min(A_I))}`.
3. Define `y'_i = M` for `i ∈ I`, and `y'_i = sqrt(d) M²/(4κ)` otherwise. This gives `LP'`:
   maximize `z^T x` subject to `a_i^T x ≤ y'_i`.
4. Choose `α` uniformly from `{α : sum_i α_i = 1, α_i ≥ 1/d²}` and set the start objective
   `t_0 = A_I α`. The basis `I` is feasible for `LP'` and optimizes every objective `A_I α` with
   `α > 0`.
5. Run the polar shadow-vertex method on `LP'` from `(I, t_0)`. If it reports unboundedness,
   return unbounded.
6. Build `LP⁺` in variables `(x_0, x)`:
   `a_i^+ = ((y'_i - y_i)/2, a_i)`, `y_i^+ = (y'_i + y_i)/2` for `1 ≤ i ≤ n`,
   plus `a_0^+ = (1,0,...,0)`, `a_{-1}^+ = (-1,0,...,0)`, and `y_0^+ = y_{-1}^+ = 1`.
   The objective is `z^+ = (1,0,...,0)`, so `x_0 = -1` is `LP'` and `x_0 = 1` is the original LP.
7. Let `J` be the basis returned for `LP'`. Choose `ζ > 0` so `{-1} ∪ J` is an initial optimal
   simplex for objective `(-ζ, z)` in `LP⁺`, and run the polar shadow-vertex method on `LP⁺`
   from that basis.
8. Recover `(x_0, x)` from the returned basis. If `x_0 < 1`, return infeasible; otherwise return
   `x` as an optimum of the original LP.

The analysis bounds the expected number of pivots in step 5 by the `LP'` shadow bound after
splitting the perturbation and fixing the plane relative to the second perturbation. It bounds
step 7 by approximating the ratio-distributed points `a_i^+/y_i^+` locally by Gaussians, applying
the extended shadow bound, and charging the small exceptional set to the trivial cap.
