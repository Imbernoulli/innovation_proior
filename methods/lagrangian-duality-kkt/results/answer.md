# Lagrangian duality and the KKT conditions

## Problem

Constrained optimization in standard form:

```
minimize    f(x)
subject to  g_i(x) ≤ 0,   i = 1,…,m
            h_j(x) = 0,   j = 1,…,p
```

with variable x ∈ ℝⁿ and differentiable f, g_i, h_j; optimal value p⋆. No convexity is assumed
unless stated. In the Slater proof below, D denotes the convex domain; for the basic ℝⁿ setting,
D = ℝⁿ.

## Key idea

Fold the constraints into the objective with multipliers, forming the **Lagrangian**

  L(x, λ, ν) = f(x) + Σ_i λ_i g_i(x) + Σ_j ν_j h_j(x),   λ ∈ ℝᵐ, ν ∈ ℝᵖ.

The multipliers act as the slopes of *linear under-estimators* of the "infinitely hard" constraint
penalties: the inequality wall (0 if g_i ≤ 0, +∞ if g_i > 0) is under-cut only by a line of
**nonnegative** slope, forcing **λ ⪰ 0**; the equality wall is under-cut by a line of any slope, so
**ν is free**. The **dual function** d(λ,ν) = inf_x L(x,λ,ν) is then a lower bound on p⋆, and
maximizing it over λ ⪰ 0 is the dual problem.

## Weak duality (always)

For any λ ⪰ 0 and any ν: **d(λ,ν) ≤ p⋆.**

*Proof.* Let x̃ be feasible. Since λ ⪰ 0 and g_i(x̃) ≤ 0, Σ_i λ_i g_i(x̃) ≤ 0; since h_j(x̃) = 0,
Σ_j ν_j h_j(x̃) = 0. Hence L(x̃,λ,ν) ≤ f(x̃), and d(λ,ν) = inf_x L(x,λ,ν) ≤ L(x̃,λ,ν) ≤ f(x̃).
Taking the infimum over feasible x̃ gives d(λ,ν) ≤ p⋆. ∎

The dual function is concave (pointwise infimum of affine functions of (λ,ν)), so the **dual
problem** maximize d(λ,ν) s.t. λ ⪰ 0 is convex with optimal value d⋆, **whatever the primal**, and

  d⋆ ≤ p⋆   (weak duality).

Equivalently, with sup_{λ⪰0,ν∈ℝᵖ} L(x,λ,ν) = f(x) on the feasible set and +∞ otherwise,
p⋆ = inf_x sup_{λ⪰0,ν∈ℝᵖ} L and d⋆ = sup_{λ⪰0,ν∈ℝᵖ} inf_x L, so weak duality is the max–min
inequality sup inf ≤ inf sup.

## Strong duality under Slater (convex case)

Assume f, g_i convex, the equality constraints are affine (Ax = b) with redundant equality rows
removed so A has full row rank, p⋆ finite, and **Slater's condition**: there exists x̃ ∈ relint D
with

  g_i(x̃) < 0 (i = 1,…,m),   Ax̃ = b

Then **d⋆ = p⋆** and the dual optimum is attained.

*Proof (separating hyperplane).* Define the convex value set
  𝒜 = {(u,v,t) : ∃ x ∈ D, g_i(x) ≤ u_i, Ax − b = v, f(x) ≤ t}
and ℬ = {(0,0,s) : s < p⋆}. They are disjoint (a common point gives a feasible x with f(x) < p⋆).
By the separating-hyperplane theorem there exist (λ̃, ν̃, μ) ≠ 0 and α with
  λ̃ᵀu + ν̃ᵀv + μt ≥ α on 𝒜,   λ̃ᵀu + ν̃ᵀv + μt ≤ α on ℬ.
Since 𝒜 is unbounded in the +u_i and +t directions, λ̃ ⪰ 0 and μ ≥ 0 (else the left side is
unbounded below on 𝒜). The ℬ-side gives μp⋆ ≤ α. Combining, for all x ∈ D,

  Σ_i λ̃_i g_i(x) + ν̃ᵀ(Ax − b) + μ f(x) ≥ μ p⋆.   (◇)

If **μ > 0**, divide by μ and set λ = λ̃/μ ⪰ 0, ν = ν̃/μ: L(x,λ,ν) ≥ p⋆ for all x, so
d(λ,ν) = inf_x L ≥ p⋆; with weak duality d(λ,ν) ≤ p⋆, hence d(λ,ν) = p⋆. If **μ = 0**, (◇) gives
Σ_i λ̃_i g_i(x) + ν̃ᵀ(Ax − b) ≥ 0 for all x ∈ D; at the strictly feasible x̃, Ax̃=b and each
λ̃_i g_i(x̃) ≤ 0 with g_i(x̃) < 0, forcing λ̃ = 0. Then ν̃ ≠ 0 and ν̃ᵀ(Ax − b) ≥ 0 for all x ∈ D.
Since x̃ ∈ relint D, that linear function cannot have a one-sided minimum at x̃ unless its gradient
vanishes on the affine hull of D; in the full-dimensional case this is Aᵀν̃ = 0, and otherwise it is
the same rank condition after restricting A to that affine hull. Full row rank after removing
redundant equality rows contradicts ν̃ ≠ 0. So μ > 0 and strong duality holds. ∎

Geometrically, the duality gap p⋆ − d⋆ is the vertical distance from (0,p⋆) to the supporting
hyperplane of 𝒜; convexity makes 𝒜 convex so a supporting hyperplane exists, and Slater forbids a
*vertical* one (μ = 0), which forces the gap to zero.

A standard refinement allows affine inequality constraints to be tight at x̃ because their flat
walls cannot create the curvature failure that strict feasibility rules out.

## The KKT conditions

Let f, g_i, h_j be differentiable. If strong duality holds and x⋆, (λ⋆,ν⋆) are primal/dual optimal,
the chain

  f(x⋆) = d(λ⋆,ν⋆) = inf_x L(x,λ⋆,ν⋆) ≤ L(x⋆,λ⋆,ν⋆) ≤ f(x⋆)

collapses to equalities, yielding the **Karush–Kuhn–Tucker conditions**:

  (stationarity)              ∇f(x⋆) + Σ_i λ⋆_i ∇g_i(x⋆) + Σ_j ν⋆_j ∇h_j(x⋆) = 0
  (primal feasibility)        g_i(x⋆) ≤ 0,   h_j(x⋆) = 0
  (dual feasibility)          λ⋆_i ≥ 0
  (complementary slackness)   λ⋆_i g_i(x⋆) = 0,   i = 1,…,m

Stationarity is "x⋆ minimizes L(·,λ⋆,ν⋆)"; complementary slackness follows because Σ_i λ⋆_i g_i(x⋆)
is a sum of nonpositive terms equal to zero, so each term vanishes (λ⋆_i > 0 ⇒ g_i active;
g_i slack ⇒ λ⋆_i = 0).

- **Necessity.** For the differentiable standard form above, if strong duality holds and primal and
  dual optima are attained, every primal–dual optimal pair satisfies the KKT conditions.
- **Sufficiency (convex case).** If f, g_i are convex and h_j affine, and (x̃,λ̃,ν̃) satisfy the
  four KKT conditions, then x̃ and (λ̃,ν̃) are primal/dual optimal with zero gap: λ̃ ⪰ 0 makes
  L(·,λ̃,ν̃) convex, stationarity makes x̃ its global minimizer, and complementary slackness +
  feasibility give d(λ̃,ν̃) = L(x̃,λ̃,ν̃) = f(x̃).
- **Convex + Slater ⇒ KKT is necessary and sufficient:** x is optimal iff some (λ,ν) closes the
  KKT conditions with it.

A first-order **constraint qualification** (Slater for convex problems; in the general
differentiable setting, that every linearized feasible direction be tangent to an actual feasible
arc — equivalently a regularity condition such as linear independence of active constraint
gradients) is required for necessity: at an irregular boundary point (e.g. a cusp) no multipliers
need exist.

For the cusp region (1−x₁)³ − x₂ ≥ 0, x₁ ≥ 0, x₂ ≥ 0 at (1,0), write the curved constraint as
g=x₂−(1−x₁)³≤0. Then ∇g(1,0)=(0,1), and the active nonnegativity wall −x₂≤0 has gradient (0,−1).
The linearized cone admits d=(1,0), but no feasible arc has that derivative because x₁>1 would force
(1−x₁)³<0 while feasibility requires 0≤x₂≤(1−x₁)³. For f=−x₁, (1,0) is optimal, yet stationarity
fails because active gradients have zero first component and cannot cancel ∇f(1,0)=(-1,0).

## Saddle-point / minimax form

Strong duality is the statement that the order of optimization may be exchanged,

  sup_{λ ⪰ 0, ν ∈ ℝᵖ} inf_x L(x,λ,ν) = inf_x sup_{λ ⪰ 0, ν ∈ ℝᵖ} L(x,λ,ν),

and the common value is attained at a **saddle point** (x⋆, λ⋆, ν⋆):

  L(x⋆, λ, ν) ≤ L(x⋆, λ⋆, ν⋆) ≤ L(x, λ⋆, ν⋆)   for all x, all λ ⪰ 0, and all ν ∈ ℝᵖ.

The Lagrangian's saddle point simultaneously certifies the primal optimum (in x) and the dual
optimum (in λ ⪰ 0 and free ν); linear programming duality is the special case in which f and the
g_i are linear and L is bilinear.

## Worked example: equality-constrained convex QP

For min ½xᵀPx + qᵀx s.t. Ax = b with P ⪰ 0, the Lagrangian is L = ½xᵀPx + qᵀx + νᵀ(Ax − b) and the
KKT conditions are stationarity Px⋆ + q + Aᵀν⋆ = 0 together with Ax⋆ = b, i.e. the linear system

  [ P  Aᵀ ] [ x⋆ ]   [ −q ]
  [ A  0  ] [ ν⋆ ] = [  b ],

whose solution gives the optimal primal and dual variables directly (there are no inequality
constraints, so no complementary-slackness branching).

## Worked example: water-filling

For min −Σ_i log(α_i + x_i) s.t. x ⪰ 0, 1ᵀx = 1 (α_i > 0), introduce λ ⪰ 0 for −x_i≤0 and ν for the
equality. KKT gives x⋆ ⪰ 0, 1ᵀx⋆ = 1, λ⋆ ⪰ 0, λ⋆_i x⋆_i = 0, and −1/(α_i + x⋆_i) − λ⋆_i + ν⋆ = 0.
Eliminating the slack variable λ⋆ yields x⋆_i = max{0, 1/ν⋆ − α_i}, with ν⋆ fixed by
Σ_i max{0, 1/ν⋆ − α_i} = 1 — "flood to water level 1/ν⋆ over ground heights α_i until unit volume
is used." Complementary slackness is exactly what turns off (x⋆_i = 0) the channels whose floor
α_i sits above the water level.
