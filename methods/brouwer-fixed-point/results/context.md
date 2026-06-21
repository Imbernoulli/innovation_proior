# Context: continuous self-maps and the search for an unavoidable fixed point

## Research question

The question is deceptively concrete. Take a region of space with no holes — a closed disk, a solid ball, a filled triangle, a cube, any nonempty compact convex body — and stir it: apply *any* continuous map of the body into itself. Must some point come back to exactly where it started? In symbols: is it true that for every continuous `f : X → X`, with `X` a closed ball (or nonempty compact convex set) in ℝⁿ, there exists `x*` with `f(x*) = x*`?

In dimension one the answer is yes and the reason is elementary (below). What is wanted is the *n*-dimensional statement, for arbitrary continuous `f` — not merely smooth `f`, not merely affine `f`. A fixed point is an *existence* claim about an unknown point of an infinite set, with no formula to solve: continuity is the structure available, and the point must be produced out of compactness, convexity (no holes), and continuity.

Why it matters: such a theorem would be a universal certificate of equilibrium. Wherever a continuous process maps a compact convex state-space into itself — an iterated transformation, a system of simultaneous relations, a self-referential assignment — it would guarantee a self-consistent state exists, with no need to construct it.

## Background

**The one-dimensional case is free, and it already exposes what matters.** Let `f : [a,b] → [a,b]` be continuous and set `g(x) = f(x) − x`. Because `f(a) ≥ a` and `f(b) ≤ b`, we have `g(a) ≥ 0 ≥ g(b)`. The Intermediate Value Theorem (Bolzano, Cauchy) gives a zero `x*` of `g`, i.e. `f(x*) = x*`. Two features of this proof are load-bearing and survive to all dimensions: (i) it is the boundary behavior — `f` cannot push the endpoints outward — that forces a sign change; (ii) it uses closedness of the domain. On the open interval `(0,1)` the map `f(x) = x²` has `f(x) < x` everywhere and no fixed point: the would-be fixed point `0` lies outside the domain. Compactness/closedness is the hypothesis that keeps the fixed point from escaping to a missing boundary.

**The shape of the domain matters too.** Rotating an annulus by a fixed angle is a continuous self-map with no fixed point. The annulus has a hole; the disk does not. A proof must use a property the disk has and the annulus lacks — informally, that the disk is "filled in," that its boundary sphere does not bound a hole inside. This is a topological invariant of the domain, not an analytic estimate.

**Degree and the index of a map (the tool of the period).** By 1910 the way to attach an integer to a continuous map was the **Kronecker index / mapping degree**, generalizing the winding number. For a map of a sphere to itself, the degree counts (with orientation sign) how many times the image sweeps over a generic point; it is invariant under continuous deformation (homotopy). Maps of different degree are not homotopic. The identity map of a sphere has degree 1; a constant map has degree 0; the antipodal map of `Sⁿ⁻¹` has degree `(−1)ⁿ`. **Hadamard (1910)**, in an appendix to Tannery's *Introduction à la théorie des fonctions d'une variable*, used the Kronecker index to prove the fixed-point theorem for *differentiable* self-maps of the ball: the argument uses the Jacobian to count preimages with sign.

**The homology of the sphere (the obstruction, in its clean modern form).** The same invariant has a homological face: for `n≥2`, `H_{n−1}(Sⁿ⁻¹) ≅ ℤ` is infinite cyclic, while the solid ball is contractible, `H_{n−1}(Dⁿ) = 0` (in dimension `1`, the same statement is the reduced `H_0` obstruction, or just connectedness of the interval versus its two-point boundary). The sphere "detects an (n−1)-dimensional hole"; the ball, being filled, detects none. This inequality is the precise sense in which the disk differs from the annulus.

**A purely combinatorial parity fact (Sperner 1928).** Independently of the analytic question, a finite parity statement about colorings of a subdivided simplex was established in 1928, in the course of new proofs of invariance of dimension/domain: it needs no analysis, only counting on a triangulation, and the count it produces is an *odd number* that survives every refinement of the mesh. It is an elementary fact about colorings.

**Triangulation as a finite probe.** The standing technique for turning a continuous statement into something countable is to triangulate the domain and read finitely many vertex data at each mesh. A simplex can be triangulated arbitrarily finely, and a continuous map can be sampled at the resulting vertices.

## Baselines

The prior approaches a new proof would be measured against:

- **IVT / one-dimensional fixed point (Bolzano–Cauchy).** Core idea: a continuous real function changing sign has a root; apply to `g(x)=f(x)−x`. Math: `g(a)≥0≥g(b) ⇒ ∃ x*, g(x*)=0`. It is a one-dimensional argument, using the order on ℝ and a single function whose sign change localizes a point.

- **Kronecker-index / degree, differentiable case (Hadamard 1910).** Core idea: attach a deformation-invariant integer (the degree) to the boundary map; a fixed-point-free map would force the same boundary map to have degree both `1` and `0`. Math: under the no-fixed-point assumption, `g(x)=(x−f(x))/|x−f(x)|` is defined on all of `Dⁿ`, while on `Sⁿ⁻¹` the maps `g_t(x)=(x−t f(x))/|x−t f(x)|`, `0≤t≤1`, homotope the identity (`t=0`) to `g|_{Sⁿ⁻¹}` (`t=1`), so the boundary degree is `1`; but any sphere map that extends over `Dⁿ` has degree `0`. The classical index realizes degree by signed preimage counts such as `Σ_{x∈F⁻¹(p)} sign det(dF_x)`, using the Jacobian.

- **Homological non-retraction.** Core idea: there is no continuous retraction of the ball onto its boundary sphere, because applying `H_{n−1}` to `r∘i = id_{Sⁿ⁻¹}` factors the identity of `ℤ` through `H_{n−1}(Dⁿ)=0` (or, when `n=1`, applying reduced `H_0` factors through `\tilde H_0(D¹)=0`), which is impossible. This statement is *equivalent* to the fixed-point theorem (each gives the other), and rests on the homology of spheres (singular/simplicial homology, exactness).

- **Sperner labeling / parity on a triangulation (Sperner 1928).** Core idea: color the vertices of a triangulated simplex so corners get distinct colors and a vertex on a face uses only that face's colors; then the number of fully-colored cells is odd. Math: an induction giving `R ≡ D_O (mod 2)` where `R` is the count of rainbow cells and `D_O` the count of correctly-colored boundary facets. It is a statement about colorings of an abstract triangulation.

## Evaluation settings

This is a theorem, so the yardstick is logical, not empirical. The natural test domains and the standard of success that existed at the time:

- **Domains.** The closed unit ball `Dⁿ = {x∈ℝⁿ : |x|≤1}`; the standard simplex `Δⁿ = {x∈ℝⁿ⁺¹ : x_i≥0, Σx_i=1}`; the closed square/cube; and, by homeomorphism inside the affine hull, any nonempty compact convex subset of ℝⁿ. Negative controls that any correct statement must *not* cover: the open ball (for example `x ↦ (x+e_1)/2` on the open unit ball, whose only fixed point would be the missing boundary point `e_1`, or `x²` on `(0,1)`), the sphere itself (the antipodal map is fixed-point-free), and the annulus/torus (rotation) — the failures that pin the hypotheses to compact + convex/no-hole.
- **Hypotheses to stress.** Continuity only (no differentiability, no contraction/Lipschitz assumption — a contraction would give the fixed point by iteration). The proof must use compactness essentially and must localize to the shape of the domain.
- **Standard of success.** A complete proof for all `n`, reducing to a finite/elementary certificate at each scale and passing to a limit; reproducible by hand on the low-dimensional cases (`n=1` interval, `n=2` triangle/square) and inductively extendable. "Validation" means: every step checked, the parity count exact, the limit argument airtight, and the counterexamples (open/holed domains) correctly excluded.
</content>
</invoke>
