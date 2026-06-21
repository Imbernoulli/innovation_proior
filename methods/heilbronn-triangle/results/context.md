# Context: maximizing the smallest triangle among n points in the square

## Research question

Place n points in the unit square [0,1]² and look at all C(n,3) triangles they span. Some triangle is the smallest; call its area the *gap* of that configuration. By moving the points we can make that smallest triangle larger. The question is how large it can be made, as a function of n:

    Δ(n) = sup over n-point sets P ⊂ [0,1]²  of  min over distinct p,q,r ∈ P  of  area(p,q,r).

We want the asymptotics of Δ(n) — matching lower and upper bounds on the exponent (and, ideally, on the logarithmic factor). A *lower* bound is an explicit family of configurations whose smallest triangle is provably not too small; an *upper* bound is a proof that *every* configuration, however cleverly arranged, must contain some triangle below a given area. The two sides are answered by completely different machinery, and a solution worth the name has to supply both: a principled construction, and a principled impossibility proof.

Two crude bounds frame the problem. Triangulate any n points inside the square: you get at least n−2 triangles with disjoint interiors, all inside a region of area 1, so by pigeonhole one has area ≤ 1/(n−2). Hence Δ(n) = O(1/n) for free. On the other side, naive constructions are terrible: n points on a circle give a triangle of area Θ(1/n³) (three consecutive points span a sliver), and points on a grid give *zero*-area (collinear) triples. So the trivial window is Ω(1/n³)-ish constructions up against an O(1/n) impossibility, and the whole game is to close it.

## Background

**The geometry of one triangle.** Fix two points x, y at distance d = |xy|. A third point z makes a triangle of area A = ½·d·h, where h is the distance from z to the line ℓ through x and y. So area(x,y,z) < ε is *exactly* the statement that z lies within distance 2ε/d of the line. With the standard convention that T_xy(w) denotes the strip of total width w, the locus is T_xy(4ε/d). This single observation — small triangle ⟺ third point in a thin strip about the base line — is the pivot of the entire subject. It converts a statement about triangle areas into a statement about incidences between points and strips, and that is the form in which both the constructions and the impossibility proofs are naturally argued.

**The probability that three random points span a small triangle.** If p, q, r are independent uniform points in the square, how likely is area ≤ ε? Condition on the pair distance. The pair distance has density bounded by that of an annulus: P(|pq| ∈ [x, x+dx]) ≤ π((x+dx)²−x²) ≈ 2πx dx. Given |pq| = x, the third point makes area ≤ ε iff it lands within distance 2ε/x of the base line, a strip of total width 4ε/x and length O(1), so the event has probability ≲ ε/x. Integrating,

    P(area(p,q,r) ≤ ε) ≲ ∫₀^{√2} (ε/x)·2πx dx ≍ ε.

The matching lower bound comes by restricting to base lengths and positions bounded away from the boundary, so the probability is Θ(ε). For the alteration count only the upper bound is needed: the 1/x blowup for short bases is exactly cancelled by the 2πx area element. This clean linear law is what makes a probabilistic construction tractable.

**The probabilistic / alteration method.** The general principle: to build a structure avoiding many forbidden patterns, generate a large random object, bound the expected number of forbidden patterns, then *delete* one element from each to destroy them, and argue that few enough elements were deleted that a large structure survives. Linearity of expectation gives the count; Markov's inequality turns "small expectation" into "exists a realization that is small." This is the Erdős alteration method, and it is the natural engine for a lower-bound construction here.

**Independent sets in sparse hypergraphs.** A 3-uniform hypergraph has vertices and triples-as-edges; an *independent set* is a vertex subset containing no full edge. For a sparse hypergraph one wants the largest such set. The relevant structural notion is *uncrowdedness*: a hypergraph is uncrowded if it contains no 2-, 3-, or 4-cycles (no two edges share two vertices, and no short cycles of edges). Ajtai, Komlós, Pintz, Spencer and Szemerédi (1982) proved that a k-uniform uncrowded hypergraph on N vertices with average degree t^{k−1} has an independent set of size

    α(H) ≥ C_k · (N/t) · (ln t)^{1/(k−1)}.

For k = 3 this reads α ≥ C₃ (N/t)(ln t)^{1/2}. The point is the (ln t)^{1/2} factor: an elementary random-selection/deletion argument in a 3-graph of average degree t² naturally gives scale N/t, while uncrowdedness lets the semi-random nibble keep a square-root logarithm more. The proof was later derandomized into a deterministic polynomial-time independent-set algorithm by Bertram-Kretzberg and Lefmann.

**Incidences, energy, and density increments — the impossibility side.** Translate Δ = (min triangle area) into the strip language: for every pair τ = {x,y}, the strip T_τ(4Δ/d(τ)) of width 4Δ/d(τ) about ℓ_τ contains only the two base endpoints from P. If Δ is large these third-point-free strips are wide — a strong, exploitable scarcity. Several tools convert this scarcity into an upper bound on Δ:
- The *Riesz 2-energy* Σ_τ 1/d(τ)² of a point set. A dyadic grid proof gives Σ_τ 1/d(τ)² ≳ n² log n: at a grid scale ρ with ρ^{-2} ≲ n, Cauchy's inequality over the ρ-squares forces ≳ n²ρ² close pairs at that scale, and each such pair contributes ≳ ρ^{-2}. Thus that scale contributes ≳ n²; summing over Θ(log n) dyadic scales yields the logarithm.
- A *density-increment* iteration: restrict to pairs with short, similarly-sloped bases; their forbidden strips are essentially disjoint; pass to the complement, where P is denser relative to its area; recurse.
- A *two-scale incidence comparison*: count (smoothed) incidences I(w; P, L) = Σ η(w⁻¹·dist(p,ℓ)) between the points and the lines L supporting short pairs. Compare the incidence density at a coarse scale (where a small-triangle-free set looks pseudo-random) against a fine scale (where the strips contain only endpoint incidences); a phase transition between the two pins Δ. The comparison is made rigorous by quasi-orthogonality of the strip-indicator differences (Selberg/Bessel-type inequalities), or by the high-low method of Guth–Solomon–Wang together with projection-theory bounds on direction sets: Marstrand for dimension above 1, and Orponen–Shmerkin–Wang for the below-dimension-1 regime when thin-tube concentration is controlled.

**Heilbronn's conjecture.** Hans Heilbronn, in the late 1940s, conjectured Δ(n) = O(1/n²) — i.e. that the explicit Ω(1/n²) constructions below are essentially optimal. This makes 1/n² the threshold to beat on the construction side and the benchmark to approach on the universal upper-bound side.

## Baselines

**Erdős's algebraic construction — Δ ≥ Ω(1/n²).** Take a prime p with n ≤ p ≤ 2n and the set {(x, x² mod p) : x ∈ {0,…,p−1}} ⊂ [p]². No three points are collinear because the parabola y = x² meets any line in at most two points mod p. Scaling these lattice points into the unit square: any triangle on lattice points has area ≥ ½ (Pick's theorem), and dividing coordinates by p shrinks areas by p², so every triangle has area ≥ 1/(2p²) ≥ 1/(8n²). Clean and explicit, but it stops exactly at 1/n² — it gives no route past Heilbronn's conjectured order.

**Greedy / simple probabilistic placement — Δ ≥ Ω(1/n²).** Drop 2n uniform points; with the linear law E[#bad triples] = O(ε·n³); set ε = c/n² so the expectation is less than n; delete one vertex per bad triple; ≥ n points survive with no triangle below c/n². Matches Erdős. Its ceiling is structural: deleting one vertex per bad triple "spends" up to one point per forbidden triple, so once ε is raised to the point where there are ≫ n bad triples, the deletion erases everything. This is the gap a better lower bound must beat.

**Komlós–Pintz–Szemerédi lower construction — Δ ≥ Ω(log n/n²).** Keep the same random-placement geometry but replace one-vertex-per-edge deletion by an independent-set problem in the bad-triple 3-graph. With N=n^{1+δ}, ε=c(log n)/n², and average degree t²≈c n^{2δ}log n, the AKPSS uncrowded-hypergraph bound gives α≳C√(δ/c)n after deleting very close-pair obstructions and the residual 2-, 3-, and 4-cycle obstructions. Choosing c small leaves n independent vertices, so every surviving triangle has area above c(log n)/n².

**Roth's 1951 upper bound — Δ = O(n⁻¹(log log n)^{−1/2}).** The first nontrivial impossibility proof. Using the endpoint-only strip scarcity, Roth restricts to a family of short, similarly-sloped pairs whose forbidden strips are nearly disjoint, finds in the complement a sub-square where the points are denser relative to area, and iterates the density increment. It barely beats the trivial 1/n but it introduces the strip method that everything later refines. (It is recognised as a precursor of Roth's theorem on three-term arithmetic progressions.)

**Schmidt's 1972 upper bound — Δ = O(n⁻¹(log n)^{−1/2}).** A much simpler argument that first makes the *energy* connection explicit. For every pair τ, take the endpoint-only strip T_τ(4Δ/d(τ)) and weight it by its width w_τ = 4Δ/d(τ). Schmidt's geometric estimate says that every point x in the square sees bounded total weighted strip mass,

    Σ_τ w_τ 1_{T_τ}(x) ≲ 1.

Integrating over the square gives O(1). Expanding the same integral by strips gives

    Σ_τ w_τ area(T_τ ∩ [0,1]²) ≳ Σ_τ (Δ/d(τ))² = Δ² Σ_τ 1/d(τ)².

With the universal Riesz-energy lower bound Σ 1/d(τ)² ≳ n² log n this gives Δ ≲ n⁻¹(log n)^{−1/2}. Simple and sharp of its kind, but energy alone cannot break the 1/n barrier polynomially.

**Roth's 1972 analytic method — first polynomial gain, Δ ≲ n^{−1−µ}.** Reintroduces a quantitative two-scale incidence comparison: a coarse scale where small-triangle-free sets behave pseudo-randomly, a fine scale where the forbidden strips contain only endpoint incidences, and a quasi-orthogonality inequality (Selberg's generalisation of Bessel) bounding how much the renormalised incidence count can change between the two. Optimising the scale parameters gives µ = 1 − √(4/5) ≈ 0.1056, later improved by the same method to µ = (9 − √65)/8 ≈ 0.1172. The first proof that Δ is polynomially below 1/n, leaving the *value* of the optimal exponent open.

**Komlós–Pintz–Szemerédi's optimized upper bound — Δ ≲ exp(c√log n)/n^{8/7}.** Optimising Roth's two-scale incidence framework pushes the polynomial saving to µ = 1/7. This is the natural barrier of that incidence setup: the initial coarse-scale estimate, the final fine scale, and the quasi-orthogonal scale comparison all balance at exponent 8/7.

**High-low and radial-projection tools.** The high-low method replaces a single Selberg comparison by a frequency split. For the normalized smoothed incidence count B(w;P,L)=I(w;P,L)/(w|P||L|), the single-scale comparison is

    |B(w;P,L) − B(w/10;P,L)| ≲ ((M_P(w×w) M_L(w×1))/(|P||L|) · w^{-3})^{1/2},

where M_P(w×w) is the maximum number of points in a w-square, and M_L(w×1) is the maximum number of lines whose intersections with [0,1]² lie inside a w-tube. Thus normalized incidences are stable across scales unless points concentrate in small squares or lines concentrate in thin tubes. Projection theory supplies direction-set spread for the line families generated inside clusters: above dimension 1 this comes from Marstrand-type direction estimates, while below dimension 1 one needs the Orponen–Shmerkin–Wang radial-projection estimate together with a check that the cluster is not itself trapped in a thin tube.

## Evaluation settings

Bounds are stated asymptotically in n, on [0,1]² (equivalently any fixed convex body, since affine maps change areas by a constant factor, so constants are dropped and only the exponent and log power are tracked). A lower-bound construction is judged by the largest provable min-area it guarantees; an upper-bound proof by the smallest exponent/log-power of the universal area it forces. For the constructive version, the yardstick is whether the configuration achieving a given guarantee can be produced by a certified selection rule, not merely shown to exist.


