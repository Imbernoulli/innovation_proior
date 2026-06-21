OK, let me think this through from scratch. I have n points to place in the unit square, and for any three of them they span a triangle; the worst three span the smallest triangle, and I want to make even *that* smallest triangle as large as I can. And separately — and this is the part I keep wanting to skip but can't — I want to know whether there's a *ceiling*: a proof that no matter how I place the points, some triangle is forced to be tiny. Two different jobs. One is "build a good set," the other is "prove every set is bad." Let me not conflate them.

What's the trivial range. If I triangulate my n points, I get a planar subdivision; the number of triangles in a triangulation of n points is on the order of n, more precisely at least n−2 if I just triangulate the convex hull and interior. Those triangles have disjoint interiors and all sit inside a region of area 1. So the smallest of them has area at most 1/(n−2). That means: in *any* configuration, some triple spans a triangle of area ≤ ~1/n. So Δ(n) = O(1/n) for free, no cleverness. That's my starting ceiling.

And the floor? Let me try the dumb constructions to feel the problem. Points evenly on a circle: take three consecutive ones. Two adjacent points are ~1/n apart along the circle, the angle subtended is tiny, the triangle they make with the next point is a sliver — its area scales like 1/n³. Awful. Grid points are worse: three in a row are collinear, area exactly zero. So "spread them out evenly" is exactly the wrong instinct; regularity creates collinearity, and collinearity is death.

So I'm sitting between Ω(1/n³)-ish junk constructions and an O(1/n) ceiling, and I have no idea where the truth is. Let me get a real construction first.

I keep coming back to one question: what does it actually *take* for three points to make a small triangle? Fix two of them, x and y, distance d apart. The third point z — the area of the triangle is ½ · base · height = ½ · d · (distance from z to the line through x and y). So area(x,y,z) is small *exactly when z is close to the line xy*. Precisely, area < ε ⟺ dist(z, line xy) < 2ε/d. If I call T_xy(w) the strip of total width w, then the bad locus is T_xy(4ε/d). So "no small triangle" means: for every pair of my points, the strip of the appropriate width around their line is empty of all other points.

That reframing feels important. A small triangle isn't really about three points being close — it's about one point lying in a thin strip determined by the other two. Let me hold onto the strip. It's going to be the only object I really use.

Now, construction. I don't trust myself to hand-place points cleverly — every regular pattern I try creates collinearities. So let me be lazy and random: throw points down uniformly and see what the strips do on average. For that I need one number: if p, q, r are three independent uniform points, what's the probability they make a triangle of area below ε?

Condition on the base. The distance between two uniform points — what's its distribution near a value x? The second point lies in an annulus of radii x and x+dx around the first, area π((x+dx)²−x²) ≈ 2πx dx. So the pair-distance density is ≲ 2πx. Now given the base length |pq| = x, the third point makes area ≤ ε iff it lands within distance 2ε/x of the line, a strip of total width 4ε/x; its intersection with the square has area ≲ ε/x. So

  P(area ≤ ε) ≲ ∫₀^{√2} (ε/x) · 2πx dx ≍ ε.

Wait — look at that. The ε/x from the strip (short bases give wide strips, dangerous) and the 2πx from the annulus (short bases are rare) *cancel*. The x drops out. The upper bound is proportional to ε. And if I restrict to base lengths and positions bounded away from the boundary, I also get a matching lower bound, so in the Θ-sense P(area ≤ ε) is linear in ε. For the construction I only need the O(ε) side, and it is exactly what makes the expected number of bad triples behave cleanly.

So now the alteration method writes itself. Throw down some number N of uniform points. The expected number of "bad" triples — triples spanning area ≤ ε — is, by linearity, C(N,3) · P(area ≤ ε) ≍ ε N³. I want to end with n good points. If I throw down N = 2n points, expected bad triples is ≍ ε n³. Choose ε = c/n² with c small: then E[#bad] ≍ c n, and with c small enough that's less than n. Now destroy every bad triple by deleting one of its three vertices. I delete at most (number of bad triples) points, so with positive probability I delete fewer than n points; starting from 2n, at least n survive with *no* triangle of area below c/n².

So Δ(n) = Ω(1/n²). Good — that already smashes the circle and beats the trivial floor, and it lands exactly on what Erdős's parabola construction gives too (the points (x, x² mod p), no three collinear because a parabola meets a line ≤ twice, scaled down so Pick's theorem floors every area at ~1/n²). Both say 1/n². And Heilbronn conjectured 1/n² is the truth. So maybe I'm done and the answer is Θ(1/n²)?

Let me not believe that yet. Let me see whether the construction can be pushed. The natural knob is ε. What if I'm greedier and set ε bigger — say ε = K/n² with K large, or even ε = (log n)/n²? Then E[#bad] ≍ ε n³ ≍ n log n. Now I have ~ n log n bad triples and only 2n points. Delete one vertex per bad triple and I annihilate everything — the deletion removes far more points than I have. So one-vertex-per-triple deletion hits a wall at ε ≍ 1/n². I can't raise the threshold this way.

So I'm stuck — unless the *wastefulness* of "one vertex per triple" is the real culprit, not the threshold itself. Let me look hard at what deletion is actually doing. I have N points and a collection of bad triples. I want a large subset of points containing *no* complete bad triple. That's not "hit every triple with a deleted vertex" — that's exactly the statement that I want a large **independent set in a 3-uniform hypergraph**, where the vertices are my points and the hyperedges are the bad triples. Deletion-one-per-edge is just a crude way to find an independent set, and crude is the operative word: it can throw away three good points to kill one edge.

So the question becomes: how large an independent set does a 3-uniform hypergraph with N vertices and m edges have? If m ≍ εN³ and I want the independent set to have ~ n vertices, what's the real tradeoff? For a generic dense hypergraph the answer is roughly N/(average degree)^{1/2} and that just reproduces the 1/n² ceiling. But hypergraphs coming from *random geometry* are not generic — they're sparse in a specific way. Let me think about what structure my bad-triple hypergraph has.

Average degree first. With N points and ε N³ edges, the average number of edges through a vertex is ~ εN³/N = εN². Write the average degree as t² (it's a 3-uniform graph, so the natural parameter is the "codegree scale" t with t² = degree). So t² ≍ εN², t ≍ N√ε.

Now — is there a theorem that a *sparse* 3-graph has an independent set bigger than the generic N/t? There is, and the structural hypothesis it needs is exactly the kind of thing random points should satisfy: **uncrowdedness** — no two edges sharing two vertices (no "2-cycle"), and no short cycles of edges (no 3- or 4-cycles). The Ajtai–Komlós–Pintz–Spencer–Szemerédi bound says: a k-uniform uncrowded hypergraph on N vertices with average degree t^{k−1} has an independent set of size

  α(H) ≥ C_k · (N/t) · (ln t)^{1/(k−1)}.

For triangles, k = 3, so α ≥ C₃ · (N/t) · (ln t)^{1/2}. There's the logarithm. The elementary random-selection-and-delete estimate naturally lives at scale N/t; uncrowdedness buys the extra (ln t)^{1/2}. The mechanism, intuitively: you don't kill an edge by deleting a vertex outright; you build the independent set by a slow semi-random nibble, admitting vertices while keeping the local conflicts controlled, and because the graph is uncrowded — locally tree-like, no short cycles to create cascading conflicts — the conflicts stay rare enough to keep the square-root logarithm.

Let me check it gives what I want. Set the independent set size equal to n and solve for the largest ε I can afford. With α ≍ (N/t)(ln t)^{1/2} and t ≍ N√ε:

  α ≍ (N / (N√ε)) (ln t)^{1/2} = (1/√ε)(ln t)^{1/2}.

I want α = n. If I keep N comparable to n and try ε ≍ (log n)/n², then t is only about √log n, so ln t is only log log n; the theorem does not pay enough. The honest way to make t a genuine power of n is to over-sample polynomially: take N = n^{1+δ}. Then

  t² ≍ εN².

Now put ε = c(log n)/n². This gives

  t² ≍ c n^{2δ} log n,
  t ≍ √c n^δ (log n)^{1/2},
  ln t = δ log n + ½ log log n + O(1).

Therefore

  α ≳ C₃ · n/(√c (log n)^{1/2}) · (δ log n + o(log n))^{1/2}
      ≍ C₃ √(δ/c) · n.

Good — not n√log n; the √log from AKPSS is spent exactly cancelling the √log in 1/√ε. That is the whole point. By choosing the constant c small enough, the independent set is larger than n by a fixed factor. Equivalently: I can push the area threshold all the way up to c(log n)/n² and still find n points with no triangle below it. That's the gain. **Δ(n) = Ω(log n / n²).** Heilbronn's conjecture that 1/n² is the truth is *false* — there's an extra logarithm.

But I cheated: I assumed the bad-triple hypergraph is uncrowded, and a random one isn't, not automatically. I need to *arrange* uncrowdedness, and I need the arranging to cost me only o(N) points or it ruins the count. So what are the obstructions?

A 2-cycle is two bad triples sharing two vertices — i.e. two points x, y that are the base of *two* different small triangles, with two different apexes both landing in the strip around xy. When does that happen a lot? When x and y are unusually close, because then the distance-to-line threshold 2ε/d, equivalently total strip width 4ε/d, blows up as d → 0 and catches several points. That's the "unexpectedly close pair" problem. So step one: remove all pairs that are unexpectedly close — points so near each other that their strip is fat. By the annulus density, very close pairs are rare, so deleting a point from each such pair removes few points. Killing the close pairs kills the dominant source of 2-cycles.

After the close pairs are gone, I still have to kill any residual 2-cycle, and then the 3- and 4-cycles: two, three, or four bad triples whose shared-vertex pattern forms a short hypergraph cycle. These are higher-order coincidences among the random points; a second-moment estimate — compute the expected number of such configurations after the close-pair cutoff and show it is o(N) — says they are rare, so deleting a point from each removes only a sublinear number of points. After removing the close pairs and the points sitting in these residual short cycles, what remains is an uncrowded 3-graph on (1−o(1))N points, and AKPSS applies to it. The constant slack in the independent-set bound is chosen large enough that these o(N) losses do not matter.

Let me re-examine the parameter choice once more, because the whole thing hinges on it. I place N = n^{1+δ} points, set ε = c(log n)/n². E[#bad] ≍ ε N³ = c n^{1+3δ} log n — many bad triples, but that's fine now because I'm not deleting one per edge; the average degree is t² ≍ εN² ≍ c n^{2δ} log n, so t ≍ √c n^δ(log n)^{1/2}. Remove the very close pairs and then the o(N) vertices needed to destroy residual 2-, 3-, and 4-cycles. Apply AKPSS: the independent set has size ≳ C₃√(δ/c)n. Choose c small, take any n of the independent vertices, and every kept triple has area > c(log n)/n². This is the Komlós–Pintz–Szemerédi lower construction: Ω(log n/n²). The algorithmic version is the same structural pipeline with a certified independent-set routine for uncrowded hypergraphs.

So that's the floor: Ω(log n/n²), beating the conjecture by a log. Now I owe myself the ceiling, and this is where I have to prove something about *every* configuration, not build one.

Back to the strips. If Δ is the smallest triangle area in my set, then for every pair τ = {x,y}, the strip T_τ of width 4Δ/d(τ) about the line ℓ_τ contains no point of P except the two base endpoints. (Width 4Δ/d because area < Δ ⟺ within 2Δ/d of the line on either side, total width 4Δ/d.) So: a min-area of Δ forces a whole family of third-point-free strips, one per pair, and if Δ is large these strips are wide. Wide strips with only their endpoints inside are a lot of forbidden real estate. The contradiction I'm looking for is "you can't have that many wide endpoint-only strips around n points." Let me find the cleanest way to cash that in.

The first cash-out is to weight and integrate. For each pair τ, the endpoint-only strip T_τ has total width w_τ = 4Δ/d(τ). Assign the strip the weight w_τ itself. Now look at a single point x of the square and ask how much *weighted* strip passes through it:

  Σ_τ w_τ · 1[x ∈ T_τ].

Schmidt's geometric estimate is exactly that this is O(1), uniformly in x. The point is not that there are few strips through x unweighted; there can be many. The width weight is doing the work, and the endpoint-only strip condition prevents wide contributions from piling up at one point. Integrate over the square:

  ∫_{[0,1]²} Σ_τ w(T_τ)1_{T_τ}(x) dx ≲ ∫_{[0,1]²} 1 dx = 1.

Now expand the same integral the other way. For τ,

  w_τ ∫_{[0,1]²} 1_{T_τ}(x) dx
  = (4Δ/d(τ)) · area(T_τ ∩ [0,1]²).

The strip-area estimate used in Schmidt's argument gives area(T_τ ∩ [0,1]²) ≳ Δ/d(τ) for these endpoint-only strips. So each weighted strip contributes at least a constant multiple of

  (Δ/d(τ)) · (Δ/d(τ)) = Δ²/d(τ)².

Summing,

  Δ² · Σ_τ 1/d(τ)² ≲ 1.

And Σ_τ 1/d(τ)² — the Riesz 2-energy of the point set — is bounded below for *any* n points. The clean way to see the logarithm is dyadic. At a grid scale ρ, with ρ^{-2} not much larger than n, Cauchy's inequality forces many pairs to lie in the same or neighboring ρ-squares; each such pair contributes ≳ ρ^{-2}. The count of close pairs at that scale times ρ^{-2} is ≳ n². There are Θ(log n) such scales before the grid reaches the average spacing, so

  Σ_τ 1/d(τ)² ≳ n² log n.

Therefore

  Δ² · n² log n ≲ 1  ⟹  Δ ≲ n^{−1}(log n)^{−1/2}.

That's the ceiling from energy alone — clean, and it already beats the trivial 1/n by √(log n). But notice it's still essentially 1/n; the energy argument is "soft" and can't produce a *polynomial* saving. It treats all the endpoint-only strips democratically and only sees their total measure. To go polynomially below 1/n I need to exploit that the strips aren't just collectively large — they have only endpoint incidences in a way that forces the points into pseudo-random behavior at one scale and impossible scarcity at another, and those two can't coexist.

Let me draw the actual picture and stare at it. The unit square, and through it a base line ℓ_τ with a strip of width 4Δ/d(τ) around it — a thick band with a thin line down its middle — and the condition is that no third point lands in the band. Lay down many such bands, one per short base, and I'm looking at a square crisscrossed by strips, each strip a thin line fattened to a width, and I want to know whether any strip swallows a point beyond the pair that generated it. Strip down the middle, extra point or no extra point inside it — that is not an area question anymore. That is *incidence* geometry: I have a set of points P and a set of strips (fattened lines) L, and the whole content of "minimum triangle area ≥ Δ" is a statement about how few nontrivial incidences occur at the strip-width scale. I have been computing measures of unions of strips; the right object is the *incidence count* between points and lines as I vary the fattening scale. And the moment I phrase it that way I recognize the picture — it is exactly the geometry that the high-low machinery is built to handle: a collection of points and a collection of thin tubes, asking how the overlap behaves across scales. Schmidt's endpoint-only strips and that incidence machinery are the same picture; the energy bound was just the crudest possible incidence estimate, counting total mass and nothing else.

So let me run it as an incidence problem honestly. Pick a length scale u and look only at *short* pairs, d(τ) ≤ u; let L be the lines supporting them. The strip condition says: around each such line, a strip of width 4Δ/u contains only the two generating endpoints (since width 4Δ/d(τ) ≥ 4Δ/u for short bases, the larger strip already has no third point). Count *incidences* between the n points and these lines L at a varying scale w: I(w) = Σ_{(p,ℓ)∈P×L} η(w⁻¹ dist(p,ℓ)), with η a fixed bump adapted to a strip of width w. At scale comparable to Δ/u, each line contributes only its endpoint incidences, so I(Δ/u) is O(|L|), and the normalized count B(w)=I(w)/(w|P||L|) is at most a constant times u/(Δ|P|).

Now play two scales against each other. If the n points were spread uniformly at random, the expected incidence count at scale w would be ≈ w·|P|·|L| — incidence density ≈ |P||L|, scale-independent. So define the renormalized count I(w)/(w|P||L|); for genuinely random points it's ≈ 1 at every scale. I need a coarse initial scale w_i where the set is forced to look pseudo-random, I(w_i) ≳ w_i|P||L|, so the renormalized count is ≳ 1. Then I need the renormalized count to barely change as I shrink the scale from w_i down to a fine scale w_f:

  |I(w_i)/(w_i|P||L|) − I(w_f)/(w_f|P||L|)| ≪ 1.

If both hold, the renormalized count is still ≳ 1 at the fine scale w_f — meaning there are still *many* incidences at scale w_f. The endpoint-only condition then gives the contradiction. In the clean mental version, enough incidences at width w_f force a third point into a strip generated by a base of length at most u, and that gives a triangle of area ≲ u w_f. When I keep the smoothing, the size of L, and the endpoint incidences honest, the conversion has the more precise shape

  Δ/|log Δ| ≲ max{u, u^{2/3}w^{1/3}} |P|^{-1/3}|L|^{-1/3}κ^{-2/3} + u|P|^{-1}κ^{-1},

whenever B(w;P,L) ≥ κ and the incidence count is large enough that endpoint incidences alone cannot explain it. That is the rigorous bridge: a coarse lower bound for B, plus stability down to a fine scale, becomes a numerical upper bound for Δ. Choosing the early Roth scales well — for instance u ≍ n^{−1/3}, w_f ≍ n^{−3/4} in the toy calculation — already gives Δ ≲ n^{−13/12}, a *polynomial* saving below 1/n.

The scale stability is the hard part. Take the difference of strip indicators at two scales,

  ϕ_ℓ(x) = (1/w_i)·1_{T_ℓ(w_i)}(x) − (1/w_f)·1_{T_ℓ(w_f)}(x),

a function that integrates to zero across the strip. For two different lines ℓ₁, ℓ₂, the functions ϕ_{ℓ₁} and ϕ_{ℓ₂} are nearly *orthogonal* — ∫ ϕ_{ℓ₁}ϕ_{ℓ₂} ≈ 0 — because two strips at different scales intersect in a parallelogram whose area is proportional to both widths, and the zero-mean cancellation kills the overlap. With a quasi-orthogonal system in hand, a Selberg/Bessel-type inequality bounds Σ_ℓ |⟨f, ϕ_ℓ⟩|² by ‖f‖², and unpacking that with f the point measure bounds exactly the change in renormalized incidence count between the two scales. Optimizing the scales through this machinery gives the polynomial exponent; pushing the optimization to its limit lands on Δ ≲ exp(c√log n)/n^{8/7} — exponent 8/7, i.e. the saving µ = 1/7 over 1/n.

Why 8/7 and not further? It's the optimum of the scale-parameter system in this two-scale incidence method: balancing the initial-estimate scale against the inductive shrink, the exponent that survives is 1 + 1/7. Roth's own analytic version got smaller savings (µ = 1−√(4/5) ≈ 0.106, then (9−√65)/8 ≈ 0.117); optimizing the same incidence framework pins it at µ = 1/7 ≈ 0.143. And 8/7 is where this stalled for forty years, so before I try to break it I should find out *what specific thing* makes 8/7 the wall — not a number that falls out of optimizing, but a configuration the method genuinely cannot beat.

Let me hunt for that worst case. The Selberg/Bessel comparison treats every line democratically, so it is blind to *where* the obstruction lives; I want a comparison that exposes it. The high-low method gives one: split the incidence count into a low-frequency part (slowly varying, ≈ the random prediction) and a high-frequency part (the fluctuation), and bound the fluctuation directly. At one scale it gives

  |B(w;P,L) − B(w/10;P,L)| ≲ ((M_P(w×w)·M_L(w×1))/(|P||L|) · w^{−3})^{1/2},

where B(w)=I(w)/(w|P||L|), M_P(w×w) is the most points in any w×w square, and M_L(w×1) is the most lines whose intersections with [0,1]² sit inside any w×1 tube. To compare w_i to w_f I sum this over the intermediate dyadic-like scales, so what matters is the largest of these errors between w_f and w_i. The renormalized count is stable — and the argument goes through — *unless* M_P or M_L is large, unless there is a small square crammed with points or a thin tube crammed with lines. A globally well-spaced set keeps M_P under control; a single narrow cluster either spans too few useful directions or is better treated as a rectangle. The configuration that pins the method at 8/7 is the mixed one: points that look spread out at the global scale, but when I zoom into sub-squares I find orderly local clusters. That is exactly the shape the optimization keeps bumping into, and it is what I have to forbid. It can't be too random and it can't be too structured; the worst case is the one in between.

So the question is no longer "how many incidences" but "how concentrated can the points be at small scales while still being spread out globally" — and concentration-while-spread-out is precisely a statement about the *dimension* of the local clusters. A cluster essentially trapped in one thin tube spans too few useful directions, so it has to be handled as a narrow rectangle. A cluster with dimension above 1 is better: Marstrand-type direction estimates force its spanned directions to fill a one-dimensional set, which gives the coarse incidence lower bound I need. The stubborn local clusters are below dimension 1 but not trapped in a single thin tube. There the classical >1-dimensional theorem says nothing, and I need a direction-set lower bound that still works for 0<s<1: if the cluster is s-regular and no thin tube captures a fixed positive fraction of it, a discretized radial-projection estimate forces many separated directions. If the tube nonconcentration fails, I have learned something too: I am in the rectangle case, where the cluster is thin enough to handle separately. The two ways of being a worst case — clumped points, many lines — start cancelling. The Orponen–Shmerkin–Wang below-dimension-1 radial-projection estimate is exactly the missing input. Feeding it into the high-low comparison, with the narrow-tube case split off as rectangles, lets the induction reach finer scales than the Selberg machinery could, and the net effect is to push the exponent just past the KPS barrier: Δ ≲ n^{−8/7−1/2000}. For homogeneous point sets — at most O(1) points per n^{−1/2}×n^{−1/2} cell, so there is no hidden small-scale clustering at all — the same high-low limit gives Δ ≲ n^{−7/6+ε} for every fixed ε>0.

Let me step back and recap the causal chain, because the two halves rhyme. The whole subject runs on one object: small triangle ⟺ a point in a thin strip about a base line. On the *construction* side, that turned P(small triangle) into a clean linear-in-ε law, which fed the alteration method to Ω(1/n²); then the realization that deletion-one-per-triple is just a wasteful independent-set heuristic — and that the bad-triple hypergraph is *uncrowded* once I remove the close pairs and the sublinear set of residual 2-, 3-, and 4-cycles — let the AKPSS independent-set theorem spend its (log t)^{1/2} gain to lift the threshold to Ω(log n/n²) and refute Heilbronn's conjecture. On the *impossibility* side, the same strips, now forced to contain only their endpoints by a large Δ, are too much forbidden incidence: weighting by width and integrating gives the energy bound Δ ≲ n^{−1}(log n)^{−1/2}; then seeing the crisscrossed-strip picture for what it is — an incidence problem between points and fattened lines — replaces the soft energy count by a two-scale incidence comparison (pseudo-random at a coarse scale, only endpoint incidences at the fine scale, stable in between by quasi-orthogonality of strip differences) and gives the first polynomial saving, optimized to exponent 8/7. The 8/7 wall is a concentration pattern — points spread out globally but locally clustered — and the high-low comparison names the exact quantity that goes bad, M_P·M_L. The remaining question becomes the dimension of those clusters: above dimension 1, Marstrand-type direction spread is enough; below dimension 1, the Orponen–Shmerkin–Wang radial-projection bound supplies the missing direction spread unless the set is trapped in a narrow rectangle, which is handled separately. That pushes just past the wall. The gap is still enormous: up to the logarithm, the construction allows Δ as large as n^{-2}, while this incidence/projection machinery only forces Δ below n^{-8/7-1/2000}.

So the construction is: drop a random over-sample, read off the thin-strip bad triples, strip the configuration down to an uncrowded hypergraph by removing the close pairs and residual short cycles, and use a certified independent-set routine — n points whose every triangle has area Ω(log n/n²). The upper-bound proof runs the identical strips backwards: a large minimum area leaves each generated strip with only its base endpoints, and the incidence/projection machinery turns that scarcity into a triangle of area at most n^{−8/7−1/2000}.
