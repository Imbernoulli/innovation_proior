OK, let me think this through from scratch. I have n points to place in the unit square, and for any three of them they span a triangle; the worst three span the smallest triangle, and I want to make even *that* smallest triangle as large as I can. And separately — and this is the part I keep wanting to skip but can't — I want to know whether there's a *ceiling*: a proof that no matter how I place the points, some triangle is forced to be tiny. Two different jobs. One is "build a good set," the other is "prove every set is bad." Let me not conflate them.

First, what's the trivial range. If I triangulate my n points, I get a planar subdivision; the number of triangles in a triangulation of n points is on the order of n, more precisely at least n−2 if I just triangulate the convex hull and interior. Those triangles have disjoint interiors and all sit inside a region of area 1. So the smallest of them has area at most 1/(n−2). That means: in *any* configuration, some triple spans a triangle of area ≤ ~1/n. So Δ(n) = O(1/n) for free, no cleverness. That's my starting ceiling.

And the floor? Let me try the dumb constructions to feel the problem. Points evenly on a circle: take three consecutive ones. Two adjacent points are ~1/n apart along the circle, the angle subtended is tiny, the triangle they make with the next point is a sliver — its area scales like 1/n³. Awful. Grid points are worse: three in a row are collinear, area exactly zero. So "spread them out evenly" is exactly the wrong instinct; regularity creates collinearity, and collinearity is death.

So I'm sitting between Ω(1/n³)-ish junk constructions and an O(1/n) ceiling, and I have no idea where the truth is. Let me get a real construction first.

Here's the thing I keep coming back to: what does it actually *take* for three points to make a small triangle? Fix two of them, x and y, distance d apart. The third point z — the area of the triangle is ½ · base · height = ½ · d · (distance from z to the line through x and y). So area(x,y,z) is small *exactly when z is close to the line xy*. Precisely, area < ε ⟺ dist(z, line xy) < 2ε/d. If I call T_xy(w) the strip of total width w, then the bad locus is T_xy(4ε/d). So "no small triangle" means: for every pair of my points, the strip of the appropriate width around their line is empty of all other points.

That reframing feels important. A small triangle isn't really about three points being close — it's about one point lying in a thin strip determined by the other two. Let me hold onto the strip. It's going to be the only object I really use.

Now, construction. I don't trust myself to hand-place points cleverly — every regular pattern I try creates collinearities. So let me be lazy and random: throw points down uniformly and see what the strips do on average. For that I need one number: if p, q, r are three independent uniform points, what's the probability they make a triangle of area below ε?

Condition on the base. The distance between two uniform points — what's its distribution near a value x? The second point lies in an annulus of radii x and x+dx around the first, area π((x+dx)²−x²) ≈ 2πx dx. So the pair-distance density is ≲ 2πx. Now given the base length |pq| = x, the third point makes area ≤ ε iff it lands within distance 2ε/x of the line, a strip of total width 4ε/x; its intersection with the square has area ≲ ε/x. So

  P(area ≤ ε) ≲ ∫₀^{√2} (ε/x) · 2πx dx ≍ ε.

Wait — look at that. The ε/x from the strip (short bases give wide strips, dangerous) and the 2πx from the annulus (short bases are rare) *cancel*. The x drops out. The upper bound is proportional to ε. And if I restrict to base lengths and positions bounded away from the boundary, I also get a matching lower bound, so in the Θ-sense P(area ≤ ε) is linear in ε. For the construction I only need the O(ε) side, and it is exactly what makes the expected number of bad triples behave cleanly.

So now the alteration method writes itself. Throw down some number N of uniform points. The expected number of "bad" triples — triples spanning area ≤ ε — is, by linearity, C(N,3) · P(area ≤ ε) ≍ ε N³. I want to end with n good points. If I throw down N = 2n points, expected bad triples is ≍ ε n³. Choose ε = c/n² with c small: then E[#bad] ≍ c n, and with c small enough that's ≤ n. Now destroy every bad triple by deleting one of its three vertices. I delete at most (number of bad triples) points, so in expectation I delete ≤ n points; starting from 2n, at least n survive. With positive probability I'm left with ≥ n points and *no* triangle of area below c/n².

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

Then 3- and 4-cycles: three or four bad triples whose shared-vertex pattern forms a short cycle. These are higher-order coincidences among the random points; a second-moment estimate — compute the expected number of such configurations and show it's o(N) — says they're rare, so deleting a point from each removes only a sublinear number of points. After removing the close pairs and the points sitting in these short cycles, what remains is an uncrowded 3-graph on (1−o(1))N points, and AKPSS applies to it. The constant slack in the independent-set bound is chosen large enough that these o(N) losses do not matter.

Let me re-examine the parameter choice once more, because the whole thing hinges on it. I place N = n^{1+δ} points, set ε = c(log n)/n². E[#bad] ≍ ε N³ = c n^{1+3δ} log n — many bad triples, but that's fine now because I'm not deleting one per edge; the average degree is t² ≍ εN² ≍ c n^{2δ} log n, so t ≍ √c n^δ(log n)^{1/2}. Remove the close pairs and the o(N) cycle-points to get uncrowdedness. Apply AKPSS: the independent set has size ≳ C₃√(δ/c)n. Choose c small, take any n of the independent vertices, and every kept triple has area > c(log n)/n². This is the Komlós–Pintz–Szemerédi lower construction: Ω(log n/n²). The algorithmic version is the same structural pipeline with a certified independent-set routine for uncrowded hypergraphs.

So that's the floor: Ω(log n/n²), beating the conjecture by a log. Now I owe myself the ceiling, and this is where I have to prove something about *every* configuration, not build one.

Back to the strips. If Δ is the smallest triangle area in my set, then for every pair τ = {x,y}, the strip T_τ of width 4Δ/d(τ) about the line ℓ_τ contains *no* third point of P. (Width 4Δ/d because area < Δ ⟺ within 2Δ/d of the line on either side, total width 4Δ/d.) So: a min-area of Δ forces a whole family of *empty* strips, one per pair, and if Δ is large these strips are wide. Wide empty strips are a lot of forbidden real estate. The contradiction I'm looking for is "you can't have that many wide empty strips around n points." Let me find the cleanest way to cash that in.

Cleanest: weight and integrate. For each pair τ, the empty strip T_τ has total width w_τ = 4Δ/d(τ). Assign the strip the weight w_τ itself. Now look at a single point x of the square and ask how much *weighted* strip passes through it:

  Σ_τ w_τ · 1[x ∈ T_τ].

Schmidt's geometric estimate is exactly that this is O(1), uniformly in x. The point is not that there are few strips through x unweighted; there can be many. The width weight is doing the work, and the empty-strip condition prevents wide contributions from piling up at one point. Integrate over the square:

  ∫_{[0,1]²} Σ_τ w(T_τ)1_{T_τ}(x) dx ≲ ∫_{[0,1]²} 1 dx = 1.

Now expand the same integral the other way. For τ,

  w_τ ∫_{[0,1]²} 1_{T_τ}(x) dx
  = (4Δ/d(τ)) · area(T_τ ∩ [0,1]²).

The strip-area estimate used in Schmidt's argument gives area(T_τ ∩ [0,1]²) ≳ Δ/d(τ) for these empty strips. So each weighted strip contributes at least a constant multiple of

  (Δ/d(τ)) · (Δ/d(τ)) = Δ²/d(τ)².

Summing,

  Δ² · Σ_τ 1/d(τ)² ≲ 1.

And Σ_τ 1/d(τ)² — the Riesz 2-energy of the point set — is bounded below for *any* n points. The clean way to see the logarithm is dyadic. At a grid scale ρ, with ρ^{-2} not much larger than n, Cauchy's inequality forces many pairs to lie in the same or neighboring ρ-squares; each such pair contributes ≳ ρ^{-2}. The count of close pairs at that scale times ρ^{-2} is ≳ n². There are Θ(log n) such scales before the grid reaches the average spacing, so

  Σ_τ 1/d(τ)² ≳ n² log n.

Therefore

  Δ² · n² log n ≲ 1  ⟹  Δ ≲ n^{−1}(log n)^{−1/2}.

That's the ceiling from energy alone — clean, and it already beats the trivial 1/n by √(log n). But notice it's still essentially 1/n; the energy argument is "soft" and can't produce a *polynomial* saving. It treats all the empty strips democratically and only sees their total measure. To go polynomially below 1/n I need to exploit that the empty strips aren't just collectively large — they're empty in a way that forces the points into pseudo-random behavior at one scale and impossible scarcity at another, and those two can't coexist.

So here's the better idea. Pick a length scale u and look only at *short* pairs, d(τ) ≤ u; let L be the lines supporting them. The empty-strip condition says: around each such line, a strip of width 4Δ/u is empty of points (since width 4Δ/d(τ) ≥ 4Δ/u for short bases, the empty strip is at least this wide). Let me count *incidences* between the n points and these lines L at a varying scale w: I(w) = #{(p,ℓ) : dist(p,ℓ) < w}, smoothed by a bump function η(w⁻¹ dist(p,ℓ)) so the analysis is clean. The forbidden-strip statement is exactly that I(4Δ/u) is *tiny* — the points avoid the lines at the empty-strip scale.

Now play two scales against each other. If the n points were spread uniformly at random, the expected incidence count at scale w would be ≈ w·|P|·|L| — incidence density ≈ |P||L|, scale-independent. So define the renormalized count I(w)/(w|P||L|); for genuinely random points it's ≈ 1 at every scale. Step (A): show that at some coarse initial scale w_i, a small-triangle-free set is forced to look pseudo-random, I(w_i) ≳ w_i|P||L|, the renormalized count is ≳ 1. Step (B): show the renormalized count barely changes as I shrink the scale from w_i down to a fine scale w_f:

  |I(w_i)/(w_i|P||L|) − I(w_f)/(w_f|P||L|)| ≪ 1.

If both hold, the renormalized count is still ≳ 1 at the fine scale w_f — meaning there are still *many* incidences at scale w_f. But the empty-strip condition says incidences are forbidden at the scale 4Δ/u. The only way to reconcile "many incidences at w_f" with "none at 4Δ/u" is w_f ≳ 4Δ/u, i.e. Δ ≲ u·w_f. Choosing the scales well — e.g. u ≍ n^{−1/3}, w_f ≍ n^{−3/4} — gives Δ ≲ n^{−13/12}, a *polynomial* saving below 1/n.

Everything hinges on step (B): why does the renormalized incidence count barely move between scales? This is the analytic core. Take the difference of strip indicators at two scales,

  ϕ_ℓ(x) = (1/w_i)·1_{T_ℓ(w_i)}(x) − (1/w_f)·1_{T_ℓ(w_f)}(x),

a function that integrates to zero across the strip. The key fact: for two different lines ℓ₁, ℓ₂, the functions ϕ_{ℓ₁} and ϕ_{ℓ₂} are nearly *orthogonal* — ∫ ϕ_{ℓ₁}ϕ_{ℓ₂} ≈ 0 — because two strips at different scales intersect in a parallelogram whose area is proportional to both widths, and the zero-mean cancellation kills the overlap. With a quasi-orthogonal system in hand, a Selberg/Bessel-type inequality bounds Σ_ℓ |⟨f, ϕ_ℓ⟩|² by ‖f‖², and unpacking that with f the point measure bounds exactly the change in renormalized incidence count between the two scales. That's step (B): the scale-to-scale stability is quasi-orthogonality of strip differences plus Bessel. Optimizing the scales through this machinery gives the polynomial exponent; pushing the optimization to its limit lands on Δ ≲ exp(c√log n)/n^{8/7} — exponent 8/7, i.e. the saving µ = 1/7 over 1/n.

Why 8/7 and not further? It's the optimum of the scale-parameter system in this two-scale incidence method: balancing the initial-estimate scale against the inductive shrink, the exponent that survives is 1 + 1/7. Roth's own analytic version got smaller savings (µ = 1−√(4/5) ≈ 0.106, then (9−√65)/8 ≈ 0.117); optimizing the same incidence framework pins it at µ = 1/7 ≈ 0.143.

Can the analytic core be sharpened past 8/7? The Selberg/Bessel comparison is one way to get scale-to-scale stability, but there's a more flexible one: the high-low method, which splits the incidence count into a low-frequency part (slowly varying, ≈ the random prediction) and a high-frequency part (the fluctuation), and bounds the fluctuation directly. It gives a comparison of the form

  |renorm. count at w_i − renorm. count at w_f| ≲ (M_P(w_f×w_f)·M_L(w_i×1)/(|P||L|))^{1/2}·w_f^{−3/2},

where M_P(w_f×w_f) is the most points in any w_f×w_f square and M_L(w_i×1) the most lines inside any w_i×1 tube. The message: the renormalized count is stable *unless* the points or lines concentrate — unless there's a small square crammed with points or a thin tube crammed with lines. So the obstruction to beating 8/7 is exactly *concentration*: a fractal-like configuration that is clustered at small scales yet spread out globally. To rule that out I need to control how much a set can concentrate while still spanning many directions — and that's a projection-theory question. The set of directions S(X) = {(x−y)/|x−y|} spanned by a planar set X has dimension ≥ min(1, dim X) (Marstrand; the sharper Orponen–Shmerkin–Wang refinement, a discretized cousin of the sum-product phenomenon). Feeding a discretized direction-set bound into the high-low comparison lets the induction reach finer scales than the Selberg machinery could, and the net effect is to push the exponent just past the KPS barrier: Δ ≲ n^{−8/7−1/2000} (and, for *homogeneous* point sets — at most O(1) points per n^{−1/2}×n^{−1/2} cell — all the way to n^{−7/6+o(1)}).

Let me step back and recap the causal chain, because the two halves rhyme. The whole subject runs on one object: small triangle ⟺ a point in a thin strip about a base line. On the *construction* side, that turned P(small triangle) into a clean linear-in-ε law, which fed the alteration method to Ω(1/n²); then the realization that deletion-one-per-triple is just a wasteful independent-set heuristic — and that the bad-triple hypergraph is *uncrowded* once you remove the close pairs and the sublinear set of short cycles — let the AKPSS independent-set theorem spend its (log t)^{1/2} gain to lift the threshold to Ω(log n/n²) and refute Heilbronn's conjecture. On the *impossibility* side, the same strips, now forced *empty* by a large Δ, are too much forbidden area: weighting by width and integrating gives the energy bound Δ ≲ n^{−1}(log n)^{−1/2}; replacing the soft energy count by a two-scale incidence comparison — pseudo-random at a coarse scale, empty at the fine scale, stable in between by quasi-orthogonality of strip differences — gives the first polynomial saving and, optimized, exponent 8/7; and replacing the Selberg comparison by the high-low method plus a projection-theoretic bound on direction sets pushes just past it. Floor Ω(log n/n²), ceiling n^{−8/7−1/2000}; the true exponent between 1 and 8/7 is still open.

Here is the computational skeleton: the basic alteration is literal, and the logarithmic construction uses the same hypergraph cleanup and independent-set selection that the proof certifies. For a finite run, exact search certifies the independent set; at the asymptotic scale, the AKPSS/BKHL selector occupies that same slot.

```python
import numpy as np
from itertools import combinations

def tri_area(p, q, r):
    return 0.5 * abs((q[0] - p[0]) * (r[1] - p[1])
                     - (q[1] - p[1]) * (r[0] - p[0]))

def small_triples(pts, eps):
    # Edges of the 3-uniform bad-area hypergraph.
    return [(i, j, k) for i, j, k in combinations(range(len(pts)), 3)
            if tri_area(pts[i], pts[j], pts[k]) <= eps]

def construct_by_deletion(n, c=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    N = 2 * n
    pts = rng.random((N, 2))
    eps = c / n**2
    alive = np.ones(N, dtype=bool)
    for i, j, k in small_triples(pts, eps):
        if alive[i] and alive[j] and alive[k]:
            alive[k] = False
    keep = np.flatnonzero(alive)
    if len(keep) < n:
        raise RuntimeError("this random trial did not certify n surviving points")
    return pts[keep[:n]], eps

def _forms_cycle(edge_sets):
    m = len(edge_sets)
    choices = []
    for a in range(m):
        inter = edge_sets[a] & edge_sets[(a + 1) % m]
        if not inter:
            return False
        choices.append(tuple(inter))

    def search(pos, used):
        if pos == m:
            return True
        for v in choices[pos]:
            if v not in used and search(pos + 1, used | {v}):
                return True
        return False

    return search(0, set())

def remove_crowding_obstructions(num_vertices, triples):
    # Brute-force finite version of the cleanup: delete vertices until no 2-, 3-, or 4-cycle remains.
    live = set(range(num_vertices))
    edges = [tuple(e) for e in triples]
    while True:
        edge_sets = [set(e) for e in edges]
        drop = None
        for length in (2, 3, 4):
            for idxs in combinations(range(len(edge_sets)), length):
                if _forms_cycle([edge_sets[i] for i in idxs]):
                    drop = next(iter(edge_sets[idxs[0]]))
                    break
            if drop is not None:
                break
        if drop is None:
            return sorted(live), edges
        live.discard(drop)
        edges = [e for e in edges if drop not in e]

def independent_subset(vertices, triples, target):
    # Exact certifier for small finite instances: find vertices containing no forbidden triple.
    vertices = list(vertices)
    edges = [frozenset(e) for e in triples]
    incident = {v: [e for e in edges if v in e] for v in vertices}
    chosen, best = set(), []

    def dfs(pos):
        nonlocal best
        if len(chosen) > len(best):
            best = list(chosen)
        if len(chosen) >= target:
            return True
        if pos == len(vertices):
            return False
        if len(chosen) + len(vertices) - pos <= len(best):
            return False

        v = vertices[pos]
        chosen.add(v)
        if not any(e <= chosen for e in incident.get(v, ())):
            if dfs(pos + 1):
                return True
        chosen.remove(v)
        return dfs(pos + 1)

    dfs(0)
    return best[:target]

def construct(n, delta=0.5, c=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    N = int(np.ceil(n ** (1 + delta)))
    pts = rng.random((N, 2))
    eps = c * np.log(max(n, 3)) / n**2
    triples = small_triples(pts, eps)
    vertices, triples = remove_crowding_obstructions(N, triples)
    keep = independent_subset(vertices, triples, n)
    if len(keep) < n:
        raise RuntimeError("this finite trial did not certify n independent vertices")
    return pts[keep], eps
```

So the construction is: drop a random over-sample, read off the thin-strip bad triples, strip the configuration down to an uncrowded hypergraph by removing the close pairs and short cycles, and use a certified independent-set routine — n points whose every triangle has area Ω(log n/n²). And the matching impossibility runs the identical strips backwards: a large minimum area forces too much empty strip, which the incidence/projection machinery forbids below n^{−8/7−1/2000}.
