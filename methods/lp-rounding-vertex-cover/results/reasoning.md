Let me start from what actually hurts. I have a graph G = (V, E), each vertex carries a weight w_v ≥ 0, and I want the cheapest set of vertices that touches every edge — a minimum-weight vertex cover. And this is NP-complete; Karp put it on the original list, and it stays hard even on planar cubic unit-weight graphs. So I am not going to compute the optimum exactly in polynomial time, full stop, unless P = NP. I have to settle for a cover that I can *prove* is within some fixed factor of optimal on every instance. That word "prove" is the whole problem. I can write down a thousand heuristics that look reasonable, but a guarantee means I bound (weight of what I output) divided by OPT, and OPT is exactly the thing I can't get my hands on. So how do I bound a ratio against a number I can't compute?

The only way out I can see is to find some *other* quantity LB that (a) I can compute in polynomial time and (b) I can prove is ≤ OPT, never above it. Then if I also arrange that my output ≤ α·LB, I get output ≤ α·LB ≤ α·OPT, and I never had to know OPT. The whole game is: find a computable lower bound on the optimum, and tie my algorithm's cost to that same lower bound. Two jobs for one quantity.

Where would a lower bound on the cost of a cover come from? Let me think about the unweighted case first, because it's cleaner. Suppose I have a matching M — a set of edges with no shared endpoints. Any cover has to hit every one of those edges, and because the matched edges share no vertices, a single vertex can cover at most one of them. So a cover needs at least |M| vertices just to deal with M. There it is: |M| ≤ OPT, and |M| is trivially computable. That's my lower bound.

Now can I turn that lower bound into a cover whose size I can charge to it? Take a *maximal* matching — keep adding disjoint edges greedily until I can't. Then output both endpoints of every matched edge. Is that even a cover? Suppose some edge (u,v) is uncovered. Then neither u nor v is an endpoint of any matched edge — but then (u,v) shares no vertex with M, so I could have added it, contradicting maximality. So it is a cover. And its size is exactly 2|M| ≤ 2·OPT. A 2-approximation, clean, combinatorial. Good.

But the moment I put weights back in, this falls apart. "Both endpoints of every matched edge" doesn't care about weight. Picture a matched edge between a vertex of weight 1000 and a vertex of weight 1 — I'm forced to pay for the 1000 even when the cheap endpoint would have done. The number of matched edges is no longer the right currency; |M| ≤ OPT is still true as a count, but 2|M| has nothing to do with the *weight* I end up paying. So the matching lower bound, which was about counting edges, doesn't translate to a weighted lower bound. I need a lower bound that speaks in the currency of weights.

Let me back all the way up and write the problem down as honestly as I can. A cover is a yes/no decision per vertex. Let x_v ∈ {0,1}, x_v = 1 meaning "v is in the cover." "Every edge is hit" means: for each edge (u,v), at least one of x_u, x_v is 1, i.e. x_u + x_v ≥ 1. And I'm minimizing total weight. So vertex cover *is*

  min Σ_v w_v x_v   s.t.  x_u + x_v ≥ 1 for every edge (u,v),  x_v ∈ {0,1}.

This is exact — its optimum is literally OPT. But it's an integer program, and integer programs are NP-hard to solve; in fact all the difficulty is sitting in that x_v ∈ {0,1}. The objective is linear, the constraints are linear; the only nonlinear, nasty thing is the demand that the variables be integers.

So stare at that. If I dropped "integer" and just let the variables be real numbers in [0,1], the whole thing would become a linear program — linear objective, linear inequalities, continuous variables — and *those* I can solve in polynomial time. The upper bound x_v ≤ 1 is value-preserving rather than essential: if some feasible solution has x_v > 1, truncating that coordinate down to 1 keeps every incident edge constraint satisfied and never increases a nonnegative objective. Keeping the box 0 ≤ x_v ≤ 1 makes the relaxation and the structure clean:

  min Σ_v w_v x_v   s.t.  x_u + x_v ≥ 1 for every edge (u,v),  0 ≤ x_v ≤ 1.

Call its optimum OPT_f. Now here's the question that decides whether this was worth anything: how does OPT_f relate to OPT? I removed the integrality constraint. Removing a constraint can only enlarge the feasible region — every integral cover {0,1} is still a feasible point of the relaxed problem (0 and 1 are in [0,1], and they satisfy the same edge inequalities). I'm minimizing over a *superset* of the points I had before. Minimizing over a bigger set can only give a smaller-or-equal optimum. So

  OPT_f ≤ OPT.

That's the lower bound I was hunting for, and this time it's stated in weights, not in edge-counts — it's the minimum *weighted* objective over the relaxed program. And OPT_f is computable: it's a linear program, polynomial time. So the relaxation hands me exactly an LB ≤ OPT that I can compute. Half the job done — and notice this lower bound exists for *weighted* graphs with no extra effort, which is precisely where the matching argument died.

Is the gap real, or did I just rewrite the same number? Let me test the triangle, three vertices, unit weights. Integrally I need 2 vertices to cover all three edges. But fractionally I can set every x_v = 1/2: each edge gets 1/2 + 1/2 = 1, satisfied, at total cost 3/2. So OPT_f = 3/2 < 2 = OPT. The relaxation genuinely undershoots — it's a real relaxation, not a disguise. Good: that 3/2 is a legitimate lower bound, and the algorithm I'm about to build will have to live with the fact that the fractional optimum can sit strictly below the integral one.

Now I have a fractional solution x* — a vector of real numbers, x*_v ∈ [0,1], one per vertex, satisfying every edge constraint, of total weight OPT_f ≤ OPT. It is *not* a cover (you can't put "half a vertex" in a cover). But it's suggestive. I want to convert it into an honest 0/1 cover without paying much more than OPT_f. The natural thing is to round: turn the fractions into 0s and 1s.

Which fractions do I round up? Let me look at a single edge constraint, x*_u + x*_v ≥ 1, and ask what it forces. Two nonnegative numbers summing to at least 1 — they can't both be small. The smaller of them could be 0, but then the larger is ≥ 1; and the most balanced case is 1/2 each. In every case, the *larger* of x*_u, x*_v is at least 1/2. So: every edge has at least one endpoint with x* ≥ 1/2. That's not a coincidence of the triangle — it's forced by the constraint itself. For every single edge, at least one endpoint already carries fractional weight ≥ 1/2.

That tells me exactly whom to keep. Let C = { v : x*_v ≥ 1/2 }. Round those up to 1, drop the rest to 0.

First, is C a cover? Take any edge (u,v). I just argued at least one of x*_u, x*_v is ≥ 1/2, so at least one endpoint is in C, so the edge is covered. Every edge — so C is a feasible vertex cover. No appeal to maximality, no greedy process; it falls straight out of the edge inequalities.

Second, what does it cost? Compare the cost of C to OPT_f vertex by vertex. If x*_v ≥ 1/2, I round it up to 1, and 1 ≤ 2·x*_v exactly because x*_v ≥ 1/2 — so I pay w_v ≤ 2·w_v x*_v on that vertex. If x*_v < 1/2, I drop it; I pay 0, which is certainly ≤ 2·w_v x*_v. Either way, the weight I assign to v is at most twice w_v x*_v. Summing over all vertices,

  w(C) = Σ_{v∈C} w_v ≤ Σ_v 2·w_v x*_v = 2·OPT_f.

And I already proved OPT_f ≤ OPT, so

  w(C) ≤ 2·OPT_f ≤ 2·OPT.

A 2-approximation for *weighted* vertex cover. The thing that broke the matching argument — weights — is handled, because the lower bound OPT_f was a weighted lower bound from the start, and the rounding charged each vertex against its own fractional weight. Both jobs done by the LP: it gave me the computable lower bound, and it gave me the fractional solution to round.

Let me make sure I see why the threshold is exactly 1/2 and not something I pulled out of the air. The covering inequality has two terms summing to ≥ 1; by pure averaging, at least one of the two is ≥ 1/2. If instead I'd had an inequality with p terms summing to ≥ 1 — which is what a general set-cover constraint looks like, an element lying in p sets — then at least one term is ≥ 1/p, and rounding everything ≥ 1/p up to 1 would multiply costs by at most p and give a p-approximation. Vertex cover is the case where every "element" (edge) lies in exactly p = 2 sets (its two endpoint-vertices), so the threshold is 1/2 and the factor is 2. The 1/2 isn't a tuning choice; it's 1 over the number of vertices on an edge.

There's an even lazier rounding I should check, because it might be cleaner: round up *every* nonzero coordinate, C' = { v : x*_v > 0 }. Does that still work, and what does it cost? Feasible, obviously — it's a superset of my C. Cost is the problem: I'd need x*_v > 0 to imply x*_v is not too small, and for an arbitrary fractional optimum that is false. A tiny positive coordinate would make the step 1 ≤ 2·x*_v fail completely. So before I know anything more about the structure of x*, "round all nonzero up" has no clean factor-2 proof. The ≥ 1/2 threshold is the safe rule because the inequality 1 ≤ 2·x*_v is built into the threshold itself.

Now I want to look harder at that fractional optimum, because something about the triangle bugs me in a productive way: x* = (1/2, 1/2, 1/2). All halves. Is that an accident of symmetry, or does the LP *like* to produce halves? Let me think about what an extreme point — a vertex of the boxed feasible polyhedron, the kind of solution simplex returns — can look like. Suppose I have a feasible x in 0 ≤ x ≤ 1 that is *not* half-integral: some coordinates are strictly between 0 and 1/2, or strictly between 1/2 and 1. I want to know whether such an x can ever be an extreme point.

Collect the offending vertices into two groups by which side of 1/2 they're on:
  V₊ = { v : 1/2 < x_v < 1 },  V₋ = { v : 0 < x_v < 1/2 }.
By assumption at least one of these is nonempty. Now perturb: pick a tiny ε > 0 and define two new solutions,
  y: add ε on V₊, subtract ε on V₋, leave everything else;
  z: subtract ε on V₊, add ε on V₋, leave everything else.
Plainly x = (y + z)/2 — the perturbations cancel in the average — and for small enough ε, y ≠ x ≠ z because V₊ ∪ V₋ is nonempty. If I can show y and z are both *feasible*, then x is a midpoint of two distinct feasible points, hence not an extreme point. So let me check feasibility.

Bounds: the moved coordinates are strictly inside (0,1/2) or (1/2,1), bounded away from 0, 1/2 and 1, so a small enough ε keeps them in [0,1] and keeps them on the same side of 1/2. Fine.

Edge constraints: take an edge (u,v). If it was slack, x_u + x_v > 1, then a small ε can't push the sum below 1 — choose ε smaller than half the slack and it survives in both y and z. The only edges I have to worry about are the *tight* ones, x_u + x_v = 1. What can a tight edge look like, given that the moved coordinates live strictly inside (0,1/2) ∪ (1/2,1)? If both endpoints are moved and both were in V₋, their sum would be < 1, so the edge couldn't be tight; if both were in V₊, their sum would be > 1. So a tight edge with both endpoints moved must have one endpoint in V₊ and one in V₋, and then in y one gets +ε and the other −ε, so y_u + y_v = x_u + x_v = 1 still, and likewise for z. If exactly one endpoint were moved and the other were half-integral, tightness would be impossible: a strict value below 1/2 needs a partner strictly above 1/2, but the half-integral options are only 1/2 and 1, giving a sum below 1 or above 1; a strict value above 1/2 similarly cannot be completed to exactly 1 by 0, 1/2 or 1. If neither endpoint is moved, the tight half-integral pairs are {1/2, 1/2}, {0, 1}, and {1, 0}, and nothing changes. In every case the tight constraint stays feasible, and the cases with moved endpoints stay exactly tight.

So y and z are both feasible, distinct from x, and average to x. Therefore any feasible x that is not half-integral is *not* an extreme point. Contrapositive: every extreme-point solution of the vertex-cover LP is half-integral — each coordinate is 0, 1/2, or 1. The triangle wasn't symmetry luck; the polyhedron has no fractional vertices other than ones built from halves.

This is lovely, and it makes the rounding even more obviously right. Simplex (or any basic-solution LP method) returns an extreme point, so it returns a solution whose coordinates are already only 0, 1/2, 1. There's no general "fractional mess" to round — only halves to deal with. Round every 1/2 up to 1, leave the 0s and 1s; the cost can at worst double (each rounded coordinate went 1/2 → 1, a factor of exactly 2; the integer coordinates didn't move at all), and the 2-approximation is immediate, with the cost analysis collapsing to "1 ≤ 2·(1/2)."

Let me give a second, independent proof of the half-integrality, because I want to be sure it isn't an artifact of my perturbation choices — and because the constraint matrix has a very particular shape worth exploiting. A basic solution is obtained by taking a square nonsingular active-constraint matrix B and solving Bx = b. The active rows are tight edge constraints and tight bounds. After multiplying rows by −1 where convenient, every entry is in {0, ±1}; an edge row has two nonzeros, and a bound row has one. So every row of B has at most two nonzeros.

The determinant fact has to be used block by block, not carelessly on a whole separable matrix. If a square matrix with this row structure can be permuted into block diagonal form, the linear system splits into independent blocks, and Cramer's rule should be applied inside each block. So I only need to understand a nonseparable square block. Induct on its size. A 1×1 block has determinant 0, 1 or −1. If a larger block has a row or column with no nonzero entry, its determinant is 0; if it has a row or column with exactly one nonzero entry, expanding along that row or column reduces to a smaller block, so the determinant stays in {0, ±1, ±2}. The remaining case is the real two-variable case: every row and every column has exactly two nonzeros. Then the incidence pattern is a single cycle. After permuting rows and columns, row i has its two nonzeros in columns i and i+1, with indices modulo m. In the determinant expansion, only two permutations can choose one nonzero from every row and every column: the product around the "diagonal" cycle and the product around the shifted cycle. Each product is ±1, so their signed sum is 0 or ±2. Thus every nonsingular nonseparable block has determinant ±1 or ±2.

Now solve the basis system block by block. In each nonsingular block, Cramer's rule says each coordinate has denominator 1 or 2, because the block determinant is ±1 or ±2 and the right-hand side is integral. Therefore every basic coordinate is an integer or a half-integer; the bounds 0 ≤ x_v ≤ 1 leave only 0, 1/2 and 1. Half-integrality again, from the two-variables-per-inequality structure. And note where the "2" comes from: it is the cycle block created by rows with two variables. A general set-cover LP, with many variables in a row, has no reason to have this determinant bound.

This two-per-row structure also tells me I don't even need a general LP solver. With coefficients {0, ±1} and two per row, the vertex-cover LP is essentially a flow/cut object. Concretely: split each vertex v into two copies a_v, b_v, give each copy weight w_v/2, and replace each edge (i,j) by the two edges (a_i, b_j) and (a_j, b_i) — a bipartite double cover. A minimum-weight vertex cover in this bipartite graph is computable by min-cut, using the weighted bipartite vertex-cover/min-cut equivalence. It maps back to the original by setting x_v = 1 if both a_v and b_v are chosen, x_v = 1/2 if exactly one is chosen, and x_v = 0 if neither is chosen. With copy weight w_v/2, the bipartite-cover cost is exactly Σ_v w_v x_v, and the values are exactly {0, 1/2, 1}. So the LP relaxation of vertex cover is solvable by a min-cut computation and produces the half-integral optimum directly.

One more structural fact, because it sharpens what the halves mean. Look at the partition the optimal half-integral x* induces: P = {x*_v = 1}, Q = {x*_v = 1/2}, R = {x*_v = 0}. I want to prove there is an optimal *integer* cover that agrees with x* on P and R — it contains every vertex in P, excludes every vertex in R, and only has to decide what to do on Q.

First note the local constraint forced by the partition: R has no edges to R or Q, because 0+0 and 0+1/2 are both less than 1. Every neighbor of an R-vertex is in P. Now take any optimal integer cover S. Let A = P \ S be the vertices that the cover surprisingly omits from P, and let B = R ∩ S be the vertices that it surprisingly includes from R. If w(A) were larger than w(B), I could improve the fractional optimum: lower every vertex of A from 1 to 1/2 and raise every vertex of B from 0 to 1/2, leaving all other coordinates unchanged. The objective change would be −w(A)/2 + w(B)/2 < 0.

Check feasibility of that fractional exchange. Edges not incident to A are unchanged or have an endpoint raised, so they stay feasible. If an edge is incident to p ∈ A, then p is not in S, so the other endpoint must be in S because S is a cover. That other endpoint is either in P ∩ S and remains 1, or in Q ∩ S and remains 1/2, or in B and has just been raised from 0 to 1/2; it cannot be in R \ B, because then neither endpoint would be in S. After p drops to 1/2, the edge sum is still at least 1 in all cases. So the exchange would produce a feasible LP solution of smaller cost, contradicting optimality of x*. Therefore w(A) ≤ w(B).

Now replace S by S' = (S ∪ P) \ R. This forces in every omitted P-vertex and removes every included R-vertex. It is still a cover: edges touching R are covered by their P-neighbor, and all other edges were already covered unless their only covered endpoint was in R, which cannot happen. Its cost changes by w(A) − w(B) ≤ 0. Since S was already an optimal integer cover, S' is optimal too. That proves the persistency claim: some optimal integer cover agrees with the 0/1 part of x*, and only Q is genuinely undecided. The rounding cost increase lives exactly on Q, where the fractional solution pays w(Q)/2 and rounding all halves up pays w(Q).

Let me make sure the whole chain is airtight, end to end, because each link mattered. (1) Vertex cover is the 0/1 program min Σ w_v x_v s.t. x_u + x_v ≥ 1, x_v ∈ {0,1}; its optimum is OPT. (2) Relax x_v ∈ {0,1} to 0 ≤ x_v ≤ 1: the feasible region only grows (every integral cover is still feasible), so the LP optimum OPT_f ≤ OPT — a computable lower bound, in weight units. (3) Solve the LP, get x* with x*_u + x*_v ≥ 1 on every edge, so every edge has an endpoint with x* ≥ 1/2. (4) Keep C = {v : x*_v ≥ 1/2}: feasible because every edge has such an endpoint; and rounding x*_v ≥ 1/2 up to 1 costs ≤ 2·w_v x*_v per vertex while dropped vertices cost 0 ≤ 2·w_v x*_v, so w(C) ≤ 2·Σ w_v x*_v = 2·OPT_f ≤ 2·OPT. Two-approximation. (5) Moreover every extreme-point x* is half-integral (perturbation proof; or the determinant-block proof from the two-per-row structure), so simplex/min-cut hands me a {0, 1/2, 1} solution and the rounding is literally "round the halves up," doubling at worst on the half-valued set.

Here is the landing form — solve the relaxation, then keep the ≥ 1/2 vertices:

```python
import pulp  # any LP oracle; the relaxation is poly-time solvable

def vertex_cover_lp_rounding(G, w):
    # (1)-(2) Build the LP relaxation of the 0/1 vertex-cover IP:
    #   min sum_v w[v] x[v]  s.t.  x[u]+x[v] >= 1 for (u,v) in E, 0 <= x[v] <= 1.
    # Dropping x in {0,1} to 0 <= x <= 1 only enlarges the feasible set, so OPT_f <= OPT:
    # the LP optimum is a computable, weighted lower bound on the integral optimum.
    prob = pulp.LpProblem("vc_relaxation", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x_{v}", lowBound=0, upBound=1) for v in G.nodes}
    prob += pulp.lpSum(w[v] * x[v] for v in G.nodes)                  # objective
    for u, v in G.edges:
        prob += x[u] + x[v] >= 1                                      # every edge constraint
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))                # (3) solve the relaxation
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP solve failed with status {pulp.LpStatus[status]}")

    # (4) Rounding: every edge has x*_u + x*_v >= 1, so at least one endpoint has x* >= 1/2.
    # Keep exactly those. Feasible (each edge has such an endpoint); and 1 <= 2*x*_v there,
    # while dropped vertices cost 0 <= 2*w[v]*x*_v, so w(C) <= 2*OPT_f <= 2*OPT.
    tol = 1e-9
    C = {v for v in G.nodes if x[v].value() >= 0.5 - tol}             # round the halves up
    return C
```

The causal chain in one breath: I needed a guarantee but couldn't see OPT, so I needed a computable lower bound; writing vertex cover as a 0/1 program and *relaxing integrality* to a linear program both (a) made it polynomially solvable and (b) gave OPT_f ≤ OPT for free, in weighted units, where the matching count had failed; the relaxed solution satisfies x_u + x_v ≥ 1 on every edge, which forces an endpoint ≥ 1/2 on every edge, so keeping the ≥ 1/2 vertices is automatically a feasible cover and rounding them up at most doubles their fractional cost, yielding w(C) ≤ 2·OPT_f ≤ 2·OPT; and because the vertex-cover polytope's extreme points are half-integral (two variables per inequality give nonseparable determinant blocks with absolute value at most 2, hence coordinates in {0, 1/2, 1}), the fractional optimum is already only halves and integers, so "round the halves up" is the entire algorithm and the factor 2 lives exactly on the half-valued vertices.
