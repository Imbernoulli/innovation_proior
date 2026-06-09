# Context: approximating minimum-weight vertex cover

## Research question

Given an undirected graph G = (V, E) with nonnegative vertex weights w : V → Q₊, a *vertex cover* is a set
C ⊆ V such that every edge has at least one endpoint in C. The problem is to find a cover of minimum total
weight. In the unweighted (cardinality) case all w_v = 1 and we want the fewest vertices.

This problem is NP-complete — Karp (1972) put it on the original list of 21 NP-complete problems, and Garey,
Johnson and Stockmeyer (1976) sharpened this to show it stays NP-complete even on planar cubic graphs with
unit weights. So unless P = NP there is no polynomial algorithm that always returns the exact optimum. The
goal is therefore weaker but still demanding: a polynomial-time *approximation algorithm* that, on every
instance, returns a cover whose weight is provably within a guaranteed factor α of the optimum OPT.

What a solution must achieve is two things at once. First, a polynomial-time procedure that outputs a valid
cover. Second — and this is the real content — a *proof* that the output weight is ≤ α·OPT for a fixed
constant α, holding on all instances. The structural obstacle is that OPT itself is NP-hard to compute, so
the guarantee cannot be obtained by comparing to OPT directly. It has to come from a quantity that is
computable in polynomial time and is provably never larger than OPT: a *lower bound* LB ≤ OPT that the
algorithm can both compute and charge its output against, so that output ≤ α·LB ≤ α·OPT.

## Background

**Vertex cover as a 0/1 integer program.** Assign a variable x_v ∈ {0,1} to each vertex, 1 meaning "v is in
the cover". The covering requirement "every edge is hit" is one linear inequality per edge:

  minimize  Σ_{v∈V} w_v x_v
  subject to  x_u + x_v ≥ 1   for every edge (u,v) ∈ E
        x_v ∈ {0,1}      for every v ∈ V.

This integer program is an exact model of vertex cover — its optimum equals OPT — but solving an integer
program is itself NP-hard, and the difficulty is concentrated entirely in the integrality requirement
x_v ∈ {0,1}.

**Linear programming as the tractable cousin.** A linear program — a linear objective over linear
inequalities with *continuous* variables — is solvable in polynomial time (the ellipsoid method establishes
this in principle; interior-point methods give practical polynomial algorithms; the simplex method is the
practical workhorse and returns a *basic*, i.e. vertex-of-the-polyhedron, optimal solution). This is the
sharp dividing line: replace x_v ∈ {0,1} by 0 ≤ x_v ≤ 1 and the same covering problem becomes polynomially
solvable, at the cost of allowing fractional values. The upper bounds x_v ≤ 1 do not change the optimum:
truncating any feasible value above 1 down to 1 preserves all edge constraints and never increases a
nonnegative objective.

**Weak duality and the lower-bound machine.** Every linear program has a dual; for a covering LP (nonnegative
costs, ≥-constraints) the dual is a packing LP, and weak duality says the value of any dual-feasible
(packing) solution is ≤ the value of any primal-feasible (covering) solution. For vertex cover the dual of
the LP is a *fractional edge packing* — assign y_e ≥ 0 to edges so that each vertex receives total weight
≤ w_v — and an *integral* edge packing is exactly a matching. So matchings and edge packings are the natural
certificates of a lower bound on cover weight.

**The unweighted combinatorial precursor.** Gavril's algorithm (reported as a private communication in Garey
and Johnson 1979) handles the unweighted case: compute any *maximal* matching M (greedily; not necessarily
maximum) and output both endpoints of every matched edge. This is the seed of the whole approach. Two
observations make it work: a single vertex can cover at most one edge of a matching, so |M| ≤ OPT; and taking
both endpoints of every matched edge yields a feasible cover, since any
uncovered edge would have both endpoints unmatched and could be added to M, contradicting maximality. The
output has 2|M| vertices, giving 2|M| ≤ 2·OPT.

**The fractional gap is real.** Relaxing integrality genuinely changes the optimum. On the triangle K₃ with
unit weights, an integral cover needs 2 vertices, but setting every x_v = 1/2 satisfies every edge
constraint (1/2 + 1/2 = 1) at total cost 3/2 < 2. So the fractional optimum can be strictly below the
integral optimum — the relaxation is a true relaxation, not an exact reformulation.

**A structural feature of the vertex-cover polyhedron.** Beyond merely being solvable, the vertex-cover LP
has special structure: each edge constraint touches exactly two variables, and each bound constraint touches
one. This two-variables-per-inequality shape is unusual among linear programs and is worth keeping in view
when reasoning about the extreme points the simplex method returns.

## Baselines

**Gavril's maximal-matching algorithm (unweighted vertex cover).** Core idea and math as above:
|cover| = 2|M| ≤ 2·OPT for any maximal matching M. *Gap:* it is intrinsically unweighted. "Both endpoints of
a maximal matching" has no weighted analogue — a matched edge between a very heavy and a very light vertex
forces the heavy one into the cover, and the resulting weight can exceed the optimum by an unbounded factor.
A weighted 2-approximation needs a different lower bound than "number of matched edges".

**The greedy heuristics.** The natural unweighted heuristic repeatedly takes a maximum-degree vertex; the
natural weighted heuristic takes a minimum weight-per-newly-covered ratio. These are the set-cover greedy
algorithms (Johnson 1974, Lovász 1975, Chvátal 1979) specialized to vertex cover. *Gap:* greedy gives only a
factor of H(d) = 1 + ½ + ⋯ + 1/d ≈ ln d (d = max degree), i.e. logarithmic in the worst case, not a
constant. Johnson exhibited graphs of maximum degree k on which max-degree greedy returns roughly H(k)·OPT
even with unit weights. For vertex cover specifically a constant factor should be reachable, so the
logarithmic greedy is a baseline to beat.

**Independent-set / largest-independent-set complementation.** V minus a maximum independent set is a minimum
vertex cover, and various heuristics partition V into independent sets {V₁,…,V_k} and take V∖V_i of smallest
weight as a cover. *Gap:* the ratio w(V∖V_i)/OPT can be made arbitrarily large (e.g. a weighted 4-path), so
this gives no constant guarantee on its own.

## Evaluation settings

The natural yardstick is worst-case approximation ratio: the supremum over all instances of (weight of cover
returned)/OPT, proved analytically rather than measured. Instances are weighted and unweighted graphs G with
vertex weights w : V → Q₊. Relevant input regimes for the analysis are general graphs, planar/cubic graphs
(where exactness is still NP-hard), and bipartite graphs (where the LP is integral and the problem is
polynomial via König's theorem). The candidate lower bounds in play — the matching/edge-packing certificate
and the optimum of the LP relaxation, computed by a polynomial LP solver (ellipsoid / interior point /
simplex) — all exist independently of any particular cover algorithm, and so do their running times.

## Code framework

The graph, the weights and an LP solver already exist. The unfinished procedure must return an integral cover
together with whatever certificate is used to bound its cost.

```python
import networkx as nx

def build_vertex_cover_lp(G, w):
    """The 0/1 integer program with integrality relaxed to 0 <= x_v <= 1:
       min  sum_v w[v]*x[v]   s.t.  x[u]+x[v] >= 1 for (u,v) in E,  0 <= x[v] <= 1.
    Returns the LP in a solver-ready form (objective, edge constraints, bounds).
    The LP is solvable in polynomial time; assume a generic LP oracle below."""
    pass  # TODO: assemble objective sum_v w[v]*x[v] and one constraint x[u]+x[v]>=1 per edge

def solve_lp(lp):
    """Polynomial-time LP oracle. Returns a basic (extreme-point) optimal solution x*
       and the optimal value OPT_f = sum_v w[v]*x*[v]."""
    pass  # TODO: call an LP solver; request or recover a basic optimum when needed

def fractional_to_cover(x_star):
    """Produce an integral vertex cover and the certificate that bounds its cost."""
    pass  # TODO

def vertex_cover_approx(G, w):
    lp = build_vertex_cover_lp(G, w)
    x_star, opt_f = solve_lp(lp)
    C = fractional_to_cover(x_star)
    return C
```
