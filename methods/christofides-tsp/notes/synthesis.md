# Synthesis: Christofides–Serdyukov 3/2-approximation for metric TSP

## The task / pain point
Metric TSP: complete graph on n cities, distance d symmetric, nonnegative, triangle inequality
d(x,y) <= d(x,z) + d(z,y). Find min-cost Hamiltonian cycle. NP-hard (Garey & Johnson 1979).
We can't solve exactly in poly time; want a tour provably within a constant factor of OPT.

## Two lower bounds we can actually compute (the engine of approximation analysis)
The whole game in min-approximation: to prove approx/OPT <= c we need a *computable lower bound*
on OPT. Two bounds:
1. **MST lower bound.** Delete one edge from any Hamiltonian cycle -> a spanning tree (a path,
   which is a spanning tree). So d(MST) <= d(OPT - e) <= d(OPT). MST computable in poly time
   (Prim/Kruskal). => d(MST) <= OPT.
2. **Sub-tour / matching bound.** OPT restricted (shortcut) to any subset S of vertices is a
   Hamiltonian cycle on S of cost <= OPT (triangle inequality). If |S| even, a cycle on S splits
   into two perfect matchings (alternate edges); min of the two <= half the cycle cost.

## Baseline: double-MST (twice-around), ratio 2
Build MST M. Walk around it keeping hand on the wall -> each edge traversed twice -> closed walk
of cost exactly 2 d(M), Eulerian (every vertex even degree because doubled). Shortcut repeated
vertices; triangle inequality => Hamiltonian tour T with
  d(T) <= 2 d(M) <= 2 d(OPT - e) <= 2 d(OPT).
Ratio 2. (Rosenkrantz–Stearns–Lewis 1977 give this and other heuristics.)

## The wall and the leap
The factor of 2 comes entirely from *doubling*. We double M only to make every degree even so an
Eulerian tour exists. But doubling is a sledgehammer: it fixes parity at *every* vertex, paying for
a whole second copy of the tree. Which vertices actually need fixing? Only the **odd-degree**
vertices O of M. The even-degree vertices are already fine. So instead of doubling all of M, just
add a *cheap* set of edges that flips the parity of exactly the odd vertices.

Adding a perfect matching on O flips each O-vertex's degree by exactly 1 (odd -> even); even
vertices untouched. So M + (perfect matching on O) is Eulerian. Key handshake fact: sum of degrees
= 2|E| is even, so the number of odd-degree vertices |O| is even => a perfect matching on O exists.
Take the **minimum-weight** perfect matching P on the complete graph over O (Edmonds' blossom
algorithm, poly time).

## Why d(P) <= OPT/2 (the crux)
Let N* be an optimal TSP tour on just O (a cycle through the odd vertices). |O| even => N* has an
even number of edges; 2-color them alternately into two perfect matchings N1, N2 of O, with
d(N1) + d(N2) = d(N*). P is the *minimum* perfect matching on O, so
  d(P) <= min(d(N1), d(N2)) <= (d(N1)+d(N2))/2 = d(N*)/2.
And N* (optimal tour on O) <= the tour got by shortcutting OPT down to O, which by triangle
inequality has cost <= d(OPT). So d(N*) <= d(OPT). Therefore d(P) <= d(OPT)/2.

## Putting it together: ratio 3/2
H = multiset of edges M ∪ P. Every vertex even degree (even vertices: only M edges, even; odd
vertices: their M-degree was odd, +1 from P => even). Connected (contains M). => Eulerian circuit
exists, cost d(M) + d(P) <= d(OPT) + d(OPT)/2 = (3/2) d(OPT). Shortcut to a Hamiltonian tour;
triangle inequality => cost not increased. Final tour <= (3/2) OPT. QED.

Running time O(n^3), dominated by the min-weight perfect matching (Edmonds' blossom; Serdyukov
used Karzanov's O(n^3 log n) implementation).

## Tightness
3/2 is tight: family of instances (e.g. path of n vertices, plus "skip-one" edges weight 1+eps)
where the ratio approaches 3/2. So the analysis can't be improved for this algorithm.

## Lineage / load-bearing ancestors (cite by author-year; in-frame)
- MST lower bound + twice-around 2-approx: folklore; Rosenkrantz, Stearns, Lewis (1977).
- Eulerian tour / parity / matching machinery: the **Chinese postman problem** — Edmonds & Johnson
  (1973), Christofides (1973), Serdyukov (1974): to traverse all edges of a graph cheaply you add a
  min-weight matching on the odd-degree vertices to make it Eulerian. This is EXACTLY the parity-fix
  trick, transplanted from "traverse all edges" to "fix MST parity".
- Edmonds (1965): poly-time max-weight matching in general graphs (blossom). Makes step 3 poly.
- Handshake lemma (Euler): |odd-degree vertices| is even.
- Euler (1736): Eulerian circuit exists iff connected + all degrees even.

## Historical facts (van Bevern & Slugina 2020, arXiv:2004.02437)
- Christofides: Management Science Research Report 388, Carnegie-Mellon Univ., 1976 (report dated
  Feb 1976); O(n^3); never formally journal-published.
- Serdyukov (1951–2001), Novosibirsk: submitted Jan 27, 1976, published 1978 (Russian); independent.
  Used Edmonds' algorithm via Karzanov's O(n^3 log n) implementation.
- Both came from the Chinese postman line (odd-vertex matching). Background only:
  Karlin–Klein–Oveis Gharan (STOC 2021, arXiv:2007.01409): first 3/2 - eps (eps>1e-36), randomized;
  samples a random spanning tree from max-entropy distribution then matches odd vertices — still the
  same parity-fix skeleton. Background gesture only; not part of the reconstruction.

## Canonical implementation (code/networkx_christofides.py)
networkx christofides(): minimum_spanning_tree -> remove even-degree nodes -> min_weight_matching on
the odd subgraph -> MultiGraph union -> eulerian_circuit -> _shortcutting. min_weight_matching
negates weights and calls Edmonds max_weight_matching with maxcardinality.

## Scaffold (pre-method) <-> final code correspondence
Pre-method skeleton: mst(G); build a tour from the tree (the slot the method fills); shortcut(walk).
Final code fills the slot with: odd_vertices -> min_weight_perfect_matching -> Euler -> shortcut.
