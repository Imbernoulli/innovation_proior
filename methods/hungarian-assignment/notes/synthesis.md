# Synthesis — Hungarian (Kuhn–Munkres) assignment

## Pain point / goal
n×n cost (or rating) matrix; pick one entry per row and column (a permutation) minimizing
(or maximizing) the sum. It IS a linear program (assignment polytope = doubly stochastic =
Birkhoff), but the simplex on a 10×10 instance was ~20 eqns / 100 vars and degenerate; want a
special combinatorial method that beats simplex and produces an OPTIMALITY CERTIFICATE.

## Lineage (sourced)
- **König 1916/1931/1936 (Theorie der Graphen)**: in a bipartite graph (0/1 matrix), max # of
  independent 1's (no two in a line) = min # of lines (rows/cols) covering all 1's. = max matching
  = min vertex cover. This is LP duality decades before Dantzig. Constructive, polynomial.
- **Egerváry 1931** (Kuhn translated it 1953): generalized König from 0/1 to integer-weighted
  matrices via integer dual potentials u_i, v_j with u_i+v_j ≥ r_ij; reduce general problem to a
  sequence of König (0/1) problems on the "tight" positions u_i+v_j=r_ij.
- **LP duality / complementary slackness** (Gale–Kuhn–Tucker 1948-51): primal min Σc_ij x_ij s.t.
  row/col sums =1, x≥0; dual max Σu_i+Σv_j s.t. u_i+v_j ≤ c_ij. CS: optimal iff matching uses only
  edges with c_ij-u_i-v_j=0.
- **Total unimodularity**: the bipartite incidence (row-sum/col-sum) constraint matrix is TU, so
  every extreme point of the LP relaxation is integral → LP relaxation solves the IP. (Birkhoff:
  doubly-stochastic = convex hull of permutation matrices.)
- **Augmenting paths / Berge 1957**: a matching is maximum iff no augmenting (alternating, both
  ends exposed) path; symmetric-difference along an augmenting path grows the matching by 1.
- **Munkres 1957**: recast Kuhn's procedure as the matrix tableau (row reduce, col reduce, cover
  zeros with min lines, subtract/add min uncovered), and PROVED it is O(n^3) strongly polynomial
  (Kuhn 1955 proved only finiteness).

## The intellectual move
- Reduced cost w_ij = c_ij - u_i - v_j ≥ 0 is dual feasibility. Tight edges (w_ij=0) are the only
  candidates by complementary slackness.
- Find a max matching on the tight subgraph via augmenting paths (König-step).
- If perfect → CS satisfied → optimal (certified by the dual).
- If not perfect → König gives a min vertex cover; the uncovered region has all w_ij>0; let
  δ=min uncovered w_ij; update duals (decrease u on one side / increase v) so at least one new edge
  becomes tight, strictly improving the dual objective. Repeat.
- This is the Egerváry/Kuhn dual-update = the "subtract min uncovered, add to covered" matrix step.

## Two faithful code forms (both verified vs brute force, 300 random matrices)
- (a) primal-dual O(n^3) potential / shortest-augmenting-path (cp-algorithms): per row, grow an
  alternating tree, maintain minv[j]=min slack, delta=min minv over unused cols, update
  u[p[j]]+=delta, v[j]-=delta on used, minv[j]-=delta on unused; stop at free column; backtrack way[].
- (b) classic matrix: row-min subtract, col-min subtract, max-matching on zeros (Kuhn's bipartite
  augmenting), König min cover, d=min uncovered, subtract d from uncovered rows + add d to covered
  cols (equivalently subtract uncovered/uncovered, add covered/covered), repeat.
  NOTE sign convention: subtract from UNCOVERED rows, add to COVERED columns (Munkres step 6 form
  is "add to covered rows, subtract from uncovered cols" — dual; both correct, mirror images).

## Example
Kuhn retrospective R=[[8,7,9,9],[5,2,7,8],[5,1,4,8],[2,2,2,6]] is a MAXIMIZATION; optimum 25 at
assignment (1→1,2→3,3→4,4→2). Negate for the min-form solvers. Verified.

## In-frame notes
- Never cite Kuhn 1955 / Munkres 1957 as artifacts. König (1916/1931), Egerváry (1931), Birkhoff
  (1946), Berge (1957), von Neumann (1953), Dantzig (1951), Gale-Kuhn-Tucker LP duality are
  citable ancestors.
- The method name "Hungarian Method" may appear in answer.md.
