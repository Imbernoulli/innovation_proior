## Research question

Given an `n × n` matrix of numbers `r_ij` — read as the rating (or, after a sign flip, the cost)
of putting individual `i` in job `j` — choose exactly one entry in each row and each column so that
the sum of the chosen entries is as large (for ratings) or as small (for costs) as possible. A
choice of one entry per row and column is a **permutation** `j_1, …, j_n` of the columns, and we
want the permutation maximizing `r_{1 j_1} + ⋯ + r_{n j_n}`. This is the **assignment problem**:
assign `n` people to `n` jobs, one each, with the total rating maximal.

Why it matters and why it is not trivial: the number of permutations is `n!`, which is hopeless to
enumerate even for moderate `n` — `12! ≈ 4.8 × 10^8`. The problem is a linear program (its feasible
set is exactly the doubly-stochastic matrices, the convex hull of the permutation matrices), so in
principle the simplex method solves it; but a `10 × 10` instance is a linear program with `100`
nonnegative variables and `20` equality constraints, and it is highly **degenerate** (the
assignment problem is the most degenerate special case of the transportation problem), which makes
general-purpose simplex slow and awkward on it. What a good solution must achieve is therefore: an
algorithm that exploits the special combinatorial structure to run far faster than general simplex,
that produces an **integral** assignment directly (a real permutation, not a fractional one), and —
most valuable of all — that comes with a **certificate of optimality** so that when it stops, one
can verify by hand that no better assignment exists. The practical yardstick is an exact solver
whose certificate is small enough to check by hand and whose work does not grow like `n!`.

## Background

**The problem as a linear program, and its dual.** Encode an assignment by a 0/1 matrix `x_ij`
with `x_ij = 1` when `i` is given job `j`. The constraints are the row sums `Σ_j x_ij = 1` (each
person gets one job) and column sums `Σ_i x_ij = 1` (each job goes to one person), with
`x_ij ≥ 0`. Maximizing `Σ_ij r_ij x_ij` over these is the primal. By the duality of linear
programming — first rigorously proved by Gale, Kuhn and Tucker around 1948–51 in their study of
linear programs and matrix games — the dual assigns a number `u_i` to each row and `v_j` to each
column, requires `u_i + v_j ≥ r_ij` for every `(i, j)`, and minimizes `Σ_i u_i + Σ_j v_j`. Weak
duality is immediate and is the seed of everything: for any feasible assignment and any feasible
`(u, v)`,
`Σ_{(i,j) chosen} r_ij ≤ Σ_{(i,j) chosen} (u_i + v_j) = Σ_i u_i + Σ_j v_j`,
because each row and each column occurs exactly once in an assignment. So **any** dual-feasible
`(u, v)` upper-bounds the value of **any** assignment; if we can drive a particular assignment up
to meet a particular dual down, both are optimal and we have a certificate.

**Birkhoff / von Neumann: the relaxation is exact.** Birkhoff (1946) showed the doubly-stochastic
matrices are exactly the convex hull of the permutation matrices, and von Neumann (1953) recast the
assignment problem as a zero-sum game; together these say the linear program's optimum is attained
at a permutation matrix — the **integrality** we need is built into the geometry. Stated in
constraint-matrix terms: the bipartite row-sum/column-sum constraint matrix is **totally
unimodular** (every square submatrix has determinant 0, ±1), so every vertex of the feasible
polytope has integer coordinates, and a 0/1 vertex of this polytope is a permutation matrix. Hence
the linear-programming relaxation of the integer assignment problem has no integrality gap — an
algorithm can respect the combinatorics rather than rounding a fractional LP optimum.

**König's theorem (the 0/1 / qualification case).** The combinatorial heart predates linear
programming. Take the special case where ratings are only `0` and `1` — a **qualification matrix**,
`1` meaning person `i` can do job `j`. Now the question is purely: what is the largest number of
`1`'s one can choose with no two in the same row or column (a set of *independent* marks)? D. König
(working in the 1910s–1930s, collected in his 1936 *Theorie der Graphen*) proved the min–max
identity: in a bipartite graph the **maximum number of independent marks equals the minimum number
of lines (rows or columns) needed to cover all the marks**. Equivalently, maximum matching = minimum
vertex cover in a bipartite graph. This is an instance of linear-programming duality proved decades
before Dantzig formulated linear programming, and König's proof is *constructive and runs in
polynomial time* — the first such combinatorial-optimization algorithm. Its constructive content:
start from any set of independent marks; an unmatched person who can do an unassigned job extends
it; otherwise a chain of reassignments — a sequence of "transfers" that frees a column — either
extends the matching or terminates, and when no transfer can free a column the current marks are
maximum, and the lines through the "essential" rows/columns form a cover of exactly that size.

**Egerváry's generalization (integers).** J. Egerváry (1931; the relevant paper was translated from
Hungarian in 1953) extended König from 0/1 matrices to integer-weighted ones. His device is exactly
the dual potentials: maintain integers `u_i, v_j` with `u_i + v_j ≥ r_ij`, call a position `(i, j)`
**tight** (or "qualified") when `u_i + v_j = r_ij`, and observe that an assignment using only tight
positions, together with the dual `(u, v)`, satisfies weak duality with equality — hence both are
optimal. The tight positions form a 0/1 qualification matrix, so finding a best assignment among
tight positions is a König problem. When no full assignment exists among the tight positions,
Egerváry's contribution is the rule for adjusting `(u, v)` to expose new tight positions while
keeping `u_i + v_j ≥ r_ij` and *decreasing* the dual objective `Σ u_i + Σ v_j`. Thus the integer
problem reduces to a finite sequence of König (0/1) problems linked by dual updates.

**Augmenting paths (the matching engine).** Finding a maximum matching in a bipartite graph is done
by alternating/augmenting paths. An **alternating path** alternates between non-matching and
matching edges; an **augmenting path** is an alternating path whose two endpoints are both exposed
(unmatched). Taking the symmetric difference of the matching with an augmenting path flips every
edge along it and yields a matching with one more edge. The characterization is: a
matching is maximum **iff** there is no augmenting path with respect to it. So the matching
subroutine is: repeatedly search for an augmenting path (a directed reachability search from an
exposed vertex), augment, and stop when none exists — at which point König's cover is read off from
the vertices reachable by alternating paths.

**The computational pressure.** Available linear-programming machinery treats the assignment
instance as a dense collection of variables and degenerate constraints, while the combinatorial
view exposes a small certificate: a matching and a covering family of lines, or a matching and dual
potentials. The practical pull is to replace a generic tableau calculation by repeated matching and
dual-adjustment steps whose intermediate states can be inspected directly.

## Baselines

**General simplex on the LP.** Treat the assignment problem as the linear program above and run the
simplex method. It is correct and finds an optimum, but the assignment polytope is highly
degenerate (many bases give the same vertex), so simplex cycles through degenerate pivots and is
slow; it also gives no special combinatorial certificate beyond the generic LP one, and for the
sizes of interest in 1953 the largest general LPs the available machines could handle were near the
problem's own size. **Gap:** does not exploit the 0/1 / matching structure; degeneracy makes it
slow and the integrality of the answer is incidental rather than guaranteed by the method.

**Transportation-problem methods (Dantzig).** The assignment problem is the special case of the
transportation problem with all supplies and demands equal to 1; Dantzig's simplex specialization
for transportation (1951) applies. **Gap:** precisely because all supplies/demands are 1, the
assignment case is the *most degenerate* instance of transportation, which is exactly where the
transportation simplex struggles; the method is built for the general flow problem, not for the
combinatorial all-ones case.

**Brute-force permutation search.** Enumerate all `n!` permutations and keep the best. Correct and
trivially certifiable, but `n!` is astronomically large past tiny `n`; useful only to *check* an
assignment solver on small instances, never to solve real ones.

**König's 0/1 algorithm alone.** Solves the qualification (0/1) special case — maximum independent
marks / minimum line cover — constructively and in polynomial time, and it carries its own
optimality certificate (the matching meets the cover). **Gap:** it handles only 0/1 ratings; the
general integer-rating problem is not directly a single 0/1 problem.

## Evaluation settings

The natural test instances are square integer matrices `R = (r_ij)` of small-to-moderate order
(`n` up to a few tens), with small positive integer entries (e.g. 1–3 digit ratings), solved by
hand or on the electronic computers of the early 1950s. The quantities of interest are: the optimal
rating sum, the optimal assignment (permutation) achieving it, and a dual certificate `(u_i, v_j)`
with `Σ u_i + Σ v_j` equal to the rating sum. The yardsticks are correctness (does it find a true
optimum, checkable on small cases against full permutation enumeration), the presence of an
optimality certificate, and running time/operation count as a function of `n`. The desired behavior
is polynomial growth in `n`, with intermediate certificates that can be checked without enumerating
permutations.

## Code framework

Available pieces: a cost/rating matrix, the dual potentials with their feasibility test, a
tight-position (qualification) matrix derived from the potentials, a bipartite maximum-matching
routine via augmenting paths, and a brute-force checker for small instances. The remaining slot is
the assignment loop that combines matching with a dual update.

```python
INF = float("inf")

def cost_at(cost, i, j):
    return cost[i][j]

def tight_positions(cost, u, v):
    """Positions (i,j) with reduced cost c_ij - u_i - v_j == 0, given dual (u,v)."""
    n = len(cost)
    return [[cost[i][j] - u[i] - v[j] == 0 for j in range(n)] for i in range(n)]

def bipartite_max_matching(adj):
    """Maximum matching on a 0/1 adjacency.
    Returns row->col and col->row arrays."""
    # TODO: alternating/augmenting-path search; standard and pre-existing.
    pass

def min_vertex_cover_from_matching(adj, match_col, match_row):
    """Given a maximum matching, return a minimum vertex cover (Koenig)."""
    # TODO: mark exposed rows, alternate; cover = unmarked rows + marked cols.
    pass

def solve_assignment(cost):
    """Fill in the primal-dual assignment loop."""
    # TODO: combine tight-edge matching, cover extraction, and potential update.
    pass

def brute_force(cost):
    """Optimal assignment by full permutation enumeration -- a checker only."""
    import itertools
    n = len(cost)
    best, bestp = INF, None
    for perm in itertools.permutations(range(n)):
        s = sum(cost[i][perm[i]] for i in range(n))
        if s < best:
            best, bestp = s, perm
    return list(bestp), best
```
