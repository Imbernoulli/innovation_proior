## Research question

Given an `n × n` matrix of numbers `r_ij` — read as the rating (or, after a sign flip, the cost)
of putting individual `i` in job `j` — choose exactly one entry in each row and each column so that
the sum of the chosen entries is as large (for ratings) or as small (for costs) as possible. A
choice of one entry per row and column is a **permutation** `j_1, …, j_n` of the columns, and we
want the permutation maximizing `r_{1 j_1} + ⋯ + r_{n j_n}`. This is the **assignment problem**:
assign `n` people to `n` jobs, one each, with the total rating maximal.

The number of permutations is `n!`, which is hopeless to enumerate even for moderate `n` —
`12! ≈ 4.8 × 10^8`. The problem is a linear program: its feasible set is exactly the
doubly-stochastic matrices, the convex hull of the permutation matrices, so in principle the simplex
method solves it. A `10 × 10` instance is a linear program with `100` nonnegative variables and `20`
equality constraints, and it is highly **degenerate** (the assignment problem is the most degenerate
special case of the transportation problem). The setting, then, is how to find — and certify — an
exact optimal assignment for square integer rating or cost matrices of small-to-moderate order. The
deliverable is a single self-contained C++17 program that reads an integer `n` and then the `n × n`
integer matrix from standard input, and writes the optimum value and chosen row/column pairs to
standard output.

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
Hungarian in 1953) extended König's min–max theorem from 0/1 matrices to integer-weighted ones,
phrasing the result in terms of integer row and column numbers `u_i, v_j`. König's 0/1 statement is
thus a special case of this result about integer-weighted matrices.

**Augmenting paths (the matching engine).** Finding a maximum matching in a bipartite graph is done
by alternating/augmenting paths. An **alternating path** alternates between non-matching and
matching edges; an **augmenting path** is an alternating path whose two endpoints are both exposed
(unmatched). Taking the symmetric difference of the matching with an augmenting path flips every
edge along it and yields a matching with one more edge. The characterization is: a
matching is maximum **iff** there is no augmenting path with respect to it. So the matching
subroutine is: repeatedly search for an augmenting path (a directed reachability search from an
exposed vertex), augment, and stop when none exists — at which point König's cover is read off from
the vertices reachable by alternating paths.

## Baselines

**General simplex on the LP.** Treat the assignment problem as the linear program above and run the
simplex method. It is correct and finds an optimum, working through the bases of the degenerate
assignment polytope and yielding the generic LP certificate. For the sizes of interest in 1953 the
largest general LPs the available machines could handle were near the problem's own size.

**Transportation-problem methods (Dantzig).** The assignment problem is the special case of the
transportation problem with all supplies and demands equal to 1; Dantzig's simplex specialization
for transportation (1951) applies directly to it.

**Brute-force permutation search.** Enumerate all `n!` permutations and keep the best. Correct and
trivially certifiable; `n!` is astronomically large past small `n`, so it serves to *check* an
assignment solver on small instances.

**König's 0/1 algorithm.** Solves the qualification (0/1) case — maximum independent marks / minimum
line cover — constructively and in polynomial time, carrying its own optimality certificate (the
matching meets the cover).

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

The program reads from stdin: an integer `n`, followed by `n × n` integer costs in row-major order.
It writes to stdout: the minimum total cost on the first line, then `n` lines of `i j`, meaning
0-based row `i` is matched to 0-based column `j`. A maximization instance can be supplied by
negating the input values and then negating the reported optimum value.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    vector<vector<long long>> cost(n, vector<long long>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cin >> cost[i][j];
        }
    }

    vector<int> column_for_row(n, 0);
    long long total = 0;

    // TODO: Fill in the optimization logic.

    cout << total << '\n';
    for (int i = 0; i < n; ++i) {
        cout << i << ' ' << column_for_row[i] << '\n';
    }
    return 0;
}
```
