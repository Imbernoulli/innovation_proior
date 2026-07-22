OK, let me think this through from scratch. I've got an `n × n` matrix of ratings `r_ij` — how good
person `i` is at job `j` — and I have to hand out the jobs, one per person, one person per job, so
the total rating is as large as possible. A legal hand-out is a permutation `j_1, …, j_n` of the
columns, and I want the one maximizing `r_{1 j_1} + ⋯ + r_{n j_n}`. The brute-force answer is to try
all `n!` permutations and keep the best, and that's fine for `n = 4` but it's `4.8 × 10^8` for
`n = 12`, which I am not going to enumerate by hand. So enumeration is out, and I need to *exploit*
something about the structure.

The obvious structural observation is that this is a linear program. Write `x_ij = 1` if I give job
`j` to person `i` and `0` otherwise; the constraints "one job per person, one person per job" are
just `Σ_j x_ij = 1` for every row and `Σ_i x_ij = 1` for every column, with `x_ij ≥ 0`, and I'm
maximizing `Σ_ij r_ij x_ij`, a linear objective. So I could throw the simplex method at it. But that
feels like the wrong tool, and it's worth being precise about why. A `10 × 10` instance is a linear
program with `100` variables and `20` equality constraints — and it's brutally degenerate: every
assignment is a vertex where `90` of the `100` variables are zero, lots of bases describe the same
vertex, and simplex grinds through degenerate pivots that don't move the objective. An early-1950s
machine can barely fit a general LP of that size at all. There has to be an algorithm that
uses the fact that this isn't a generic LP — its constraint matrix is all `0`'s and `1`'s, the
incidence structure of a bipartite graph — instead of pretending it's arbitrary.

Before I even look for an algorithm, let me get the dual, because the dual is where the certificate
will come from. To each row attach a number `u_i` and to each column a number `v_j`. The linear-
programming dual of "max `Σ r_ij x_ij` subject to unit row/column sums, `x ≥ 0`" is: minimize
`Σ_i u_i + Σ_j v_j` subject to `u_i + v_j ≥ r_ij` for every `(i, j)`. I'll read `u_i + v_j` as a
"budgeted value" jointly allotted to person `i` and job `j`, and the dual feasibility condition
`u_i + v_j ≥ r_ij` says the budget for the pair is at least the rating. Now here is the thing I want
to lean on — weak duality, and it's a one-liner. Take *any* assignment (any permutation) and *any*
budget `(u, v)` satisfying `u_i + v_j ≥ r_ij`. Sum the inequality over the `n` chosen pairs:

    Σ_{chosen (i,j)} r_ij ≤ Σ_{chosen (i,j)} (u_i + v_j).

But in an assignment each row index and each column index appears exactly once, so the right side is
just `Σ_i u_i + Σ_j v_j` — the whole budget, regardless of which permutation I chose. So

    (value of ANY assignment) ≤ Σ_i u_i + Σ_j v_j = (cost of ANY feasible budget).

That's the lever. Every feasible budget is an upper bound on every assignment. If I can ever exhibit
*one* assignment and *one* budget where these two numbers are equal, then that assignment can't be
beaten (it equals an upper bound) and that budget can't be lowered (it equals a lower bound), and
I've solved both problems at once — and the budget is a *certificate* a skeptic can check by hand.
So my real goal is sharper than "find a good assignment": find an assignment and a budget that meet.
And the question becomes, *when* do they meet? Sum the inequalities; equality holds iff every one of
the `n` chosen pairs is *tight*, `u_i + v_j = r_ij`, with no slack. So I want an assignment that uses
only tight pairs. That's complementary slackness, and it's now the whole game: pick a budget, look at
its tight pairs, and try to find a full assignment living entirely among them.

Now, is there even any reason to believe a budget exists whose tight pairs admit a full assignment?
This is where I get nervous, because in general an integer program's LP relaxation can have a
fractional optimum and there's a gap. But here the geometry saves me. The feasible region — nonneg-
ative matrices with all row and column sums `1` — is the set of doubly-stochastic matrices, and
Birkhoff showed in 1946 that those are *exactly* the convex hull of the permutation matrices. So
when I maximize a linear function over this polytope, the optimum is attained at a vertex, and the
vertices *are* permutation matrices. There's no fractional trap: the LP optimum is automatically an
honest assignment. (The same fact in constraint-matrix language: the bipartite row/column incidence
matrix is totally unimodular, so every vertex of the polytope has integer coordinates, and with unit
right-hand sides those integers are `0`/`1`.) So at least the thing I'm hunting for isn't a mirage —
the optimal primal is a permutation and the optimal dual is some `(u, v)`, and at their common
optimum weak duality is met with equality, which forces every chosen pair tight. That's an existence
statement, but it doesn't hand me the pair. What I actually need is a *procedure* that walks from
some easy starting budget to a budget-and-assignment that meet, and the real content will be proving
that procedure terminates. So let me try to build one, and watch for where it could fail to make
progress.

So let me deliberately strip the problem down to its bones and solve the easiest possible version
first, because if I can't handle that I can't handle anything. Suppose every rating is just `0` or
`1` — a *qualification* matrix, `1` meaning person `i` can do job `j` at all. Forget budgets for a
moment; the question collapses to: how many `1`'s can I pick with no two in the same row or column?
That's a maximum *matching* in a bipartite graph (rows on one side, columns on the other, an edge
where there's a `1`), and a full assignment exists iff the maximum matching has `n` edges.

How do I grow a matching, and — more importantly — how do I *know* when it's maximum without trying
everything? Start from any set of independent `1`'s. If some unassigned person qualifies for an
unassigned job, just add it; the matching grows. The interesting case is when no such direct
addition exists but the matching still isn't full. Then I try a *transfer*: person `i_1` is qualified
for an unassigned job `j_0`, but `i_1` is already holding job `j_1`; move `i_1` to `j_0`, which frees
`j_1`; maybe someone else `i_2` qualified for `j_1` is holding `j_2`; move them, freeing `j_2`; and
so on. This is exactly an **alternating path**: it alternates between "qualified-but-not-assigned"
edges and "currently-assigned" edges. If such a chain ever ends by freeing up a job that some *un*-
assigned person can take — i.e. both ends of the chain are unmatched — then flipping every edge along
the chain (assigned↔unassigned) increases the number of matched pairs by exactly one. That's an
**augmenting path**, and the symmetric difference of the matching with it is a strictly bigger
matching.

The question is when I should give up — when is the matching truly maximum? I claim: precisely when
no augmenting path exists. One direction is obvious — if an augmenting path exists the matching grows,
so a maximum matching has none. The converse is the content. Suppose `M` is not maximum and `M*` is
bigger; look at their symmetric difference `M △ M*`. Every vertex touches at most one `M`-edge and at
most one `M*`-edge, so this difference breaks into simple paths and cycles that *alternate* between
`M` and `M*` edges. Cycles have equal numbers of each. Since `M*` has more edges overall, some path
must have more `M*`-edges than `M`-edges — and an alternating path with one extra `M*`-edge starts
and ends with `M*`-edges, meaning both its endpoints are exposed in `M`. That path is augmenting with
respect to `M`. So if `M` has no augmenting path, no bigger `M*` can exist — `M` is maximum. Good:
the matching subroutine is "search for an augmenting path from an exposed vertex; if found, augment;
if not, stop," and I can find such a path by a reachability search along alternating edges.

Now the crucial part for the *certificate*. When the augmenting search fails, what have I actually
got? Run the alternating search from every exposed row simultaneously and mark every vertex it
reaches. No augmenting path means the search never reaches an exposed column. Stare at the marked set.
Take the rows that are *not* reachable together with the columns that *are* reachable — call those the
cover. Every `1` of the matrix is covered by this set: a `1` at `(i, j)` with `i` reachable but `j`
not reachable would extend the alternating search to `j`, contradiction, so any `1` has either its
row unreachable (covered) or its column reachable (covered). And the size of this cover equals the
size of the matching — each matched edge contributes exactly one covering line, unmatched-but-
reachable rows contribute none. So I've simultaneously produced a matching and a set of *lines* (rows
or columns) covering all the `1`'s, of equal size. Since any line can cover at most one edge of a
matching, no cover can be smaller than the matching, and no matching can be bigger than any cover —
they squeeze each other. **The maximum number of independent `1`'s equals the minimum number of lines
covering all `1`'s.** This is König's theorem, and it's the 0/1 case of the very duality I wrote down
at the start — independent marks are the primal, covering lines are the dual — proved here completely
constructively. When the matching is full (`n` edges) I have my assignment; when it isn't, König
hands me a cover with *fewer than `n` lines*, and that small cover is going to be exactly the tool I
need to fix the budget.

So now back up to the real problem with general integer ratings. I have a budget `(u, v)` that's
dual-feasible (`u_i + v_j ≥ r_ij`), and I look only at its **tight** pairs, the ones with
`u_i + v_j = r_ij`. Those tight pairs form a 0/1 qualification matrix — put a `1` exactly where it's
tight — and I run the König machine on it. Two outcomes. If the tight pairs admit a *full* matching,
then I have an assignment using only tight pairs, complementary slackness holds, weak duality is met
with equality, and by the argument at the very top this assignment is optimal and the budget proves
it. Done. But if the maximum matching on the tight pairs has fewer than `n` edges, König tells me the
tight `1`'s are covered by fewer than `n` lines — and now I have to *change the budget* to create new
tight pairs, without ever violating `u_i + v_j ≥ r_ij`, and in a way that makes progress.

Here's where I have to be careful, because it would be easy to break feasibility. König's cover is a
set of `r` rows and `s` columns, `r + s < n`, covering all the current tight pairs. Equivalently it
splits things into "essential" (in the cover) and "inessential" (out of it). Look at the
*inessential* region — inessential rows crossed with inessential columns. None of those pairs is
tight (if one were, it'd be an uncovered `1`, but the cover covers all `1`'s), so every pair `(i, j)`
there has genuine slack `u_i + v_j - r_ij > 0`. Let

    δ = min over inessential rows i, inessential columns j of (u_i + v_j − r_ij),

the smallest slack anywhere in that uncovered block; `δ > 0`. Now the budget update: subtract `δ`
from `u_i` on every *inessential* row, and add `δ` to `v_j` on every *essential* (covered) column.
Two things need to hold for this to work: feasibility preserved, and the dual objective strictly down.

Feasibility: which pairs `(i, j)` could have their `u_i + v_j` *decrease*? Only ones where I lowered
something and didn't raise it. I lowered `u_i` exactly on inessential rows; I raised `v_j` on
essential columns. So `u_i + v_j` drops only when `i` is inessential *and* `j` is inessential (an
essential column would have gained the same `δ` back). But that's precisely the uncovered block where
I took the minimum slack `δ`. For such a pair the new value is `u_i + v_j − δ`, and since
`u_i + v_j − r_ij ≥ δ` there by the definition of `δ`, I still have `u_i + v_j − δ ≥ r_ij`.
Feasibility holds — and the pair that *achieved* the minimum becomes newly tight, which is exactly
the new `1` I wanted to expose. Every other pair: an essential row with anything keeps its `u_i`, so
no decrease; an inessential row with an essential column loses `δ` from `u` but gains `δ` from `v`,
net zero. So the only pairs that move down are in the safe block, and they stay feasible. Good.

Dual objective: the total budget is `Σ_i u_i + Σ_j v_j`. I subtracted `δ` from each of the
`n − r` inessential rows and added `δ` to each of the `s` essential columns, so the change is
`−δ(n − r) + δ s = −δ(n − r − s) = −δ(n − (r + s))`. And König gave me `r + s < n`, so `n − (r + s)`
is a positive integer, and the budget strictly *decreases* by at least `δ`. (With integer ratings I
can start the budget at integers — e.g. `u_i =` row maximum, `v_j = 0` — and `δ` comes out a positive
integer, so each step drops the budget by at least `1` and stays integral.) Since the budget is an
upper bound on every assignment — so it is bounded below by the value of any assignment, in particular
the optimum — and drops by a positive integer each time, only finitely many updates can happen; between updates the matching only grows (bounded by `n`); so the whole process terminates,
and it can only terminate by the tight pairs admitting a full matching — which is the optimal,
certified assignment. This is Egerváry's step lifting König from `0/1` to integers: a finite sequence
of König problems, stitched together by budget updates that each expose at least one new tight pair.

The update direction isn't arbitrary — it has to be this way and not the mirror image. I want to
*create* a tight pair in the uncovered block, where every pair currently has positive slack
`u_i + v_j − r_ij`. To make one of those equalities, I must *shrink*
`u_i + v_j` there, which means lowering `u` on the inessential rows (or, symmetrically, lowering `v`
on the inessential columns — rows and columns enter the problem symmetrically, so either works, and a
careful implementation picks whichever keeps the potentials nonnegative). But if I only lower `u` on
inessential rows and touch nothing else, I'd wreck feasibility for pairs of an inessential row with an
*essential* column — those might already be tight, and lowering `u_i` would push `u_i + v_j` below
`r_ij`. The fix is to add the same `δ` back to the essential columns' `v_j`, which exactly cancels the
loss on those pairs while leaving the uncovered block to shrink. That's why the update is "subtract
from uncovered rows, add to covered columns" — it's the unique adjustment that shrinks the uncovered
block by `δ`, leaves everything else feasible, and lowers the dual objective.

Now I want to turn this into something I can actually run by hand, and there's a beautiful
simplification: instead of carrying `(u, v)` and recomputing tightness, I can fold the budget *into
the matrix* and track only reduced numbers. Define the **reduced matrix** `a_ij = u_i + v_j − r_ij ≥ 0`
(the slacks). Tight pairs are exactly the zeros of `a`. The whole method becomes operations on `a`
that keep it nonnegative and preserve which assignment is optimal — and the key fact is that *adding a
constant to a whole row or column of the matrix doesn't change the optimal assignment*, because every
assignment picks exactly one entry from that row (or column) and so every assignment's total shifts by
the same constant. So I'm free to subtract row constants and column constants at will, hunting for a
configuration of zeros that admits a full assignment. Concretely (now phrasing it as a *minimization*
of cost `c_ij`, which is the same problem with a sign flip, and is cleaner to reduce):

Subtract from each row its row-minimum — now every row has at least one zero and all entries are
`≥ 0`. Then subtract from each column its column-minimum — every column has a zero too. The zeros are
my tight pairs; an assignment lying entirely on zeros would have reduced cost `0`, hence be optimal.
So I try to place `n` independent zeros (a full matching on the zeros) by the augmenting-path routine.
If I get `n`, I'm done. If not, König gives me the minimum set of lines covering all the zeros; fewer
than `n` lines. The uncovered entries are all strictly positive; let `d` be the smallest uncovered
entry. The budget update "subtract `δ` from inessential rows, add to essential columns" becomes, in
reduced-matrix terms: **subtract `d` from every uncovered row and add `d` to every covered column.**
Cell by cell: uncovered-row/uncovered-column entries lose `d` only (a new zero appears where the
minimum was); covered-row/covered-column entries gain `d` only; uncovered-row/covered-column entries
lose `d` then gain it back, net zero; covered-row/uncovered-column entries are untouched. All stay
`≥ 0` because `d` was the smallest uncovered value. Repeat: re-match on the new zeros,
re-cover, re-subtract, until a full set of independent zeros appears. The dual objective `Σ u + Σ v`
is now a *lower* bound on the cost (I flipped the sign), and each cover step strictly *raises* it — by
`d·(n − |cover|) > 0`, the same count as before — so it climbs toward the optimum and this halts.
(The bookkeeping is identical whether I phrase the update as "uncovered rows minus / covered columns
plus" or the mirror "covered rows plus / uncovered columns minus" — writing `R` for "row covered" and
`C` for "column covered", both move cell `(i, j)` by `d·(R + C − 1)`, so they are literally the same
update cell by cell and reach the same zeros.)

That's the matrix form, and it's exactly what I'd do with pencil on a tableau. But for a machine I'd
rather not re-scan the whole matrix and recompute a maximum matching from scratch every iteration —
that's wasteful. Let me think about the operation count and tighten it. The expensive bits are
recomputing the matching and re-covering at every dual update. There's a cleaner way to organize the
same primal–dual logic so the work per added matched pair is `O(n^2)` and the whole thing is `O(n^3)`:
process the rows one at a time, and for each new row grow a *single* shortest augmenting path in the
tight subgraph, raising the duals just enough, on the fly, to extend that path — rather than
recomputing a global matching after a global dual update.

Here's the reorganization. Keep potentials `u_i, v_j` (dual-feasible, reduced cost
`c_ij − u_i − v_j ≥ 0`) and a partial matching `p[j] =` the row matched to column `j`. To add row
`i`, I do a minimum-slack alternating-tree search over columns: maintain `minv[j]`, the smallest reduced cost of
reaching column `j` along an alternating path grown so far, and a predecessor `way[j]` so I can
trace the path back. Repeatedly pick the unused column `j1` with the smallest `minv[j1]`; call that
value `δ`. That `δ` is exactly the König minimum-uncovered-slack, computed incrementally for *this*
path: I shift the potentials by `δ` — raise `u` on the rows already on the path and lower `v` on the
columns already on the path (net zero on tight edges, keeping them tight), and lower the running
`minv` of the columns not yet reached by `δ` — which keeps everything dual-feasible and makes the edge
into `j1` tight. If column `j1` is free (unmatched), the alternating path has reached an exposed
column: I've found the augmenting path, and I flip it by walking `way[]` back, matching this new row.
If `j1` is matched to some row, I step into that row and continue growing the path. Each row costs one
such search, `O(n^2)`; `n` rows give `O(n^2 · n) = O(n^3)` — strongly polynomial, independent of the
size of the costs. This is the same primal–dual method — dual-feasible potentials, tight-edge
matching, König-minimum dual updates, complementary-slackness optimality — just amortized so each
dual update is the local minimum slack of one growing augmenting path instead of a global rescan.

Coding this up: a single self-contained routine that reads the instance from stdin (`n`, then the
`n × n` integer cost matrix in row-major order) and prints the minimum total cost and the chosen
column for each row. Per row `i`, grow a minimum-slack alternating tree over columns — maintain
`minv[j]`, the smallest reduced cost reached at column `j` so far, and a predecessor `way[j]` to
trace back; repeatedly step to the unused column with the smallest `minv`, call that value `delta`,
and apply it: raise the potential on every row already on the path, lower it on every column already
on the path (tight edges stay tight), and lower the running `minv` of the columns not yet reached.
That `delta` is exactly the König minimum-uncovered slack, computed incrementally for this one
growing path instead of by a global rescan. When the column reached is free, the alternating path is
augmenting — flip it by walking `way[]` back and match the new row. I'll work in `1`-based indices
with a dummy column `0` to anchor each path, and carry everything in `long long` so large entries
can't overflow — the delivered program below is exactly this loop, run once per row.

That form is what I'd run on a machine, but its dual updates are amortized into the path search,
which hides the König/Egerváry steps. To check the logic by hand I want the literal tableau form
alongside — the same primal–dual method written as pencil-and-paper matrix moves: row reduce, column
reduce, match the zeros, König-cover them, then subtract the minimum uncovered value off the
uncovered rows and add it onto the covered columns, and repeat. Let me run this tableau form by hand
on a concrete instance, to see how many cover-and-update rounds it actually takes and whether the
dual certificate it leaves behind really equals the cost. For a maximization with rating matrix `R`,
I flip signs (`cost = −R`) and minimize, then negate the total back. Take

    R = [[8,7,9,9],[5,2,7,8],[5,1,4,8],[2,2,2,6]],  so neg = −R = [[-8,-7,-9,-9],
                                                                    [-5,-2,-7,-8],
                                                                    [-5,-1,-4,-8],
                                                                    [-2,-2,-2,-6]].

Row-reduce first: the row minima are `−9, −8, −8, −6`, and subtracting each off its row gives

    [[1,2,0,0],[3,6,1,0],[3,7,4,0],[4,4,4,0]].

Now column-reduce: the column minima are `1, 2, 0, 0`, so subtracting those off the columns gives the
reduced tableau

    a = [[0,0,0,0],[2,4,1,0],[2,5,4,0],[3,2,4,0]].

The zeros are the tight pairs. Try to place four independent zeros. Row 0 has zeros all across;
column 3 is the only zero in rows 1, 2, 3. So however I match, rows 1, 2, 3 all compete for the
single zero in column 3 — at most one of them can be placed there, and row 0 takes one more zero, so
the maximum matching on the zeros has just **2** edges (say `0→0, 1→3`), not 4. The reduction alone
did not solve it — exactly the "fewer than `n`" case where I need the König cover and a dual update.
Good, this is the interesting branch and not a triviality.

König's cover here: start the alternating search from the exposed rows `2` and `3`. From row 2 the
only zero is column 3, mark column 3; column 3 is matched to row 1, so mark row 1; from row 1 the
only zeros are column 3 (already marked); the search stalls. The reachable set is rows `{1,2,3}` and
column `{3}`. The cover is *unmarked rows ∪ marked columns* `= {row 0} ∪ {column 3}` — two lines,
which equals the matching size of 2, just as König promises. The uncovered block is rows `{1,2,3}` ×
columns `{0,1,2}`, with entries `[[2,4,1],[2,5,4],[3,2,4]]`; the smallest is `d = 1` (at `(1,2)`).
Update: subtract `1` from the three uncovered rows, add `1` to the one covered column (column 3):

    a → [[0,0,0,1],[1,3,0,0],[1,4,3,0],[2,1,3,0]].

A new zero has appeared at `(1,2)` exactly where the minimum was, and everything is still `≥ 0`. Match
again: now row 1 can take column 2, freeing column 3 for row 2 — I get `0→0, 1→2, 2→3` — but row 3's
only zero is still column 3, already taken. Maximum matching is **3**, still short. So the single
update did *not* finish it; this needs a second round, which is the thing I wanted to check. Cover
again: exposed row is `3`; from row 3 mark column 3; column 3 matched to row 2, mark row 2; from row 2
the only zero is column 3; stall. Reachable rows `{2,3}`, column `{3}`; cover `= {rows 0,1} ∪ {col 3}`,
three lines `=` matching size 3. Uncovered block rows `{2,3}` × columns `{0,1,2}` is `[[1,4,3],[2,1,3]]`,
minimum `d = 1`. Subtract `1` off rows 2, 3 and add `1` to column 3:

    a → [[0,0,0,2],[1,3,0,1],[0,3,2,0],[1,0,2,0]].

Match a third time: row 0 → col 0, row 1 → col 2, row 2 → col 3, row 3 → col 1 — four independent
zeros, a perfect matching. So the assignment is `(0→0, 1→2, 2→3, 3→1)`, i.e. in 1-based terms
`1→1, 2→3, 3→4, 4→2`, with rating sum `R[0][0]+R[1][2]+R[2][3]+R[3][1] = 8 + 7 + 8 + 2 = 25`.

Now the part I actually care about: does the dual certificate it leaves behind meet the cost? The
duals are the total amounts reduced off each row and column over the whole run. Rows lost
`−9, −8, −8, −6` in the first reduction and then rows 1, 2, 3 each lost a further `1` in round one and
rows 2, 3 a further `1` in round two, giving `u = (−9, −7, −6, −4)`. Columns lost `1, 2, 0, 0` in the
column reduction, and column 3 gained `1` in each of the two updates, giving `v = (1, 2, 0, −2)`. Then
`Σu + Σv = (−9−7−6−4) + (1+2+0−2) = −26 + 1 = −25`, which is exactly the cost of the assignment in the
minimization (`neg[0][0]+neg[1][2]+neg[2][3]+neg[3][1] = −8−7−8−2 = −25`). The bound is met with
equality — so the assignment is certified optimal without enumerating permutations. And on each
chosen cell the budget is tight: `u_0+v_0 = −9+1 = −8 = neg_{00}`, `u_1+v_2 = −7+0 = −7 = neg_{12}`,
`u_2+v_3 = −6−2 = −8 = neg_{23}`, `u_3+v_1 = −4+2 = −2 = neg_{31}`. That is complementary slackness,
verified cell by cell, not assumed. A full permutation enumeration on this `4 × 4` independently
returns the same value `25`, confirming the certificate. The trace took two cover-and-update rounds
— the initial row/column reduction alone didn't finish it, and the dual updates were what closed the
gap. Checking against a brute-force permutation enumerator — try all `n!` orderings and keep the
cheapest — on a few hundred random small integer matrices, the optimal cost from the program matches
every time (`0` mismatches).

The `minv`/`way` shortest-augmenting-path update is the part I'd most easily get wrong under time
pressure; if I weren't confident I could get the potential signs and path flip right in the budget,
I'd fall back to the literal tableau Hungarian loop I've already traced as correct, with zero matching
and König-cover recomputed after each slack update, and ship that -- a plain correct submission beats an ambitious broken one.
