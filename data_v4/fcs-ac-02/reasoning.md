I have one integer `k` (up to `10^8`) and must emit a simple graph with exactly `k` triangles on at
most `1000` vertices — an exact target, no optimization, and a solution is promised to exist. The
awkward part is that the triangle count is a lumpy function of the edge set: adding a single edge to a
dense region can create many triangles at once, so I cannot tune the count edge by edge. A brute
enumeration over all graphs on 4 vertices makes the lumpiness concrete — the achievable counts are
`{0, 1, 2, 4}`, with a hole at `3`: from `K4` (4 triangles) the only way down is to delete an edge,
and deleting one edge of `K4` destroys two triangles at once (the two other vertices are common
neighbors of its endpoints). So near dense graphs the count moves in jumps, and any construction has
to reach `k` with moves whose triangle delta I can predict exactly, not by nudging one edge at a time.

Two binomial facts give me exactly-predictable moves. A clique on `c` vertices holds `C(c,3)`
triangles. And when I add a brand-new vertex and join it to a set `S` of existing vertices, the number
of new triangles is exactly the number of edges *inside* `S` — each such edge closes into a triangle
with the new vertex. That delta collapses to a clean binomial `C(c,2)` only when `S` is a clique of
size `c`; join to an arbitrary set and the delta is some unpredictable internal edge count. So the
whole construction has to keep attaching new vertices to *cliques*.

On scale: `k` and every binomial in play — `C(1000,2) = 499500`, and the largest clique level
`C(m,3) ≈ 10^8` — sit comfortably inside 32 bits, let alone 64. I still carry the residual and the
`C2` table in `long long`, so overflow is simply never a question I have to revisit downstream.

**The construction.** Add vertices `0, 1, 2, ...` one at a time; attach vertex `i` to the *first*
`c_i` vertices `{0, ..., c_i - 1}`, choosing `c_i` as the largest value with `C(c_i, 2) <= r` (the
residual I still owe), capped at `i`. Then subtract `C(c_i, 2)` from `r`. Attaching to the *first*
`c_i` rather than to some arbitrary `c_i`-subset is the whole trick, because it keeps me landing on a
clique: the first `p` vertices always form a clique. While `r` is large the greedy takes `c_i = i`, so
each new vertex joins all its predecessors and the prefix clique grows by one — `{0}`, `{0,1}`,
`{0,1,2}`, and so on. The first time `c_i < i`, the residual has dropped below `C(i,2)` and vertex `i`
attaches to a *prefix* `{0..c_i-1}` of that clique — and a prefix of a clique is itself a clique. From
then on every attachment is to a prefix of the same established clique, so it contributes exactly
`C(c_i, 2)`, and the running total lands on `k` to the unit.

Seen from a distance this is nothing more than "build the largest clique `K_m` with `C(m,3) <= k`,
then pay the residual `k - C(m,3)` with a few extra vertices" — the clique-building phase is precisely
the opening run of steps where `c_i = i`, which continues exactly while `C(i+1,3) <= k`, i.e. up to
`m`. Folding both into one loop is what makes the prefix-clique invariant, and hence the exactness,
obvious; there is no separate residual phase to keep in sync.

A trace on the contract's own example, `k = 2`: vertices `0,1,2` build a triangle (`c` runs `0,1,2`,
adding `0,0,1`), leaving `r = 1`; vertex `3` attaches to `{0,1}` (`C(2,2) = 1`), `r = 0`. Four
vertices, exactly two triangles — matching the sample.

**Vertex budget.** Correctness is not enough; I also need `n <= 1000` for every `k <= 10^8`, or the
graph is illegal even when the count is right. The dangerous inputs are the ones just below a clique
boundary, where the count jumps in big steps and the residual is paid in a tail of touch-ups. The
largest `m` with `C(m,3) <= 10^8` is `m = 844` (`C(844,3) = 99 846 044`); after building `K_m` the
residual is below `C(m,2)`, so it clears in a short tail. Simulating the vertex usage across the range
— including dense clusters just below every `C(c,3)` and `C(c,2)` boundary — the greedy never exceeds
`849` vertices, comfortably inside `1000`. So a solution always fits and the program never has to fail
on in-range input; I still keep a `-1` tripwire if the counter ever passes `1000`, which the
simulation says never fires.

**Implementation.** Precompute `C2[c] = c(c-1)/2` up to `1000`; each step binary-searches for the
largest `c` with `C2[c] <= r`, capped at the current vertex index `i`, joins the new vertex to
`{0..c-1}`, subtracts `C2[c]` from `r`, and repeats until `r` reaches `0`.

Two things about the output contract bite here, both invisible to the count-side logic. First,
`k = 0`: the loop never runs, so `n` stays `0` and I would print `0 0` — but the contract requires
`1 <= n`, and a zero-vertex graph is malformed even though it "has" zero triangles. I bump `n` to `1`
and emit a lone isolated vertex. Second, indexing: my vertices are internally `0`-based, but the
contract numbers them `1..n`. Emitting `e.first` and `e.second` directly would print a literal `0`
endpoint — out of range `[1, n]` and rejected outright. So I print `e.first + 1` and `e.second + 1`.
With both fixed, `k = 0` prints `1 0`, `k = 1` prints a single `K3` as `1 2 / 1 3 / 2 3`, and `k = 2`
reproduces the four-vertex sample above.

**Verification.** An independent counter — not reusing my decomposition — parses `n, m` and the
edges, enforces `n <= 1000`, simplicity, and in-range endpoints, then counts triangles by
common-neighbor bitmasks. Across a few hundred `k` mixing tiny values, the full `10^8` range, and
adversarial clusters just below `C(c,3)` and `C(c,2)` boundaries, every emitted graph has exactly `k`
triangles; the top case `k = 10^8` uses `848` vertices and `356329` edges. The final program is the
greedy with the two output fixes and the defensive `-1` guard; the full module is in the answer.
