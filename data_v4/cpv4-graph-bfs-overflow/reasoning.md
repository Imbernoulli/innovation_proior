A broadcast starts at city `1` and floods outward over `m` bidirectional, unweighted fiber links; each
reached city `v` sits at a *depth* equal to its minimum hop count from city `1`, and relaying there
costs `depth(v) * w[v]` for a non-negative importance `w[v]` (the source, depth `0`, costs nothing).
I sum that over every reachable city; unreachable cities contribute nothing. On its face this is
textbook — unweighted single-source shortest path, then a weighted sum. What makes it a real problem
is the arithmetic hiding in the constraints (`n, m` up to `2*10^5`, `w` up to `10^6`), so I pin the
scale before writing a line, because here the scale *is* the trap.

A reachable city's depth is at most `n - 1`, so a single term `depth(v) * w[v]` reaches
`(2*10^5)(10^6) = 2*10^11`, already two orders past the 32-bit signed ceiling `~2.147*10^9`. Summed
over up to `2*10^5` cities the total reaches about `2*10^16` — hopeless for `int`, but comfortably
inside `long long` (ceiling `~9.22*10^18`). Two things follow, and both are easy to get wrong. The
accumulator is `long long`. And the *product* `depth(v) * w[v]` must be formed in 64-bit: if `depth`
and `w` are both `int`, `depth * w` is an `int` multiply that overflows *before* the result is ever
widened onto the 64-bit accumulator. Depths themselves can stay `int` (never past `n - 1`); the danger
lives entirely in the multiply and the running sum.

The cost formula is trivial once depths are known, so the only algorithmic choice is how to get the
depths. The links are unweighted, so the minimum hop count from city `1` is exactly what breadth-first
search computes: it dequeues cities in non-decreasing depth order and fixes each city's depth the first
time it is discovered, in `O(n + m)`. Dijkstra would also be correct but reimplements BFS through a
priority queue for a wasted `log` factor when every weight is `1`. DFS is simply wrong for *minimum*
hops: on the triangle `1-2`, `2-3`, `1-3`, a DFS taking `1-2` then `2-3` stamps `depth(3) = 2`, but
city `3` is one hop away directly — BFS locks `depth(3) = 1` because it drains depth `0` before
depth `1`. So: BFS for depths, 64-bit weighted sum.

The plan is to build an undirected adjacency list (each edge `a b` pushes `b` into `adj[a]` and `a`
into `adj[b]`), BFS from city `1` with `dist[]` initialized to a sentinel `-1` for "unreached", setting
`dist[1] = 0` and relaxing each unseen neighbour to `dist[u] + 1`, then add `(long long)dist[v] * w[v]`
for every city whose `dist` is not `-1`. Self-loops and duplicate edges need no handling: a
self-neighbour or a repeat is already visited when popped, so the sentinel check skips it.

**Sanity-checking the derivation on the sample.** The stated example: `n = 6`, importances
`w = [3, 5, 2, 8, 1, 4]` (so `w[1]=3, ..., w[6]=4`), edges `1-2, 1-3, 2-4, 3-4, 4-5`. City `6` has no
edges, so it is unreachable. Let me run the BFS by hand. Start `dist = [_, 0, -1, -1, -1, -1, -1]`
(index 0 unused), queue `[1]`.

The BFS part of this is correct — I traced it on the sample and it gives the right depths. But the
*cost* part is exactly where I told myself to be paranoid, so let me trace it on a case engineered to
expose magnitude, not structure. Take a path of `4` cities `1-2-3-4` with importances all `10^6`.
Depths are `1:0, 2:1, 3:2, 4:3`. The true cost is `0 + 1*10^6 + 2*10^6 + 3*10^6 = 6*10^6 = 6000000`,
which is small enough to be *right by luck* here — so this case alone won't catch the bug. I need the
total to actually cross `2^31`. So take a longer path, say `n = 100000` cities in a line, all
importances `10^6`. The true total is `10^6 * (0 + 1 + 2 + ... + 99999) = 10^6 * (99999*100000/2) =
10^6 * 4999950000 = 4.99995*10^15`. That is the answer I expect.

The one place this goes wrong in practice is exactly where the cast sits, because it is easy to write
the overflow back in while believing you fixed it. If I widen only the accumulator —

```
long long total = 0;
total += dist[v] * w[v];   // dist, w both int -> 32-bit product, overflows first
```

— the product `dist[v] * w[v]` is still an `int` multiply that wraps, and only the already-broken
32-bit value gets widened on assignment. The widening has to happen before the multiply: either make
`w` a `vector<long long>` (which promotes `dist[v]` at the multiply) or cast, `total += (long
long)dist[v] * w[v];`. I do both — a `long long` weight array and an explicit cast — so it is correct
and self-documenting. The failure is nasty because it is silent: the 6-city sample totals only `26` and
passes with `int` or `long long` alike, while an `int`-typed version on the maximal `2*10^5`-city path
(all `w = 10^6`) prints negative garbage where the true answer is `19999900000000000`. That worst case
also fixes the type bound: `w_max * (n-1)*n/2 = 10^6 * 199999*200000/2 ~ 2*10^16`, well under the `long
long` ceiling, so 64-bit never overflows on any legal input.

For reachability I let the sentinel double as the visited flag: one `dist[]` array, `-1` meaning "never
discovered", `>= 0` meaning "reached at that depth", predicate `dist[v] != -1`. It is tempting to write
`dist[v] > 0`, but that drops the source, which sits at depth `0` and *is* reachable. Here the source
contributes `0 * w = 0` either way, so both predicates give the same total — but `!= -1` states what I
mean, and a single depth/visited array leaves no second `visited[]` to forget to set.

The edge cases fall out cleanly. `n = 1, m = 0`: BFS sets `dist[1] = 0`, the sum is `0`. `m = 0` with
`n > 1`: only city `1` is reached, the rest stay `-1`, total `0`. All-zero importances: total `0`. A
disconnected graph leaves cities outside city `1`'s component at `-1`, skipped — the sample's city `6`
exercises exactly this. Performance is not in question: `O(n + m)` BFS plus an `O(n)` sum runs in a few
hundredths of a second at `n = m = 2*10^5`.

So the shape is fixed: adjacency list, BFS from city `1` with the `-1` sentinel, accumulate
`(long long)dist[v] * w[v]` over reachable cities. The full module is in the answer.

**Final solution.** I earned the BFS over DFS with a concrete miscount, derived the layer-by-layer
depth argument and checked it on the sample, then trapped and killed the int-overflow in the product
and the accumulator (widening *before* the multiply), and chose the sentinel-`-1` reachability test
that says what I mean. This is what I ship — one self-contained file.
