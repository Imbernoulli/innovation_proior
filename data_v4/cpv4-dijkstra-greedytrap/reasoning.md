I have a directed graph on `n` stations with `m` edges, each edge carrying a line id `c` and a fare
`w`, and I want the cheapest route from `1` to `n` where every change of line along the way costs a
fixed surcharge `S`. The first boarding is free of surcharge and arriving at `n` is free; if `n` is
unreachable I print `-1`. Two things about the numbers set the terms before any algorithm. First,
`n, m <= 2*10^5` while fares and `S` each run up to `10^9`, and a route can string together on the
order of `m` edges, each paying a fare and possibly a transfer, so the accumulated cost reaches about
`2*10^5 * 10^9 * 2 = 4*10^14` — well past the 32-bit ceiling of `~2.1*10^9`. Every distance and every
accumulator has to be 64-bit; an `int` anywhere is a silent wrong answer on the large tests. Second,
and more consequential for the modelling, the surcharge makes the cost of *extending* a route depend
not on where I am but on the line I arrived on. That is the whole difficulty, and it decides what a
graph node is.

All weights are non-negative, so Dijkstra is the tool; the only real question is the state. The
tempting choice is one label per station, `dist[v]` = cheapest cost to reach `v`, charging `S` when
the boarded line differs from the line `v`'s best route arrived on. It is `O((n+m) log n)` and
textbook. But it rests on the assumption that the cheapest arrival at a station summarizes everything
the future needs, and the future here cares about the arriving line, which the cheapest route has no
reason to optimize. A concrete instance settles it.

Take `S = 5` and three edges: `1 ->(Red,1) 2`, `1 ->(Blue,3) 2`, `2 ->(Blue,1) 3`. Station-only
Dijkstra relaxes both edges into `2`, keeps the cheaper Red arrival at cost `1`, and records `2`'s
arriving line as Red. Then `2 ->(Blue,1) 3` is a Red-to-Blue change: `1 + 1 + 5 = 7`. But the pricier
Blue arrival at `2` (cost `3`) continues Blue-to-Blue for free: `3 + 1 + 0 = 4 < 7`. The cheap Red
arrival blocked the free continuation that only a Blue arrival unlocks. So the station-only label is
wrong, and wrong precisely because it discards the arriving line. The node has to be
`(station, arriving line)`.

So a state is `(u, lc)` where `lc` is the line of the edge I last rode into `u`. I start at `(1, 0)`
with a sentinel `lc = 0`, since real line ids are `>= 1` and the first boarding must not count as a
transfer. The transition from `(u, lc)` along `(u, v, c, w)` is

```
new cost = dist[(u, lc)] + w + ((lc != 0 && c != lc) ? S : 0)   into state (v, c)
```

The surcharge fires only when I have already ridden something (`lc != 0`) and the line changes. Each
edge `(u, v, c, w)` can only ever create the state `(v, c)`, so there are `O(m)` reachable states and
the run is `O((n+m) log m)`. On the expanded graph all weights are non-negative, so Dijkstra is valid
and the first time I pop a state at station `n` its cost is the global minimum over all ways to arrive
at `n` — and arriving is free, so that cost is the answer and I can stop. If the heap drains with no
`n`-state ever popped, `n` is unreachable and the answer stays `-1`.

Now to write it: a per-station map `best[u] : line -> cost`, a min-heap of `(cost, station, line)`,
the `(1, 0)@0` start, and the stop-on-first-`n`-pop rule. Two traps in this particular code are
invited by the specifics of this problem, not by Dijkstra in general.

The first is the sentinel. It is tempting to write the surcharge as just `(c != lc) ? S : 0` — it
reads cleaner — but `lc = 0` at the start is not a real line, and `c != 0` holds for every real
`c >= 1`, so a bare `c != lc` taxes the very first boarding. On the clean counterexample that turns
the true `4` into `9` (every route pays a phantom `+S` on its first edge), and on the metro sample it
turns `8` into `11` by taxing the optimal route's opening `1 ->(Blue,2)` leg. The `lc != 0` guard is
exactly what keeps the first edge free, so the surcharge condition must be `lc != 0 && c != lc`.

The second is stale heap entries. I push a fresh `(nd, v, c)` every time I improve `best[v][c]` and
never delete the superseded ones, so a single state can sit in the heap under two different costs.
When I pop a triple I therefore have to confirm it is still the current best for its state —
`best[u][lc] == d` — and skip otherwise. A stale pop can never win a relaxation (its base cost is
larger, so the update rule rejects everything it would produce), so letting it through cannot corrupt
the answer; but it re-expands the whole adjacency list, and on a graph where a state is improved many
times that decays toward quadratic work and blows the 2-second limit. Comparing the popped cost to the
stored best and skipping on mismatch bounds each state to a single expansion, keeping the total at
`O((n+m) log m)` — comfortably fast at `n = m = 2*10^5`.

To be sure the algorithm and not just my one counterexample is right, I check it against an
independent label-correcting fixpoint over the same `(station, line)` state space — one that makes no
Dijkstra ordering assumption and just relaxes every transition until nothing improves. Over random
small cases (varying line counts, `S` including `0`, self-loops, parallel edges, disconnected targets)
the two agree everywhere, while the discarded station-only Dijkstra disagrees on real graphs — the
trap is genuine, and the augmented state is what removes it.

The corners then fall out of the construction. If `n` is unreachable the heap empties without an
`n`-pop and `answer` keeps its initial `-1`. With `S = 0` the surcharge term is always zero and the
state split, though still present, never affects cost, so the answer collapses to an ordinary shortest
path. On a single-line graph no edge is ever a transfer, and the guards make every edge free — again a
plain shortest path. A self-loop only adds non-negative cost, so the update rule never improves through
it; parallel edges on different lines are distinct states `(v, c)` and are handled by construction. And
with every accumulator in `long long`, the `~4*10^14` worst case fits with room to spare.

The full program — augmented-state Dijkstra with the `lc != 0` surcharge guard and the stale-cost
skip, keyed by a per-station `line -> cost` hash map — is in the answer.
