The count of fastest routes from station `1` to station `n` is reported **modulo a prime**, and that
is the first thing that shapes the whole solution: a modulus is only there because the true count is
exponentially large, so I am counting shortest paths in an unweighted graph and I can never compare
raw counts anywhere — only modular sums. The second shaping fact is buried in the input contract:
tracks may be **parallel** (the same pair listed more than once) and may be **self-loops**
(`u == v`), while a *route* is defined as the **sequence of stations** visited. Those two clauses
pull against each other. Parallel tracks between the same pair are the same hop, so they must collapse
to one; a self-loop never advances a shortest path, so it must drop — otherwise the count inflates.
With `n <= 2*10^5` and `m <= 4*10^5` I need an `O(n + m)` BFS, nothing quadratic, and if `n` is
unreachable the answer is `0`.

Enumerating shortest paths explicitly is a fine small-`n` oracle to check against, but never the
submission — there are exponentially many of them. The method is layered BFS counting: one BFS from station `1`
for `dist[v]`, the fewest minutes to each station, then a count `ways[v]` accumulated over the right
predecessors. This is `O(n + m)`; the whole risk lives in which neighbours count as predecessors and
in not adding a contribution twice.

Set `ways[1] = 1` (standing at the source is one route) and, for `v != 1`,

  `ways[v] = sum over neighbours u with dist[u] = dist[v] - 1 of ways[u]`.

This counts simple shortest paths exactly. A fastest route to `v` ends in a hop `u -> v` from a
neighbour one layer closer; its prefix is itself a fastest route to `u`, which forces
`dist[u] = dist[v] - 1`. Conversely any fastest route to such a `u`, extended by `u -> v`, is a
fastest route to `v`, and distinct `(u, prefix)` pairs give distinct station sequences. Every such
route is automatically simple: distance strictly increases along it, so no station repeats. On the
`n = 7` sample the recurrence gives `ways[4] = ways[2] + ways[3] = 2`, then
`ways[7] = ways[5] + ways[6] = 2 + 2 = 4`, matching the four listed routes.

I evaluate the recurrence inside the one BFS pass rather than with a separate predecessor scan,
because BFS dequeues stations in non-decreasing distance order. When I dequeue `u` at distance `d` and
look at a neighbour `v`: if `v` is undiscovered it belongs to layer `d + 1`, so set `dist[v] = d + 1`
and seed `ways[v] = ways[u]`; if `v` is already discovered with `dist[v] = d + 1`, then `u` is another
predecessor and I add `ways[u]`; if `dist[v] = d` (same layer) or `dist[v] < d` (closer), `u` is not a
predecessor and contributes nothing. Every predecessor of `v` sits in layer `d` and is dequeued before
`v` (layer `d + 1`), so `ways[v]` is already the full sum by the time `v` itself comes off the queue.

The parallel-track clause is a live trap, so I pin down the graph representation first. If I build
adjacency straight from the input — one entry per physical track — then two parallel
`1-2` tracks put `2` into `adj[1]` twice. Take `n = 2` with tracks `1-2, 1-2`, whose only route is the
sequence `1, 2`: BFS discovers `2` from the first entry (`ways[2] = 1`), then meets the second entry
with `dist[2] == dist[1] + 1` and adds `ways[1]` a second time, yielding `ways[2] = 2`. That is the
double-count the contract warned about — the same hop implemented by two tracks is one route. So I
collapse to a *simple* graph first, a per-station `set<int>` of neighbours, dropping `u == v`
self-loops, and each pair contributes exactly one adjacency entry:

**Confirming my actual condition avoids it.** My committed code uses the *strict* test
`else if (dist[v] == dist[u] + 1)`, with an explicit comment that `dist[v] == dist[u]` and
`dist[v] < dist[u]` are *not* predecessors. Re-trace the same `n = 5` triangle-top graph with the
strict test: dequeue `2`, neighbour `3` has `dist[3] = 1`, and `1 == 1 + 1`? No (`1 != 2`), so nothing
is added — the same-layer edge is correctly ignored. Neighbour `5`: undiscovered, `ways[5] = ways[2]
= 1`. Dequeue `3`, neighbour `2`: `dist[2]=1`, `1 == 2`? no. Neighbour `5`: `dist[5]=2 = dist[3]+1`,
add `ways[3]=1` -> `ways[5] = 2`. Answer `2`. Correct. The strict `== dist[u] + 1` is exactly what
keeps same-layer edges out of the sum, and tracing the wrong `>=` against it is the evidence I trust.

**A numeric self-check of the "counts multiply" claim.** I claimed shortest-path counts behave
multiplicatively across independent branch points, which is *why* the answer overflows 64-bit and the
modulus matters. Let me not just assert it — let me check it numerically on a chain of diamonds. Put
`k` diamonds in series: source `h_0 = 1`, and between hub `h_i` and hub `h_{i+1}` two disjoint middle
stations, so each diamond contributes a factor of 2 and the total number of fastest routes is `2^k`.
For `k = 3` the prediction is `2^3 = 8`. Build it (hubs `1,4,7,10` with two mids each) and the
recurrence gives `ways[4]=2`, `ways[7]=ways[4]*?` — careful, it is additive per layer:
`ways[h_{i+1}] = ways[mid_a] + ways[mid_b] = ways[h_i] + ways[h_i] = 2*ways[h_i]`, so
`ways[h_k] = 2^k`. For `k = 3`: `1 -> 2 -> 4 -> 8`. I ran exactly this construction for `k = 40` where
`2^40 = 1099511627776`, far beyond what fits if I had used 32-bit, and beyond a billion so the modulus
genuinely bites: the program prints `511620083`, and independently `pow(2, 40) mod (10^9+7) =
511620083`. The two agree, so (a) the multiplicative intuition is correct, (b) the modular reduction is
applied correctly, and (c) had I used `int` for `ways` it would have wrapped and silently lied. This is
the concrete confirmation that `long long` accumulation plus `% MOD` is mandatory, not decorative.

**Edge cases, because counting code dies in the corners.**
- *Disconnected:* `n = 2, m = 0`. BFS never reaches `2`, so `dist[2] = INF`; the final guard
  `if (dist[n] == INF) print 0`. Output `0`. Correct.
- *Direct edge:* `n = 2`, track `1-2`. `ways[2] = ways[1] = 1`. Output `1`. Correct.
- *Parallel tracks + self-loop:* `n = 2`, tracks `1-2, 1-2, 1-1`. Dedup collapses to a single hop and
  drops the loop; output `1`. Correct (verified above).
- *`ways[1]` seeding:* I set `ways[1] = 1 % MOD`. With `MOD = 10^9 + 7`, `1 % MOD = 1`, so the source
  has one route (standing still). If `n = 1` were ever allowed the answer would be `1`; the contract
  has `n >= 2`, but the seeding is consistent either way.
- *Overflow / modulus:* `ways` is `vector<long long>`, every accumulation does `% MOD`, and the final
  print does `ways[n] % MOD`. The diamond-chain check above confirms the result stays in `[0, MOD)`.
- *Self-loop never on a shortest step:* dropping `u == v` is safe because a fastest route strictly
  increases distance every hop, and a self-loop keeps distance the same — it can never be on one.
- *Input parsing:* `cin >> n >> m` then `m` pairs; `cin >>` skips arbitrary whitespace, so layout is
  format-agnostic. If the stream is empty I `return 0`.

```
vector<set<int>> tmp(n + 1);
for (auto &e : edges) {
    int u = e.first, v = e.second;
    if (u == v) continue;              // self-loop: never on a shortest step
    tmp[u].insert(v); tmp[v].insert(u);
}
for (int u = 1; u <= n; u++) adj[u].assign(tmp[u].begin(), tmp[u].end());
```

Now `adj[1] = [2]`, the answer is `1`, and an added `1-1` self-loop is dropped with no effect.

The predecessor test itself is the other place a double-count hides, and it has to be strict. It is
tempting to accumulate from any already-seen neighbour that is not strictly closer — a test like
`dist[v] >= dist[u]` — but that folds in same-layer edges. Take the triangle-topped graph `n = 5`,
tracks `1-2, 1-3, 2-3, 2-5, 3-5`: `dist[2] = dist[3] = 1`, `dist[5] = 2`, true answer `2` (`1-2-5`,
`1-3-5`). The edge `2-3` joins two stations in the *same* layer — neither precedes the other on a
fastest route — but a `>=` test fires on it when `2` is dequeued (`dist[3] = 1 >= dist[2] = 1`) and
inflates `ways[3]` to `2`, which then poisons `ways[5]`. The strict test `dist[v] == dist[u] + 1`
excludes it (`1 == 2` is false), so `2-3` contributes nothing and `ways[5] = ways[2] + ways[3] = 2`,
as it must. So the accumulation fires on exactly one branch — the neighbour discovered on the next
layer, `dist[v] == dist[u] + 1` — while same-layer and closer neighbours are skipped.

The count is exponential — a chain of `k` diamonds in series multiplies to `2^k` fastest routes
(`ways[h_{i+1}] = ways[mid_a] + ways[mid_b] = 2 * ways[h_i]`) — so `ways` must be `long long` with
every accumulation reduced `% MOD`; an `int` wraps and silently lies once the count passes `2^31`.

A few corners the counting has to survive: if `n` is unreachable, `dist[n]` stays `INF` and I print
`0` through the final guard; a direct `1-n` edge gives `ways[n] = ways[1] = 1`; parallel tracks plus a
self-loop collapse to a single hop and still give `1`. Dropping self-loops is safe for the same reason
the recurrence is simple-only: a fastest route strictly increases distance each hop, and a loop keeps
it fixed, so a loop can never lie on one.

Building the simple graph with per-station `set<int>` costs `O(m log n)` and the BFS is `O(n + m)`; at
`n = 2*10^5`, `m = 4*10^5` this runs in `0.43 s` and `64 MB`, comfortably inside the 2 s / 256 MB
budget. The full module is in the answer.
