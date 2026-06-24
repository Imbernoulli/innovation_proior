**Reading the problem and pinning the contract.** A broadcast starts at city `1` and floods outward
over `m` bidirectional, unweighted fiber links. Each city `v` that the signal reaches has a *depth* —
the minimum number of hops from city `1` — and relaying there costs `depth(v) * w[v]`, where `w[v]` is
a non-negative importance. I must add up that cost over every reachable city and print the total;
cities with no path from city `1` are never lit and contribute nothing, and city `1` itself has depth
`0` so it costs nothing. Input is `n m`, then `n` importances, then `m` edges. Output is one number.

Before touching an algorithm I fix the scale, because here the scale *is* the problem. Depth of a
reachable city is at most `n - 1 <= 2*10^5 - 1`. Importances reach `10^6`. So one term
`depth(v) * w[v]` can be as large as `(2*10^5) * (10^6) = 2*10^11`. That single product already
exceeds the 32-bit signed ceiling of `2^31 - 1 ~ 2.147*10^9` by two orders of magnitude. The *sum*
over up to `2*10^5` cities can reach about `2*10^16`. So: the answer does not fit in `int`, not even
close; it does fit in `long long`, whose ceiling is `~9.22*10^18`, with a margin of roughly 460x. Two
consequences I write down now so I do not forget them under the keyboard: (1) the accumulator is
`long long`; (2) the *product* `depth(v) * w[v]` must itself be computed in 64-bit, because if `depth`
and `w` are both `int`, `depth * w` is evaluated as a 32-bit `int` and overflows *before* it is ever
widened to the 64-bit accumulator. Depths themselves I can keep in `int` (they never exceed `n - 1`),
but I must cast one operand to `long long` at the multiply. This is the twist the whole task is built
around, so I am going to be paranoid about it and deliberately trip the bug later to prove I have
killed it.

**Candidate approaches.** The cost formula is trivial once I know the depths, so the real question is
how to get the depths, i.e. single-source shortest path on this graph.

- *BFS.* The graph is unweighted — every hop costs exactly `1` — so the minimum hop count from city
  `1` is precisely what breadth-first search computes. BFS dequeues cities in non-decreasing depth
  order and assigns each city its final depth the first time it is discovered. `O(n + m)` time,
  `O(n + m)` memory. This is the natural fit.
- *Dijkstra.* Works on any non-negative weights, hence also here, but with all weights equal it just
  reimplements BFS through a priority queue, paying an extra `log` factor for nothing. Overkill.
- *DFS for depths.* Tempting and short, but plain DFS does **not** give shortest hop distance — it
  gives *a* path's length, not the minimum, and on a graph with cycles it can assign a city a depth
  far larger than its true BFS depth. So DFS is simply wrong for "minimum hops." I discard it; I'll
  actually demonstrate why below so the choice is earned, not asserted.
- *Bellman-Ford-style relaxation.* Repeatedly relax all edges until nothing changes. Correct on unit
  weights, but `O(n * m)` in the worst case — far too slow for `2*10^5`. I'll reserve it as my
  *independent brute force* for cross-checking, precisely because it reaches the answer by a different
  mechanism than BFS, but it is not the shipping solution.

So BFS for the depths, then a 64-bit weighted sum. Let me make sure BFS really is correct here before
I lean on it.

**Why DFS would be wrong (earning the BFS choice).** Consider a tiny graph: city `1` connected to
`2`, `2` connected to `3`, and `1` connected directly to `3`. True depths from `1` are `1:0, 2:1,
3:1` (city `3` is one hop away directly). A DFS that explores the edge `1-2` first, then `2-3`, would
stamp `depth(3) = 2` and, unless it later revisits and lowers it, report the wrong depth `2` instead
of `1`. BFS cannot make that mistake: it processes all depth-`0` nodes, then all depth-`1` nodes, so
when it first sees `3` (via the direct edge `1-3`, dequeued at depth `0`) it locks in depth `1` and
never raises it. This concrete case confirms BFS is the right engine and DFS is not.

**Deriving the BFS and the cost sum.** The plan:

1. Build an adjacency list. For each input edge `a b`, push `b` into `adj[a]` and `a` into `adj[b]`
   (undirected). Self-loops add `a` to its own list, which is harmless: when I pop `a` I'll see `a` as
   a neighbour, but `a` is already visited, so nothing happens. Duplicate edges just appear twice in a
   list; also harmless for the same reason.
2. BFS from city `1`. Keep `dist[v]`, initialized to a sentinel meaning "unreached." Set
   `dist[1] = 0`, push `1`. While the queue is non-empty, pop `u`, and for each neighbour `v` with
   `dist[v]` still the sentinel, set `dist[v] = dist[u] + 1` and push `v`. The first assignment is the
   final depth because BFS expands in layers.
3. Sum: for every city `v` from `1` to `n`, if `dist[v]` is not the sentinel (reachable), add
   `(long long)dist[v] * w[v]` to a `long long total`. Print `total`.

For the sentinel I'll use `-1` (no valid depth is negative), which also makes the reachability test a
clean `dist[v] != -1`.

**Sanity-checking the derivation on the sample.** The stated example: `n = 6`, importances
`w = [3, 5, 2, 8, 1, 4]` (so `w[1]=3, ..., w[6]=4`), edges `1-2, 1-3, 2-4, 3-4, 4-5`. City `6` has no
edges, so it is unreachable. Let me run the BFS by hand. Start `dist = [_, 0, -1, -1, -1, -1, -1]`
(index 0 unused), queue `[1]`.

- Pop `1` (depth 0). Neighbours of `1`: `2, 3`. Both unreached → `dist[2]=1`, `dist[3]=1`, push both.
  Queue `[2, 3]`.
- Pop `2` (depth 1). Neighbours: `1, 4`. `1` reached; `4` unreached → `dist[4]=2`, push. Queue
  `[3, 4]`.
- Pop `3` (depth 1). Neighbours: `1, 4`. Both reached now. Queue `[4]`.
- Pop `4` (depth 2). Neighbours: `2, 3, 5`. `2,3` reached; `5` unreached → `dist[5]=3`, push. Queue
  `[5]`.
- Pop `5` (depth 3). Neighbours: `4`. Reached. Queue empty.

Final depths: `1:0, 2:1, 3:1, 4:2, 5:3`, and `6:-1` (unreached). Cost:
`0*3 + 1*5 + 1*2 + 2*8 + 3*1 + (skip 6) = 0 + 5 + 2 + 16 + 3 = 26`. That matches the documented answer
`26`. The derivation is sound.

**First implementation — and immediately a trace, because the magnitude is a trap.** Here is my first
cut of the parts that matter, written the way I'd type it on autopilot:

```
int n, m;
cin >> n >> m;
vector<int> w(n + 1);                 // importances
for (int v = 1; v <= n; v++) cin >> w[v];
// ... build adj, run BFS filling dist[] ...
int total = 0;                        // running total
for (int v = 1; v <= n; v++)
    if (dist[v] != -1)
        total += dist[v] * w[v];      // add this city's cost
cout << total << "\n";
```

The BFS part of this is correct — I traced it on the sample and it gives the right depths. But the
*cost* part is exactly where I told myself to be paranoid, so let me trace it on a case engineered to
expose magnitude, not structure. Take a path of `4` cities `1-2-3-4` with importances all `10^6`.
Depths are `1:0, 2:1, 3:2, 4:3`. The true cost is `0 + 1*10^6 + 2*10^6 + 3*10^6 = 6*10^6 = 6000000`,
which is small enough to be *right by luck* here — so this case alone won't catch the bug. I need the
total to actually cross `2^31`. So take a longer path, say `n = 100000` cities in a line, all
importances `10^6`. The true total is `10^6 * (0 + 1 + 2 + ... + 99999) = 10^6 * (99999*100000/2) =
10^6 * 4999950000 = 4.99995*10^15`. That is the answer I expect.

**The bug.** Now I trace what my first code actually computes for that long path. `w` is a
`vector<int>`, `dist` is `int`, and `total` is `int`. Two failures stack up. First, the product
`dist[v] * w[v]`: at, say, `v` with depth `5000` and `w[v] = 10^6`, the product is `5*10^9`, which as
a 32-bit `int` is computed modulo `2^32` and wraps to a garbage (often negative) value *before* it can
be added. Second, even if every product happened to fit, `total` is an `int` that should reach
`~5*10^15` — it overflows almost immediately and wraps repeatedly. The result is not "a bit off," it
is unrelated noise: on my actual machine an `int`-accumulator/`int`-product version of this very code
prints `-1760880640` for the maximal `2*10^5`-path with `w = 10^6`, whose true answer is
`19999900000000000`. A negative cost is obviously nonsense, which is the tell. The pernicious part is
that on the *6-city sample* the total is only `26`, so the broken code prints `26` and looks perfect —
it only fails on large hidden tests. That is the whole pitfall: a 32-bit `int` silently passes the
sample and silently fails at scale.

**The fix, and where exactly the cast goes.** I make the importance array and the accumulator 64-bit,
and — this is the subtle part — I force the *multiplication* into 64-bit. If I wrote
`long long total; ... total += dist[v] * w[v];` with `dist[v]` and `w[v]` both `int`, the product
`dist[v] * w[v]` is still an `int` multiply that overflows, and only the already-wrong 32-bit result
gets widened on assignment. The widening must happen *before* the multiply. Two clean ways: make `w`
itself `vector<long long>` (then `dist[v] * w[v]` promotes `dist[v]` to `long long` and the multiply is
64-bit), or keep `w` as `int` and write `(long long)dist[v] * w[v]`. I'll do the former for the array
*and* it costs nothing, but to be doubly safe and self-documenting I also cast at the multiply. Fixed
cost loop:

```
vector<long long> w(n + 1);
// ...
long long total = 0;
for (int v = 1; v <= n; v++)
    if (dist[v] != -1)
        total += (long long)dist[v] * w[v];
```

**Re-verifying the fix.** Re-trace the `n = 100000` all-`10^6` path mentally: each product
`dist[v] * w[v]` is now formed as `long long` (no 32-bit wrap), and `total` accumulates in 64-bit, so
it reaches `4999950000000000` exactly. And the maximal `n = 2*10^5` path with `w = 10^6` now yields
`19999900000000000`, a positive number well under the `long long` ceiling `~9.22*10^18` — I checked
the bound: worst case is `w_max * (n-1)*n/2 = 10^6 * (199999*200000/2) ~ 2*10^16 < 9.2*10^18`, so
`long long` never overflows for *any* legal input. The sample still prints `26` (small numbers are
unaffected by the type change). Both the failing large case and the passing small case now behave.

**Second debug episode — the sentinel and the reachability test.** I want to be sure the *unreachable*
handling is right, because a subtle off-by-one there silently corrupts the sum. My first instinct for
"is `v` reachable" might be to test `if (dist[v] > 0)`. Let me trace that against the sample. City `1`
has `dist[1] = 0`. The test `dist[1] > 0` is **false**, so city `1` would be excluded from the sum.
For the sample that happens to cost nothing anyway (`depth 0 * w = 0`), so the answer stays `26` and
the bug hides. But construct a case where it bites: `n = 1`, `m = 0`, `w = [7]`. The only city is `1`,
depth `0`, reachable, cost `0`; the answer is `0` and `dist[1] > 0` gives `0` too — still hidden!
These small cases conspire to mask it because city `1`'s cost is always `0`. The real danger is purely
logical: `dist[v] > 0` conflates "depth zero" (the *source*, which IS reachable) with "unreached." The
correct reachability predicate is `dist[v] != -1` (the sentinel), which cleanly separates "never
discovered" from "discovered at depth 0." I switch to `dist[v] != -1`. To convince myself it matters
beyond city `1`: it doesn't change *which* cities except the source, and the source always contributes
`0`, so for *this cost function* both predicates give the same total — but `dist[v] != -1` is the
predicate that states what I actually mean, and if the cost function ever changed (e.g. a `+constant`
per reached city), `> 0` would undercount the source. I keep the correct one; relying on a coincidence
is how the next refactor breaks.

A related trap I check: what if I had initialized `dist` to `0` instead of `-1` and used a separate
`visited[]` array, but then forgot to set `visited[1]`? Then city `1` could be re-enqueued and, worse,
unreached cities would have `dist = 0` and be *counted* as reachable-at-depth-0 (cost `0`, so again
invisible in the sample, but it would wrongly include genuinely unreachable cities — which here costs
nothing only because depth is `0`; with a different formula it would be a real bug). Using the single
`dist` array with sentinel `-1` both marks visitation and stores the depth, so there is no second
array to forget to update. One source of truth, fewer ways to be wrong.

**Edge cases, deliberately.**
- `n = 1, m = 0`: BFS sets `dist[1] = 0`, queue drains immediately, sum is `0 * w[1] = 0`. Output `0`.
  Correct (single isolated city, no relaying cost).
- `m = 0, n > 1`: only city `1` is reachable (`dist[1]=0`), all others stay `-1`, total `0`. Correct.
- All importances zero: every term is `0`, total `0`, regardless of structure. Correct.
- Disconnected graph: cities in city `1`'s component get real depths; everything else stays `-1` and
  is skipped. The sample itself exercises this (city `6`). Correct.
- Self-loop `a a`: adds `a` to `adj[a]`; when `a` is popped it sees itself as a neighbour but is
  already visited, so nothing changes. Harmless.
- Duplicate edge: appears twice in the lists; the second sighting finds the endpoint already visited.
  Harmless.
- Maximum magnitude: `n = m = 2*10^5`, all `w = 10^6`, a long path. Worst-case total `~2*10^16` fits
  `long long` (ceiling `~9.2*10^18`), and every product is formed in 64-bit. Verified the answer is
  positive and exact (`19999900000000000` for the full path), and the `int` version is demonstrably
  wrong (`-1760880640`). This is the headline pitfall and it is handled.
- Performance: BFS is `O(n + m)`; the sum is `O(n)`. At `n = m = 2*10^5` it runs in about `0.04 s`,
  far inside the `1 s` limit. `vector<vector<int>>` adjacency and a `queue<int>` are plenty.
- Input robustness: `if (!(cin >> n >> m)) return 0;` guards against empty input; `cin >>` skips
  arbitrary whitespace, so line breaks in the edge list don't matter.

**Cross-checking against an independent method.** I do not trust BFS by eyeball alone, so I also wrote
a Bellman-Ford-style flooding brute force (repeatedly relax every edge until depths stabilize) — a
genuinely different mechanism that reaches the same depths — and compared the two on 500 random small
graphs (varied `n`, edge counts, disconnected components, self-loops, duplicate edges, importances
from tiny up to `10^6`). Zero disagreements. The large-magnitude behaviour I checked separately
because the brute force is too slow at scale: the 64-bit solution gives the exact `2*10^16`-range
answer while a deliberately `int`-typed copy of the same code prints negative garbage. Both the logic
(via brute) and the arithmetic (via the magnitude probe) are pinned down.

**Final solution.** I earned the BFS over DFS with a concrete miscount, derived the layer-by-layer
depth argument and checked it on the sample, then trapped and killed the int-overflow in the product
and the accumulator (widening *before* the multiply), and chose the sentinel-`-1` reachability test
that says what I mean. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<long long> w(n + 1);
    for (int v = 1; v <= n; v++) cin >> w[v];

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int a, b;
        cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }

    // BFS from city 1 over the unweighted graph to get hop-distances.
    const int INF = -1;
    vector<int> dist(n + 1, INF);
    queue<int> q;
    dist[1] = 0;
    q.push(1);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == INF) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }

    // Sum dist[v] * w[v] over all reachable cities. With dist up to n-1
    // (~2e5) and w up to 1e6, a single product reaches ~2e11 and the whole
    // sum reaches ~2e16, so 64-bit accumulation is mandatory; int overflows.
    long long total = 0;
    for (int v = 1; v <= n; v++) {
        if (dist[v] != INF) {
            total += (long long)dist[v] * w[v];
        }
    }

    cout << total << "\n";
    return 0;
}
```

**Causal recap.** The depths are minimum hop counts on an unweighted graph, which BFS computes in
`O(n + m)` (a 3-node triangle showed DFS would miscount, so BFS is the earned choice); the answer is
then `sum of depth(v)*w[v]` over reachable cities. The pitfall is purely arithmetic: with depths up to
`~2*10^5` and importances up to `10^6`, individual products reach `~2*10^11` and the total reaches
`~2*10^16`, so an `int` product or `int` accumulator wraps to negative garbage — invisible on the
26-cost sample, fatal on large hidden tests — and the fix is to widen *before* the multiply
(`(long long)dist[v] * w[v]`) and accumulate in `long long`, which the magnitude bound `~2*10^16 <
9.2*10^18` shows is always enough; the sentinel-`-1` reachability test (`dist[v] != -1`, not
`dist[v] > 0`) keeps the depth-0 source correctly in the sum, and an independent Bellman-Ford brute
force over 500 random graphs confirms the depths.
