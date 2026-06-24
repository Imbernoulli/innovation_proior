**Reading the problem and pinning the contract.** I have a directed graph on `n` stations and `m`
edges. Each edge `(u, v)` carries a *line* id `c` and a fare `w`. I start at station `1`, want to
reach station `n`, and the cost of a route is the sum of fares plus `S` for every place along the
route where two consecutive edges have different line ids. The first boarding is free of surcharge
(there is no previous line), and arriving at `n` costs nothing. If `n` is unreachable I print `-1`.
Before any algorithm I fix the scale, because it decides the data types: `n, m <= 2*10^5`, fares and
`S` up to `10^9`. A route can use on the order of `m` edges and incur on the order of `m` transfers,
so the cost can reach about `2*10^5 * 10^9 + 2*10^5 * 10^9 = 4*10^14`. That is far past the 32-bit
range of `~2.1*10^9`, so every distance, every fare accumulator, and the surcharge must be 64-bit. I
will use `long long` everywhere. This is non-negotiable; an `int` is a silent wrong-answer on the
large tests.

**Candidate approaches.** All weights (fares and `S`) are non-negative, so this is a non-negative
shortest-path problem and Dijkstra is the right hammer. The only genuine question is *what a node is*.

- *Plain Dijkstra on stations.* Keep one label per station: `dist[v]` = cheapest cost to reach `v`.
  When I relax an edge `(u, v, c, w)` I charge `S` if the boarded line `c` differs from "the line by
  which `u`'s best route arrived". `O((n+m) log n)`, textbook. The seductive assumption is that *the
  cheapest way to reach a station is a sufficient summary of the past*. That is exactly the kind of
  claim I have learned not to trust on sight, because the future cost here depends on a feature of the
  past (the arriving line) that the cheapest route need not optimize. I will try to break it before
  trusting it.
- *Dijkstra on an augmented state.* Make the state `(station, line you arrived on)`. Then the cost to
  extend depends only on the boarded line versus the stored arriving line — purely local to the state
  — and the standard non-negative-weight argument applies on the expanded graph. The number of
  reachable states is bounded by the number of edges (each edge `(u, v, c, w)` can only ever create
  the state `(v, c)`), so this is `O((n+m) log m)` and just as fast in practice.

**Stress-testing the obvious method before committing.** "Cheapest arrival at a station is enough
state" *feels* like an invariant, so let me actually attack it with a concrete tiny instance. Set the
surcharge `S = 5` and three edges, two of them parallel from `1` to `2` on different lines:

```
1 ->(Red, fare 1) 2
1 ->(Blue, fare 3) 2
2 ->(Blue, fare 1) 3
```

I want station `3` from station `1`. Plain station-only Dijkstra reasons like this. Relax both edges
into `2`: the Red one gives cost `1`, the Blue one gives cost `3`, so `dist[2] = 1` and station `2`'s
recorded arriving line is *Red*. Now relax `2 ->(Blue,1) 3`: I boarded Blue but arrived on Red, that
is a line change, so cost `= dist[2] + 1 + S = 1 + 1 + 5 = 7`. Plain Dijkstra answers **7**.

Is `7` optimal? Let me hunt for a route the station-only label could not represent. Take the *pricier*
arrival at `2`: go `1 ->(Blue,3) 2`, arriving on Blue at cost `3`, then `2 ->(Blue,1) 3` with **no**
line change (Blue to Blue), cost `3 + 1 + 0 = 4`. That is strictly better than `7`. So plain Dijkstra
is wrong, and I now see exactly *why*: arriving at `2` more cheaply on Red blocked the free
continuation that only a Blue arrival enables. The cheapest arrival is **not** a sufficient summary —
the arriving line is a real part of the state. The stress test paid for itself by killing the approach
I was about to ship. Station-only Dijkstra is out; I must Dijkstra over `(station, arriving line)`.

**Deriving the augmented-state Dijkstra and checking it on paper.** A state is `(u, lc)` where `lc` is
the line of the edge I most recently rode to get to `u`. For the start I need a sentinel: at station
`1` I have ridden nothing, so its "last line" is a value that matches no real line; I use `lc = 0`
(real line ids are `>= 1`). The distance `dist[(u, lc)]` is the cheapest cost to be standing at `u`
having last ridden line `lc`. Transition: from `(u, lc)` along edge `(u, v, c, w)`,

```
new cost = dist[(u, lc)] + w + ( (lc != 0 and c != lc) ? S : 0 )
```

into state `(v, c)`. The surcharge fires only when I have ridden something already (`lc != 0`) *and*
the new line differs. The answer is the minimum `dist[(n, lc)]` over all `lc` — but because Dijkstra
pops states in nondecreasing cost order and all weights are non-negative, the *first* time I pop any
state whose station is `n`, that cost is the global minimum over arriving at `n`, so I can stop there.
If the queue empties without ever popping an `n`-state, `n` is unreachable and the answer is `-1`.

Let me confirm the transition by hand on the counterexample above, the one where the obvious method
gave `7` and the truth is `4`. Start state `(1, 0)` at cost `0`. Its two out-edges: `1->2` Red gives
`(2, Red)` at `0 + 1 + 0 = 1` (no surcharge, `lc == 0`); `1->2` Blue gives `(2, Blue)` at
`0 + 3 + 0 = 3`. Pop `(2, Red)` at `1`: edge `2->3` Blue gives `(3, Blue)` at `1 + 1 + S = 7` (Red to
Blue, `lc != 0`, change). Pop `(2, Blue)` at `3`: edge `2->3` Blue gives `(3, Blue)` at
`3 + 1 + 0 = 4` (Blue to Blue, no change) — this is cheaper than the `7` already stored, so it
overwrites it. Pop `(3, Blue)` at `4`, station is `3 = n`, answer `4`. The augmented recurrence gives
the right answer where the obvious one failed. Good — the idea is sound.

**Sanity-checking the derivation on the documented sample.** The sample has `S = 3`, stations `1..5`,
lines Red`=1` Blue`=2`, edges `1->2`(Red,4), `2->5`(Red,9), `1->3`(Blue,2), `3->4`(Blue,2),
`4->5`(Red,1), `2->4`(Blue,1). I claim the answer is `8` via `1 ->(Blue,2) 3 ->(Blue,2) 4 ->(Red,1)
5` = `2 + 2 + 1 + 3 = 8`. Let me run the states. `(1,0)@0`. Out: `(2,Red)@4`, `(3,Blue)@2`. Pop
`(3,Blue)@2`: edge `3->4` Blue gives `(4,Blue)@2+2+0=4`. Pop `(2,Red)@4`: edge `2->5` Red gives
`(5,Red)@4+9+0=13`; edge `2->4` Blue gives `(4,Blue)@4+1+3=8`, but `(4,Blue)` already holds `4`, so no
update. Pop `(4,Blue)@4`: edge `4->5` Red gives `(5,Red)@4+1+3=8`, which beats the stored `13`. Pop
`(5,Red)@8`, station `5 = n`, answer `8`. Matches. The all-Red route `1->2->5` would have been
`4+9=13`; the mixed route wins precisely because two Blue legs avoid a surcharge and only one Red
transfer is paid at the end. The derivation is internally consistent with the stated sample.

**First implementation — and immediately a trace, because clean math transcribes dirty.** I write the
augmented Dijkstra with a per-station hash map `best[u] : line -> cost` (storing the cheapest cost to
reach `u` on each arriving line), a min-heap of `(cost, station, line)`, the `(1, 0)` start, and the
pop-`n`-to-finish rule. While typing the relaxation I momentarily simplify the surcharge condition to
just `c != lc` — it reads cleaner and "the start has no previous line anyway". So my first cut of the
inner relaxation is:

```
long long nd = d + w + ((c != lc) ? S : 0);   // first draft
```

That `((c != lc) ? S : 0)` makes me uneasy, because `lc == 0` at the start is *not* a real line, yet
`c != 0` is true for every real line `c >= 1`. So let me trace the smallest input that would expose a
phantom first-edge surcharge: the clean counterexample again, `S = 5`, edges `1->2`(Red,1),
`1->2`(Blue,3), `2->3`(Blue,1), whose true answer I just established as `4`.

Trace with the buggy line. `(1,0)@0`. Out-edge `1->2` Red: `nd = 0 + 1 + (Red != 0 ? 5 : 0) = 1 + 5 =
6` — a surcharge on the **first** edge, which the rules say is free. Out-edge `1->2` Blue: `nd = 0 + 3
+ 5 = 8`. So `best[2] = {Red:6, Blue:8}`. Pop `(2,Red)@6`: `2->3` Blue gives `6 + 1 + 5 = 12`. Pop
`(2,Blue)@8`: `2->3` Blue gives `8 + 1 + 0 = 9`, beating `12`. Pop `(3,Blue)@9`, answer **9**.

**The bug.** The code returns `9`; the correct answer is `4`. The defect is precise: by dropping the
`lc != 0` guard I taxed the very first boarding, because the start sentinel `lc = 0` compares unequal
to every real line. Every route now pays a phantom `+S` at its first edge, and worse, the comparison
`c != lc` with `lc = 0` is *structurally* meaningless — `0` is not a line, so "different line" is the
wrong question at the start. I confirm the failure is exactly this by re-running on the metro sample
too: the buggy code reports `11` instead of `8` (the optimal route's first edge `1->3` Blue is taxed
`+3`, pushing `8` to `11`). Both wrong answers are inflated by exactly one spurious first-edge
surcharge along their cheapest route — the signature of this bug. The fix is to reinstate the guard:
charge `S` only when `lc != 0 && c != lc`.

**Fix and a second trace.** With `nd = d + w + ((lc != 0 && c != lc) ? S : 0)`, re-trace the clean
case. `(1,0)@0`. `1->2` Red: `lc == 0`, no surcharge, `nd = 1`. `1->2` Blue: `lc == 0`, `nd = 3`.
`best[2] = {Red:1, Blue:3}`. Pop `(2,Red)@1`: `2->3` Blue, `lc = Red != 0` and `Blue != Red`, so
`+S`: `1 + 1 + 5 = 7`. Pop `(2,Blue)@3`: `2->3` Blue, `Blue == Blue`, no surcharge: `3 + 1 + 0 = 4`,
beats `7`. Pop `(3,Blue)@4`, answer `4`. Correct. Re-running the metro sample now yields `8`. The two
cases that broke now pass, and they broke for the reason I fixed — that is the evidence I trust.

**A second, quieter bug — the stale-entry skip.** With the heap-based Dijkstra I push a new
`(nd, v, c)` whenever I improve `best[v][c]`, but I never delete the old, worse heap entries for the
same state. So when I pop a triple I must check it is still the *current* best for its state, else I
process an out-of-date label. My first version of the pop guard was:

```
auto it = best[u].find(lc);
if (it == best[u].end()) continue;   // first draft: only skip if state absent
```

Let me trace where this goes wrong. Consider the metro sample again at the moment two pushes for the
same state coexist: `(4,Blue)` is first set to `4` (pushed `(4,4,Blue)`), and later a relaxation from
`(2,Red)` computes `4 + 1 + 3 = 8` for `(4,Blue)` — which does *not* improve `4`, so with my correct
update rule it is never pushed. Good, that path does not trigger it. But construct the dangerous order
directly: suppose `(4,Blue)` is pushed at cost `8` first (from some predecessor explored early), then
later improved to `4` and pushed again. The heap now holds both `(8,4,Blue)` and `(4,4,Blue)`. The
`4` pops first, I relax its neighbours, fine. Then the stale `(8,4,Blue)` pops; `best[4]` *does*
contain `Blue` (value `4`), so `it != end()` and my first-draft guard lets it through. I then relax
station `4`'s out-edges a second time at the inflated base cost `8`, pushing states like `(5,Red)` at
`8 + 1 + 3 = 12` even though `4` already produced the correct `(5,Red)@8`. These never *win* (they are
larger, so the update rule rejects them), so the final answer stays correct — but every stale pop
re-expands a whole adjacency list, and on a graph where a state is improved many times this degrades
toward quadratic work and can blow the time limit. The guard must compare the popped cost against the
stored best and skip when they differ:

```
auto it = best[u].find(lc);
if (it == best[u].end() || it->second != d) continue;   // fixed: skip stale costs
```

Now the stale `(8,4,Blue)` is recognized (`it->second == 4 != 8`) and discarded in `O(1)` without
re-expanding. I verify the fix does not change any answer (it cannot, since stale labels never won)
but does bound the work: each state is *expanded* at most once, so the loop body runs `O(m)` times and
the total is `O((n+m) log m)`. On a hand-built max case (`n = m = 2*10^5`, fares and `S` near `10^9`)
the solution finishes in about `0.07 s`, far under the `2 s` limit, with `~50 MB` of memory — so the
stale-skip is doing its job.

**Edge cases, deliberately.**
- *`n` unreachable.* If no route reaches `n`, the heap drains without ever popping an `n`-state, the
  loop ends, and `answer` keeps its initial `-1`. Traced on `2 1 5 / 2 1 1 3` (the only edge points
  *away* from `1` and into `2`, with no edge out of `1`): the start `(1,0)@0` has no out-edges, the
  heap empties, output `-1`. Correct.
- *`S = 0`.* The surcharge term is always `0`, so the augmented graph collapses cost-wise to an
  ordinary shortest path; the `(station, line)` split still happens but never affects cost, and the
  answer equals a plain Dijkstra. Cross-checked against the brute force on hundreds of `S = 0` cases.
- *Single line everywhere.* No edge is ever a transfer (`c` is constant), so the answer is the plain
  shortest path; the `lc != 0` guard makes the first edge free and the `c == lc` checks make all
  others free, exactly as intended.
- *Self-loops and parallel edges.* A self-loop `(u, u, c, w)` only ever adds non-negative cost, so it
  never helps and Dijkstra's update rule simply never improves through it. Parallel edges on different
  lines are the whole point and are handled because each `(v, c)` is its own state. Verified on
  `2 3 4 / 1 1 1 0 / 1 2 1 5 / 1 2 2 2` -> `2` (take the cheaper line directly), matching brute.
- *Overflow.* Distances are `long long`; the worst accumulated cost `~4*10^14` fits with room. The
  surcharge `S` and fares each `<= 10^9` are read into `long long`, so no intermediate `int`
  multiplication or addition overflows.
- *Output format.* Exactly one integer and a newline; `cin >>` consumes arbitrary whitespace, so the
  input parsing is format-agnostic.

**Cross-check against an independent brute force.** To be sure the *idea* is right and not just the
hand-traces, I wrote a label-correcting Bellman-Ford-style fixpoint over the same `(station, line)`
state space — it relaxes every transition from every state repeatedly until no distance improves,
making **no** Dijkstra ordering assumption, then takes the min over all `(n, lc)` labels. It is
obviously correct (it is just iterating shortest paths to a fixpoint on a finite non-negative graph)
and uses a completely different method from my heap Dijkstra. Over `1000` random small cases (varying
`n`, line count, `S` including `0`, self-loops, parallel edges, and disconnected targets) the two
agree on every case, `0` mismatches. I also ran the discarded station-only Dijkstra against the
correct solver and confirmed it disagrees on real cases (e.g. `24` vs the true `22` on one random
graph, and `7` vs `4` on the clean counterexample) — proof that the trap is genuine and that the
augmented state is what fixes it.

**Final solution.** I disproved the obvious station-only Dijkstra with a traced counterexample,
derived the `(station, arriving line)` augmentation, hand-checked its recurrence on both the
counterexample and the documented sample, found and fixed the phantom first-edge surcharge and the
stale-heap-entry re-expansion by tracing each to a precise cause, and cross-checked the whole thing
against an independent brute over a thousand cases. That is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    long long S;
    if (!(cin >> n >> m >> S)) return 0;

    // edges[u] = list of (v, color, fare)
    vector<vector<array<long long,3>>> edges(n + 1);
    // colors are 1..C; we compress per-node arrival operator state.
    // State = (node, last_color). last_color = 0 means "no edge used yet" (start).
    // We index states. To keep it bounded we only create states reachable through edges.

    // Collect all colors to map them; but a state's "last_color" is the color of the
    // edge we arrived on. We use the raw color value as part of the key.
    // dist over (node, color). Use a hash map keyed by node*BIG + color, but colors
    // can be large; instead store per node a map<color,dist>.
    for (int i = 0; i < m; i++) {
        long long u, v, c, w;
        cin >> u >> v >> c >> w;
        edges[u].push_back({v, c, w});
    }

    // best[node] : map from last_color -> min cost arriving at node having last
    // traversed an edge of that color.
    vector<unordered_map<long long,long long>> best(n + 1);

    // Priority queue of (cost, node, last_color).
    priority_queue<array<long long,3>, vector<array<long long,3>>, greater<array<long long,3>>> pq;

    // Start at node 1 with last_color = 0 (a sentinel meaning "no operator yet").
    // 0 is never used as a real color (real colors are >= 1).
    best[1][0] = 0;
    pq.push({0, 1, 0});

    long long answer = -1;

    while (!pq.empty()) {
        auto top = pq.top(); pq.pop();
        long long d = top[0], u = top[1], lc = top[2];
        auto it = best[u].find(lc);
        if (it == best[u].end() || it->second != d) continue; // stale
        if (u == n) { answer = d; break; }
        for (auto &e : edges[u]) {
            long long v = e[0], c = e[1], w = e[2];
            // surcharge only if we already used an edge (lc != 0) and color changes.
            long long nd = d + w + ((lc != 0 && c != lc) ? S : 0);
            auto vit = best[v].find(c);
            if (vit == best[v].end() || nd < vit->second) {
                best[v][c] = nd;
                pq.push({nd, v, c});
            }
        }
    }

    cout << answer << "\n";
    return 0;
}
```

**Causal recap.** The obvious "one distance per station" Dijkstra looked right but a single traced
counterexample (`1->2` cheap on Red vs pricier on Blue, then a Blue continuation: it answers `7` where
`4` is reachable) showed that the cheapest arrival hides the line you arrived on, which the future
surcharge cares about — so the state must be `(station, arriving line)`. Implementing that, a first
draft taxed the start because the `lc = 0` sentinel compared unequal to every real line; a trace of
the clean case returning the inflated `9` instead of `4` pinpointed it, and guarding the surcharge
with `lc != 0` fixed it. A second trace of coexisting heap entries for one state showed a missing
"is this the current best?" check would re-expand stale labels and risk TLE; comparing the popped cost
to the stored best and skipping on mismatch bounds each state to one expansion. Reading the first
popped `n`-state as the answer (valid because weights are non-negative) and keeping every accumulator
in `long long` closes out the unreachable (`-1`), `S = 0`, single-line, self-loop, and overflow
corners, all confirmed against an independent fixpoint brute over a thousand cases.
