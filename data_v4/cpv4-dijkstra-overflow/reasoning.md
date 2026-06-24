**Reading the problem and pinning the contract.** A freight dispatcher wants the cheapest one-way toll route from depot `1` to depot `n` on a directed graph with `n` depots and `m` toll roads; each road `u -> v` adds a fixed toll `w`, tolls accumulate along the route, and I print the minimum total or `-1` if `n` is unreachable. Input is `n m` on the first line, then `m` lines of `u v w`; output is one integer. Before I pick an algorithm I fix the scale, because here the scale decides the *data type* and that is the entire trap. `n <= 2*10^5`, `m <= 5*10^5`, `0 <= w <= 10^9`. A simple route uses at most `n - 1` roads, so the largest possible *finite* answer is about `(2*10^5 - 1) * 10^9 ≈ 2*10^14`. That dwarfs the 32-bit signed maximum `2147483647 ≈ 2.1*10^9`. So distances, heap keys, and every partial sum must be 64-bit `long long`. I write that down first and treat it as non-negotiable: an `int` distance is not "a bit risky", it is a guaranteed silent wrong answer on any test whose optimum crosses ~`2.1*10^9`, and the constraints make such tests trivial to construct. This single observation is the spine of the problem.

**Candidate approaches.** Weights are non-negative (`w >= 0`), which is exactly the regime where Dijkstra's algorithm is correct and fast, so it is the obvious first candidate. Let me still lay the field out so I commit to something I can defend.

- *Greedy cheapest-road walk.* From the current depot, step along the cheapest outgoing road, repeat until I hit `n`. It is `O(route length)` and trivial. But it is obviously unsound: it can step into a dead end, can loop forever on a cycle, and even when it terminates the locally cheapest first road can force an expensive tail. There is no optimality argument at all. Discarded immediately — I only list it to be explicit that "cheapest next edge" is *not* what Dijkstra does.
- *Bellman-Ford.* Relax all `m` edges `n - 1` times: `O(n*m)`. Correct for non-negative (indeed for any non-negative-cycle) graphs, but `O(n*m)` is up to `2*10^5 * 5*10^5 = 10^{11}` — far too slow for the limit. Useful only as an *independent brute-force oracle* for small cases, where its obviousness is a virtue. I will use it exactly that way to verify, never to submit.
- *Dijkstra with a binary heap.* Maintain `dist[v]`, push `(dist, v)` onto a min-heap, pop the smallest, skip stale pops, relax outgoing edges. `O((n + m) log n) ≈ 7*10^5 * ~18 ≈ 1.3*10^7` heap operations — comfortable under 2 seconds. Non-negative weights guarantee that the first time a vertex is popped its distance is final (the settled-vertex invariant). This is what I commit to.

**Why Dijkstra is correct here, briefly.** The invariant: when a vertex `u` is popped from the min-heap with key `d`, `d` is the true shortest distance to `u`. Suppose not; then the real shortest path to `u` reaches some vertex `x` still in the heap before `u`, with `dist[x] < d`. But the heap pops in nondecreasing key order, so `x` would have been popped before `u` — contradiction. The argument needs `w >= 0` so that extending a path never decreases its cost; the constraints give exactly that (`w` can be `0`, which is fine — a zero edge keeps the cost equal, still nondecreasing). Good, the idea is sound; the danger is now entirely in the transcription and the integer width.

**Sanity-checking the derivation on the sample.** The stated sample:

```
5 6
1 2 800000000
2 3 800000000
3 5 800000000
2 4 900000000
4 5 900000000
3 4 50000000
```

Let me run Dijkstra by hand. `dist = [_, 0, INF, INF, INF, INF]` (1-indexed), heap `{(0,1)}`. Pop `(0,1)`: relax `1->2` (800000000) so `dist[2]=800000000`, push. Heap `{(8e8,2)}`. Pop `(8e8,2)`: relax `2->3` -> `dist[3]=1.6e9`, push; relax `2->4` (9e8) -> `dist[4]=1.7e9`, push. Heap has `(1.6e9,3),(1.7e9,4)`. Pop `(1.6e9,3)`: relax `3->5` (8e8) -> `dist[5]=2.4e9`, push; relax `3->4` (5e7) -> `1.6e9+5e7=1.65e9 < 1.7e9`, so `dist[4]=1.65e9`, push. Heap: `(1.65e9,4),(1.7e9,4 stale),(2.4e9,5)`. Pop `(1.65e9,4)`: relax `4->5` -> `1.65e9+9e8=2.55e9`, but `dist[5]=2.4e9` already smaller, no update. Pop `(1.7e9,4)`: stale (`1.7e9 > dist[4]=1.65e9`), skip. Pop `(2.4e9,5)`: that is `n`. Answer `dist[5] = 2400000000`. This matches the stated `2400000000`, and crucially `2.4e9 > 2147483647`, so the sample alone would expose an `int` bug. The recurrence and the unreachable/answer reading are right; now I implement.

**First implementation — and immediately a trace, because the type is the whole point.** My first cut, written the way muscle memory wants to write Dijkstra:

```cpp
const int INF = INT_MAX;
vector<int> dist(n + 1, INF);
dist[1] = 0;
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
pq.push({0, 1});
while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (d > dist[u]) continue;
    for (auto [v, w] : adj[u]) {
        int nd = d + w;                  // <-- int arithmetic
        if (nd < dist[v]) { dist[v] = nd; pq.push({nd, v}); }
    }
}
cout << (dist[n] == INF ? -1 : dist[n]) << "\n";
```

I trace it on the sample above. Everything proceeds as my hand-run until I form distances near `1.6e9`, `1.7e9`, `2.4e9`. The relaxation `int nd = d + w` with `d = 1.6e9` and `w = 8e8` computes `nd = 1600000000 + 800000000 = 2400000000`, but `2400000000` does **not** fit in a signed 32-bit `int` (max `2147483647`). It wraps around to `2400000000 - 2^32 = 2400000000 - 4294967296 = -1894967296`, a *negative* number. Then `if (nd < dist[v])` compares `-1894967296 < INT_MAX`, which is true, so `dist[5]` is set to `-1894967296`. The program would print `-1894967296` for the sample whose correct answer is `2400000000`.

**The bug.** This is the planted pitfall, and the trace nails it precisely: the defect is not in the algorithm but in the *width* of the arithmetic. Three places are infected: (1) `dist` is `vector<int>`, so even a correctly computed large distance cannot be stored; (2) the relaxation `int nd = d + w` overflows *during* the addition, before any comparison, producing a negative `nd` that masquerades as a tiny distance and poisons `dist[v]`; (3) the heap stores `pair<int,int>` keys, so the priority order itself is corrupted once a wrapped negative key enters. I confirm the consequence empirically: an `int` build of this exact code prints `-1894967296` on the sample, while the intended answer is `2400000000`. Every accumulator on the distance path must be 64-bit. Note `w` itself fits in `int` (`<= 10^9 < 2.1*10^9`), so reading `w` as `int` is fine; the overflow is purely in the *sum*, which means I must cast or store in `long long` at the moment of addition, not merely at the end.

**Fix and re-verification.** Make every distance quantity `long long`, use `LLONG_MAX` as `INF`, and force the addition into 64-bit:

```cpp
const long long INF = LLONG_MAX;
vector<long long> dist(n + 1, INF);
...
priority_queue<pair<long long,int>, vector<pair<long long,int>>,
               greater<pair<long long,int>>> pq;
...
long long nd = d + (long long)w;   // 64-bit add: max ~2*10^14, no overflow
```

Re-trace the sample relaxation that broke: `d = 1600000000` (now a `long long`), `nd = 1600000000 + (long long)800000000 = 2400000000`, which fits in `long long` (max `~9.2*10^18`) with room to spare, `2400000000 < INF` so `dist[5] = 2400000000`. The hand-run now reproduces my earlier paper trace exactly, ending with `dist[5] = 2400000000`. I run the compiled `long long` version on the sample: it prints `2400000000`. Fixed, and fixed for the reason the trace identified.

**A second, subtler trace — the `INF + w` underflow trap.** With `INF = LLONG_MAX` I have to worry about a different overflow: what if I ever compute `d + w` when `d` is the sentinel `INF`? `LLONG_MAX + anything` overflows to a *negative* value, which is exactly the kind of silent corruption I just escaped — only now it would make an *unreachable* vertex look reachable with a hugely negative distance. So I trace the disconnected case `3 1 / 1 2 5` (depots 1,2,3; one road `1->2`; target `n = 3` is unreachable). `dist = [_, 0, INF, INF]`, heap `{(0,1)}`. Pop `(0,1)`: relax `1->2`, `dist[2] = 5`, push `(5,2)`. Pop `(5,2)`: `adj[2]` is empty, nothing to relax. Heap empty, loop ends. `dist[3] == INF`, so I print `-1`. The key observation: I **only ever relax outgoing edges of a vertex I have just popped**, and a vertex is pushed (hence eventually popped) only after its `dist` becomes finite via a relaxation. Vertex `3` is never pushed, never popped, so `d + w` is *never* evaluated with `d == INF`. The sentinel is read only inside the comparisons `nd < dist[v]` (where `dist[v]` may be `INF`, and `INF` on the right of `<` is harmless) and in the final `dist[n] == INF` test. So no `INF + w` is ever formed. I double-check there is no stray code path that adds to `dist[v]` directly — there is not; all additions are `d + (long long)w` where `d` came off the heap and is therefore finite. The sentinel is safe. I run the compiled solution on `3 1 / 1 2 5`: it prints `-1`. Correct.

**Stale-entry handling — trace that I do not double-process.** Dijkstra with a lazy heap pushes a vertex multiple times (once per improving relaxation) and never deletes the outdated copies, so I must skip stale pops. My guard is `if (d > dist[u]) continue;`. Trace it on the sample: I pushed `(1.7e9,4)` and later, after finding the cheaper route through `3`, pushed `(1.65e9,4)`; `dist[4]` settled to `1.65e9`. When `(1.7e9,4)` surfaces, `d = 1.7e9 > dist[4] = 1.65e9`, so I `continue` and do not re-relax `4`'s edges. Without this skip I would still get the right distances (relaxations are monotone), but I would do redundant work and, more importantly, the complexity argument relies on each vertex being *processed* at most once. The guard `d > dist[u]` (strict `>`) is correct: when `d == dist[u]` I process exactly the fresh copy; a later equal-key duplicate, if any, is harmless because all its relaxations fail the strict `<`. Good.

**Edge cases, deliberately.**
- `n = 1`: the source is the target. `dist[1] = 0`; I pop `(0,1)`, relax its edges (none of which can lower `dist[1]` since `w >= 0`), and finish with `dist[1] = 0`. The final read `dist[n] = dist[1] = 0`, not `INF`, so I print `0`. Traced on input `1 0`: prints `0`. Correct — the shipment is already at the port.
- `m = 0`, `n > 1`: no roads. Only `(0,1)` is ever popped; `dist[n]` stays `INF`; output `-1`. Correct.
- Self-loop `u -> u` with weight `w`: relaxing it computes `dist[u] + w >= dist[u]` (since `w >= 0`), which never improves `dist[u]`, so it is silently ignored. No infinite loop. Correct.
- Parallel roads `u -> v` with different weights: both sit in `adj[u]`; relaxation naturally keeps the smaller. Correct.
- Zero-weight road: `w = 0` gives `nd = d`, never strictly less than an already-equal `dist[v]`, so no spurious re-push; and the correctness proof only needs `w >= 0`, which includes `0`. Correct.
- Maximum / overflow: heaviest finite answer `~2*10^14` fits in `long long`. The sentinel `INF = LLONG_MAX` is never added to. The heap keys are `long long`. No 32-bit value touches a distance. Safe.
- Output format: exactly one integer plus newline; `cin >>` skips arbitrary whitespace so the line/whitespace layout of the input is irrelevant.

**Brute-force cross-check.** I cannot fully trust a hand-trace, so I verify against an independent method: a Python Bellman-Ford (unbounded integers, so genuinely overflow-free) on small random graphs (`n <= 7`, mixed tiny and near-`10^9` weights, self-loops and parallel edges allowed, sometimes disconnected). Over 500 random seeds the Dijkstra solution and Bellman-Ford agree on every case, including the `-1` (unreachable) ones and the `n = 1` ones. Two different algorithms landing on the same answer 500 times — together with the explicit `int`-vs-`long long` divergence on the sample (`-1894967296` vs `2400000000`) — is the evidence I trust before shipping.

**Final solution.** I disproved the greedy walk, proved Dijkstra's settled-vertex invariant under `w >= 0`, hand-traced the sample to `2400000000`, caught the planted 32-bit overflow by tracing the `1.6e9 + 8e8` relaxation (it wrapped to `-1894967296`), fixed it by making every distance quantity `long long` and forcing the add into 64-bit, separately proved the `INF + w` underflow can never occur because only finite-distance popped vertices are relaxed, and confirmed the corners and 500 brute-force cases. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<pair<int,int>>> adj(n + 1); // (neighbor, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX;
    vector<long long> dist(n + 1, INF);
    dist[1] = 0;

    priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                   greater<pair<long long,int>>> pq;
    pq.push({0LL, 1});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue; // stale entry
        for (auto [v, w] : adj[u]) {
            long long nd = d + (long long)w; // long long: path cost can exceed 32-bit int
            if (nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    if (dist[n] == INF) cout << -1 << "\n";
    else cout << dist[n] << "\n";
    return 0;
}
```

**Causal recap.** Non-negative tolls make Dijkstra the provably correct `O((n+m) log n)` tool, and a hand-run gave the sample answer `2400000000`; but my first `int` implementation computed the relaxation `1600000000 + 800000000` in 32-bit arithmetic, wrapping it to `-1894967296` and poisoning the result — a trace pinned the defect to the *width* of the addition, not the algorithm, so I promoted `dist`, the heap keys, and the sum to `long long` and forced the add into 64-bit. A second trace on a disconnected graph confirmed the new `INF = LLONG_MAX` sentinel never participates in an `INF + w` overflow, because Dijkstra only relaxes from already-popped finite-distance vertices; the stale-pop guard, the `n = 1`, `m = 0`, self-loop, parallel-edge, and zero-weight corners, and 500 agreeing Bellman-Ford cross-checks close the case.
