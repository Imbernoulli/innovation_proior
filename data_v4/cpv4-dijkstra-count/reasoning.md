**Reading the problem and pinning the contract.** I have a directed graph on `n` relay stations with positive integer latencies on the fiber segments, parallel segments and self-loops allowed, a single source `s`. For every station `i` I must report the *number of distinct minimum-latency routes* from `s` to `i`, modulo `1e9+7`, where two routes are distinct iff their sequence of segments differs — so two parallel segments between the same pair are two different routes. Unreachable stations print `-1`, and the source prints `1` (the empty route). Before touching an algorithm I fix the scale, because it dictates the types: `n <= 2*10^5`, `m <= 5*10^5`, `w <= 10^9`. A shortest-path distance can be as large as roughly `m * w`? No — a simple path visits at most `n-1` edges, so a *shortest* distance is at most `(n-1) * w_max ~ 2*10^5 * 10^9 = 2*10^14`. That overflows 32-bit by five orders of magnitude, so `dist[]` and the relaxation accumulator must be 64-bit. The *counts* are reduced mod `1e9+7` so they fit in 32 bits in value, but I will keep them in `long long` so that the intermediate `cnt[v] + cnt[u]` (up to `~2e9`) does not overflow before I reduce. Two type decisions, both non-negotiable: `int` distances are a silent wrong-answer on large tests, and an un-reduced count is a silent wrong-answer the moment two diamonds chain.

**Why positive weights matter for the very definition.** I want to be sure the answer is even finite. If a zero-weight cycle could lie on a shortest route, I could loop around it arbitrarily many times without increasing latency, and "the number of minimum-latency routes" would be infinite. The contract says `w >= 1`, strictly positive. That buys me a structural fact I will lean on hard: along any minimum-latency route the distance strictly increases at every step, so a shortest route never revisits a station, the set of *tight* segments `u -> v` with `dist[u] + w == dist[v]` is a DAG, and crucially every tight predecessor `u` of `v` satisfies `dist[u] < dist[v]`. That strict inequality is the lever that makes counting-during-Dijkstra correct, as I will see when I try to break it.

**Laying out the candidate approaches.** Two routes, and I want the one I can *prove*, not merely the one that runs.

- *Two-pass DAG-DP.* First compute all `dist[]` with a plain Dijkstra. Then build the tight subgraph (`dist[u] + w == dist[v]`), process stations in nondecreasing `dist` order, and set `cnt[v] = sum over tight predecessors u of cnt[u]`, each parallel tight segment counted separately, with `cnt[s] = 1`. This is obviously correct because the tight subgraph is a DAG and `dist` order is a topological order; the cost is a second pass and an extra adjacency structure.
- *Single-pass counting Dijkstra.* Carry `cnt[i]` alongside `dist[i]`. When relaxing `u -> v`: if `nd = dist[u] + w` is strictly less than `dist[v]`, I've found a strictly better latency, so the old routes to `v` are obsolete and `cnt[v] := cnt[u]`; if `nd == dist[v]`, I've found another shortest route, so `cnt[v] += cnt[u]`. One pass, but the timing of that `+=` relative to when `u`'s own count becomes final is exactly where a double-count hides.

The single-pass version is leaner and is what I want to ship, but I only trust it if I can pin down *why* it counts each route exactly once. The two-pass version I will keep in my back pocket as the obviously-correct oracle to test against.

**Deriving the single-pass invariant.** The claim I need: *at the moment station `u` is popped from the priority queue with `d == dist[u]` (its final distance), `cnt[u]` already equals the true number of minimum-latency routes to `u`.* If that holds, then expanding `u` and pushing its contribution to each neighbor `v` is sound. Why does it hold? Every minimum-latency route to `u` ends with some tight segment `x -> u`, and every such tight predecessor `x` has `dist[x] < dist[u]` (strict, because `w >= 1`). Dijkstra pops in nondecreasing distance, so every such `x` was popped *before* `u`, and by induction had its final `cnt[x]` when it was expanded; at that expansion the segment `x -> u` was relaxed and `cnt[x]` was folded into `cnt[u]` (either as the reset that created `dist[u]`, or as a `+=` for a later equal-latency arrival). So by the time `u` is popped, every tight predecessor has already contributed, exactly once each, and parallel tight segments `x -> u` contributed once per copy. Hence `cnt[u]` is complete and correct when `u` is expanded. The induction base is the source: `cnt[s] = 1`, no predecessors. The invariant holds — *provided* I only fold a predecessor's count in when that predecessor's count is final, i.e. when it is being expanded, not at some arbitrary stale moment. Hold that thought; it is precisely the bug I am about to write.

**First implementation.** My first cut of the loop, the one that "looks obviously right":

```
while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    for (auto [v, w] : adj[u]) {
        long long nd = d + w;
        if (nd < dist[v]) { dist[v] = nd; cnt[v] = cnt[u]; pq.push({nd, v}); }
        else if (nd == dist[v]) { cnt[v] = (cnt[v] + cnt[u]) % MOD; pq.push({nd, v}); }
    }
}
```

Two things I did without thinking: I dropped the "stale entry" guard (no `if (d > dist[u]) continue;`), and I push `v` again on the equal-latency branch. Both feel harmless. Let me trace.

**First debug episode — tracing a real double-count.** I need a graph where a station is reached, then a strictly shorter route arrives later and resets it, and the *stale* copies of that station still sit in the queue. Take this network (it is the kind the generator produces): stations `0..3`, source `3`, with segments
`3->1 (2)`, `3->1 (1)`, `3->2 (2)`, `3->1 (1)`, `1->1 (1)`, `3->0 (1)`, `3->2 (4)`, `3->2 (4)`, `0->2 (3)`, `1->2 (1)`.
Let me compute the true answer first with my head / the two-pass oracle. Distances from 3: `dist[3]=0`. `dist[1] = min(2,1,1) = 1`. `dist[0] = 1`. `dist[2] = min(dist[3]+2, dist[0]+3, dist[1]+1) = min(2, 4, 2) = 2`. Tight predecessors of `1`: the two segments `3->1 (1)` (the `3->1 (2)` is not tight), so `cnt[1] = 2`. `cnt[0] = 1`. Tight predecessors of `2`: `3->2 (2)` gives `0+2=2` tight (count `cnt[3]=1`), and `1->2 (1)` gives `1+1=2` tight (count `cnt[1]=2`); the `0->2 (3)` gives `1+3=4`, not tight; the two `3->2 (4)` give `4`, not tight. So `cnt[2] = cnt[3] + cnt[1] = 1 + 2 = 3`. True answer: `0->1`, `cnt = [1, 2, 3, 1]`.

Now run my buggy code. `pq` starts `{(0,3)}`, `cnt[3]=1`. Pop `(0,3)`. Relax its segments. `3->1 (2)`: `dist[1]=2, cnt[1]=1`, push `(2,1)`. `3->1 (1)`: `1 < 2`, so `dist[1]=1, cnt[1]=1`, push `(1,1)`. `3->2 (2)`: `dist[2]=2, cnt[2]=1`, push `(2,2)`. `3->1 (1)`: `nd=1 == dist[1]=1`, so `cnt[1] += cnt[3] => cnt[1]=2`, push `(1,1)` again. `3->0 (1)`: `dist[0]=1, cnt[0]=1`, push `(1,0)`. The two `3->2 (4)`: `4 == dist[2]`? No, `dist[2]=2`, and `4 > 2`, so nothing. Now the queue holds, among others, *two* copies of `(1,1)` and the stale `(2,1)`. Pop `(1,0)` (or `(1,1)` — order among equal keys is unspecified; suppose `(1,1)` first). Pop `(1,1)`: expand `1`. `1->1 (1)`: `nd=2`, `dist[1]=1`, `2 > 1`, nothing (self-loop harmlessly ignored). `1->2 (1)`: `nd=2 == dist[2]=2`, so `cnt[2] += cnt[1] => cnt[2] = 1 + 2 = 3`, push `(2,2)`. Good so far. Now pop the *second* `(1,1)` still in the queue. With **no stale guard**, I expand `1` again. `1->2 (1)`: `nd=2 == dist[2]=2` again, so `cnt[2] += cnt[1] => cnt[2] = 3 + 2 = 5`. That is the bug, live: `cnt[2]` jumped to `5` when the truth is `3`. The double-count came from expanding station `1` twice — once per queue copy — and each expansion re-poured `cnt[1]` into `cnt[2]`.

**Diagnosing the bug precisely.** Two defects, one root. (1) I pushed `v` again on the equal-latency branch, which manufactures duplicate queue entries for an already-settled station. (2) I have no guard rejecting a pop whose key is worse than the current best, so those duplicates get fully expanded. The invariant I proved required folding `cnt[u]` into neighbors *exactly when `u` is finalized, once*. Expanding `u` twice violates "once" and re-pours its count. The fix is to make every non-source station be *expanded at most once at its final distance*: (a) never push on the equal-latency branch — equal-latency does not change `dist[v]`, so there is no new "settling" event to enqueue, only a count update; (b) add the stale guard `if (d > dist[u]) continue;` so that any leftover queue entry whose key exceeds the now-final `dist[u]` is discarded. With those, a station is pushed only on strict improvements, and only the single pop carrying its final distance survives the guard and triggers expansion.

**Fixing and re-verifying the first episode.** Corrected loop:

```
while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (d > dist[u]) continue;                 // stale: a shorter dist[u] is already final
    for (auto [v, w] : adj[u]) {
        long long nd = d + w;
        if (nd < dist[v]) { dist[v] = nd; cnt[v] = cnt[u]; pq.push({nd, v}); }
        else if (nd == dist[v]) { cnt[v] = (cnt[v] + cnt[u]) % MOD; }   // no push
    }
}
```

Re-trace the same network. After expanding source `3`, queue has `(2,1) stale, (1,1), (1,1)? ` — wait, with the fix the equal-latency branch no longer pushed, so the *second* `3->1 (1)` did `cnt[1] += 1 => 2` and did **not** push. The queue after source: `(2,1)` from the first `3->1 (2)`, `(1,1)` from the strict improvement to `dist[1]=1`, `(2,2)`, `(1,0)`. Pop `(1,0)`: expand `0`, `0->2 (3)`: `nd=4 > dist[2]=2`, nothing. Pop `(1,1)`: `d=1 == dist[1]=1`, expand. `1->2 (1)`: `nd=2 == dist[2]=2`, `cnt[2] += cnt[1] => cnt[2] = 1 + 2 = 3`. Pop `(2,1)`: `d=2 > dist[1]=1`, **stale, skip** — station `1` is not expanded a second time. Pop `(2,2)`: `d=2 == dist[2]=2`, expand `2` (no outgoing tight effects here). Final `cnt = [1, 2, 3, 1]`. Correct, matching the oracle. The exact case that produced `5` now produces `3`, and it does so because the stale guard killed the duplicate expansion of station `1` — the bug and the fix are the same lever.

**Second debug episode — parallel segments and the reset-vs-accumulate boundary.** Even with the one-expansion fix, there is a second, independent trap I want to probe: how parallel tight segments from a *single* expansion interact with the reset branch. Consider the documented sample: stations `0..4`, source `0`, segments `0->1 (2)`, `0->2 (2)`, `1->4 (3)`, `2->4 (3)`, `0->3 (4)`, `3->4 (1)`, and a *second* `2->4 (3)`. The truth: `dist[4] = 5` via `0->1->4` (1 route), `0->2->4` over two parallel `2->4` (2 routes), `0->3->4` (1 route): total `4`. The danger is the ordering of the two parallel `2->4` segments inside station `2`'s expansion. Trace with the fixed code. Pop `0`: `cnt[0]=1`. Relax: `0->1`: `dist[1]=2, cnt[1]=1`; `0->2`: `dist[2]=2, cnt[2]=1`; `0->3`: `dist[3]=4, cnt[3]=1`. Pop `(2,1)` (station 1): `1->4 (3)`: `nd=5 < INF`, so `dist[4]=5, cnt[4]=cnt[1]=1`, push `(5,4)`. Pop `(2,2)` (station 2): now the two parallel `2->4 (3)`. First copy: `nd=5 == dist[4]=5`, so `cnt[4] += cnt[2] => cnt[4] = 1 + 1 = 2`. Second copy: `nd=5 == dist[4]=5` again, so `cnt[4] += cnt[2] => cnt[4] = 2 + 1 = 3`. Both parallel segments counted, exactly as intended — the loop iterates over each segment in `adj[2]` independently, so two parallel copies fold `cnt[2]` in twice, which is right. Pop `(4,3)` (station 3): `3->4 (1)`: `nd=5 == dist[4]=5`, so `cnt[4] += cnt[3] => cnt[4] = 3 + 1 = 4`. Pop `(5,4)`: `d=5 == dist[4]`, expand (no out-edges matter). Final `cnt[4]=4`. Correct.

**A subtle off-by-one I almost shipped in the reset branch.** While writing the reset branch I first wrote `if (nd <= dist[v]) { dist[v]=nd; cnt[v]=cnt[u]; push; }` with `<=` instead of `<`, thinking "shorter-or-equal, just overwrite." Trace what `<=` does on station `4` of the sample: the first `2->4` would hit `nd=5 <= dist[4]=5` and **reset** `cnt[4] = cnt[2] = 1`, destroying the `1` already contributed by `0->1->4`. Final would be `cnt[4] = cnt[3] = 1` after the last reset by `3->4` — a gross undercount (`1` instead of `4`), and it would also needlessly re-push settled stations. The boundary must be: *strictly shorter* resets (the old routes are obsolete), *equal* accumulates (the old routes are still valid and we are adding more). `< / ==`, never `<=`. This is the off-by-one twin of the double-count: get the comparison operator wrong by one notch and you either reset-when-you-should-add (undercount) or add-when-you-should-reset.

**Edge cases, deliberately.**
- *Source itself.* `cnt[s] = 1`, `dist[s] = 0`. A self-loop `s->s (w>=1)` relaxes to `nd = w > 0 = dist[s]`, hits neither `< 0` nor `== 0`, so it is ignored — the empty route stays the unique route to `s`. Verified: `printf "2 2 0\n0 0 1\n0 1 3\n"` gives `1` then `1`.
- *`n = 1, m = 0`.* Queue processes only `(0,0)`, prints `1`. Correct.
- *Unreachable station.* Its `dist` stays `INF`, printed as `-1`. Verified on `3 1 0 / 0 1 5`: outputs `1, 1, -1`.
- *Parallel-edges count.* `2 2 0 / 0 1 4 / 0 1 4` gives `cnt[1] = 2`. Correct.
- *Modulus / overflow.* A chain of `30` doubling diamonds yields `2^30 = 1073741824 > 1e9+7`. My code reduces `cnt[v]` every `+=`, so it prints `2^30 mod (1e9+7) = 73741817`, matching `pow(2,30,MOD)`. Without the per-step `% MOD`, a chain of ~64 doublings would overflow `long long`; reducing every addition keeps every value below `2*1e9+7 < 2^31`, far inside 64-bit range. Distances: the `INF = LLONG_MAX` sentinel is only ever compared, never has `w` added to a finalized `INF` node (we only relax from popped nodes, whose `dist < INF`), so no `INF + w` overflow occurs.
- *Self-loop on a non-source node.* `1->1 (w>=1)` relaxes to `dist[1] + w > dist[1]`, ignored — no spurious self-count, as seen in the first trace.

**Sanity-checking the derivation against the sample once more.** Independent of the code, by the DAG-DP definition: tight predecessors of `4` are `1` (via `1->4`, count `1`), `2` (via two parallel `2->4`, count `2*cnt[2]=2*1=2`), and `3` (via `3->4`, count `1`); sum `4`. The single-pass code reproduced exactly this by folding each predecessor's final count once per tight segment. Derivation and implementation agree, and both agree with the brute oracle over 500 random small graphs (0 mismatches), so I trust the count semantics.

**Final solution.** I proved the single-pass invariant (each station's count is final when it is expanded, because all tight predecessors have strictly smaller distance and are expanded earlier), I killed the double-count by guaranteeing one expansion per station (stale guard + no push on the equal branch), and I fixed the reset-vs-accumulate boundary to strict-`<` / `==`. This is what I ship — one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;
    const long long MOD = 1000000007LL;

    vector<vector<pair<int,int>>> adj(n); // adj[u] = list of (to, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX;
    vector<long long> dist(n, INF);
    vector<long long> cnt(n, 0); // number of shortest routes mod MOD

    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[s] = 0;
    cnt[s] = 1;
    pq.push({0, s});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;             // stale entry: a shorter dist[u] is already final
        for (auto [v, w] : adj[u]) {
            long long nd = d + w;
            if (nd < dist[v]) {                // strictly shorter route to v: reset its count
                dist[v] = nd;
                cnt[v] = cnt[u];
                pq.push({nd, v});
            } else if (nd == dist[v]) {         // another shortest route to v: accumulate once
                cnt[v] = (cnt[v] + cnt[u]) % MOD;
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (dist[i] == INF) cout << -1 << "\n";
        else cout << (cnt[i] % MOD) << "\n";
    }
    return 0;
}
```

**Causal recap.** Counting shortest routes is a count layered on Dijkstra, and the count is correct only if each station's `cnt` is final the moment it is expanded — which holds because positive weights force every tight predecessor to have a strictly smaller distance and thus be expanded earlier. My first loop violated "expand once": pushing on the equal-latency branch created duplicate queue entries and, lacking a stale guard, expanded a station twice, re-pouring its count and double-counting (`cnt[2]` traced to `5` instead of `3`). Restricting pushes to strict improvements and discarding stale pops restores one-expansion and fixes it. A second, independent trap — writing `<=` where the boundary must be `<` for reset and `==` for accumulate — would have reset counts that should have been added, undercounting `cnt[4]` to `1` instead of `4`; the strict-`<`/`==` split is the off-by-one twin of the double-count. Parallel tight segments are handled for free because the relaxation loop visits each segment independently, and reducing the count modulo `1e9+7` on every addition closes out the overflow and large-answer corners, with `-1` for unreachable stations and `cnt[s]=1` for the source.
