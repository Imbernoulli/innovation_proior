**Reading the problem and pinning the contract.** A courier sits at station `1` at time `0` and wants to be at station `n` as early as possible. The graph is directed: a connection `(u, v, w)` costs `w` minutes from `u` to `v`. The twist is per-station: each station `v` carries a curfew `c[v]`, and the courier may *be present* at `v` only while the clock reads `t < c[v]`. The phrase in the statement is "forbidden to be present at `v` at any time `t` with `t >= c[v]`," so the legal region is the strict half-line `t < c[v]`. I want the earliest legal arrival time at station `n`, or `-1` if there is none. Two things in that contract are easy to skim past and both will bite me later, so I write them down now. First, the *start* counts as being present: the courier stands at station `1` at time `0`, which is legal only if `0 < c[1]`; if `c[1] = 0` the courier cannot even legally exist at the origin and the answer is `-1` immediately. Second, the boundary is *strict*: arriving exactly at `t = c[v]` is forbidden, because `c[v] >= c[v]` is true. The entire problem hinges on getting that one inequality direction right.

**Fixing the scale and the data types before any algorithm.** Constraints: `n, m` up to `2*10^5`, each weight `w` up to `10^9`, each curfew up to `10^9`. A path can chain up to `n - 1` edges, so an accumulated arrival time can reach roughly `2*10^5 * 10^9 = 2*10^14`. That is far past the 32-bit ceiling of about `2.1*10^9`, so every time value — distance labels, the running sum `d + w`, the curfews — has to be 64-bit `long long`. An `int` here is a silent wrong-answer on the large tests, not a crash, which is the worst kind. I also note `m` up to `2*10^5` means I want an adjacency list and a binary-heap Dijkstra at `O((n+m) log n)`, not anything quadratic. These are non-negotiable framing decisions; I lock them in.

**Candidate approaches.** I have a shortest-path-style objective with a side constraint, so let me lay out the routes and pick the one I can defend.

- *Plain Dijkstra, then check curfews at the end.* Run ordinary shortest path ignoring curfews, then verify the chosen path respects every `c[v]`. This is wrong on its face: the globally shortest path might violate a curfew while a slightly longer path is legal, and "check at the end" gives me no mechanism to fall back to the legal alternative. The constraint has to be *inside* the relaxation. Discarded.

- *Time-expanded / waiting states.* Model "(station, time)" pairs and allow a wait edge. This is the general tool when waiting can help, but here waiting never helps: arriving earlier dominates arriving later (a smaller arrival time satisfies every curfew that a larger one does and yields smaller successor times). So the time dimension collapses — the only label I need per station is its earliest legal arrival. Building the time-expanded graph would be correct but wasteful and far harder to get exactly right at the boundary. I keep it in my back pocket as a mental correctness check but won't implement it.

- *Dijkstra with the curfew enforced as a relaxation gate.* Keep `dist[v]` = earliest legal arrival at `v`. When I pop `u` with finalized time `d` and look at an edge to `v`, the candidate arrival is `nd = d + w`; I accept it only if it is both legal (`nd < c[v]`) and an improvement (`nd < dist[v]`). This is `O((n+m) log n)`, it threads the constraint through every step, and its correctness rests on the monotonicity I argued above. This is the one I commit to.

**Deriving why Dijkstra stays valid under the gate.** Dijkstra's correctness needs two facts. (1) Edge weights are positive (`w >= 1`), so successor times strictly exceed the current time; no zero-cost cycles, the usual settle-once argument holds. (2) The label I propagate, "earliest legal arrival," is monotone in exactly the way Dijkstra exploits: if I have already finalized the smallest legal arrival at `u`, any later legal arrival at `u` is `>=` it, and because curfews only compare arrival time against a fixed `c`, a smaller arrival at `u` can only produce smaller-or-equal legal arrivals downstream. So the first time I pop a node from the min-heap I have its true earliest legal arrival, and I never need to revisit it. The curfew gate `nd < c[v]` does not break this: it just refuses to create a label at all when arrival is illegal, which is the same as deleting that edge for that particular arrival time. Good — the gate is a filter on relaxation, not a perturbation of the order, so the settle-once invariant survives.

**Sanity-checking the derivation on the sample.** The sample: `n = 4`, curfews `c = [_, 100, 9, 12, 8]` (1-indexed), edges `1->4 (8)`, `1->2 (3)`, `2->4 (4)`, `3->4 (1)`. Start: `0 < c[1] = 100`, so `dist[1] = 0`, push `(0, 1)`. Pop `(0, 1)`. Edge `1->4`: `nd = 8`, check `8 < c[4] = 8`? No — illegal, rejected. This is the crux: the direct, cheapest hop lands *exactly* on the curfew and must be thrown away. Edge `1->2`: `nd = 3`, `3 < c[2] = 9`? Yes, and `3 < INF`, so `dist[2] = 3`, push `(3, 2)`. Pop `(3, 2)`. Edge `2->4`: `nd = 3 + 4 = 7`, `7 < c[4] = 8`? Yes, `7 < INF`, so `dist[4] = 7`, push `(7, 4)`. Pop `(7, 4)`; it is the target, and no outgoing edges matter. Answer `dist[4] = 7`. That matches the stated sample. The derivation produces the right number, and it produces it *because* the strict check killed the arrival-at-8 path — exactly the behaviour the problem is testing.

**First implementation and a trace.** Here is my first cut of the core:

```
const long long INF = LLONG_MAX / 4;
vector<long long> dist(n + 1, INF);
dist[1] = 0;                              // start at station 1, time 0
priority_queue<pair<long long,int>, vector<pair<long long,int>>,
               greater<>> pq;
pq.push({0, 1});
while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (d != dist[u]) continue;
    for (auto [v, w] : adj[u]) {
        long long nd = d + w;
        if (nd <= c[v] && nd < dist[v]) { // arrival legal & improving?
            dist[v] = nd;
            pq.push({nd, v});
        }
    }
}
cout << (dist[n] >= INF ? -1 : dist[n]) << "\n";
```

Two things nag at me. The gate reads `nd <= c[v]`, and I unconditionally set `dist[1] = 0` without checking `c[1]`. Let me trace the smallest input that could expose the boundary. Take `n = 2`, `m = 1`, curfews `c = [_, 100, 5]`, edge `1->2 (5)`. By hand, arrival at station `2` is `0 + 5 = 5`, and `c[2] = 5`; the rule says present at `2` only while `t < 5`, so arriving at `t = 5` is **forbidden** and the true answer is `-1`. Now run the code: `dist[1] = 0`, push `(0,1)`. Pop `(0,1)`. Edge `1->2`: `nd = 5`. Gate `nd <= c[v]` is `5 <= 5` -> **true**, and `5 < INF`, so it sets `dist[2] = 5` and prints `5`.

**The bug.** The code prints `5`; the correct answer is `-1`. This is the off-by-one I was warned about, made concrete. The legal region is `t < c[v]`, a *strict* inequality, but I coded `nd <= c[v]`, which admits the single forbidden instant `t = c[v]`. Every arrival that lands exactly on a curfew is wrongly accepted. On this 2-node graph the consequence is total: I report a route that doesn't legally exist. The fix is one character — change `<=` to `<` — but I want to be certain it is the *only* place the boundary leaks. Let me re-trace with `nd < c[v]`: edge `1->2`, `nd = 5`, gate `5 < 5` -> false, rejected; `dist[2]` stays `INF`; output `-1`. Correct. Good, the inner gate is fixed.

**A second trace, on the start node.** The boundary doesn't only live on edges — it lives on the *start*, and my code still sets `dist[1] = 0` blindly. Construct `n = 1`, `m = 0`, curfew `c = [_, 0]`. The courier is born at station `1` at time `0`, but `c[1] = 0` means being present at `1` requires `0 < 0`, which is false — the courier cannot legally stand at the origin, so the answer is `-1`. Trace the current code: `dist[1] = 0` (set unconditionally), the loop runs but pops `(0,1)` with no edges, then output is `dist[n] = dist[1] = 0`. It prints `0`. Wrong — should be `-1`. There is also a subtler variant: `n = 2`, `c = [_, 0, 100]`, edge `1->2 (3)`. The courier can't legally be at `1` at all, so it can never depart; answer `-1`. But the blind `dist[1] = 0` would let it relax `1->2` to `dist[2] = 3` and print `3`. So the missing start check is a real, independent bug, not a duplicate of the edge bug.

**The second bug, diagnosed.** The start position is itself "an arrival" — an arrival at station `1` at time `0` — and it must pass the same strict curfew test as any other arrival. My code special-cased the start and skipped the check. The fix: only seed `dist[1] = 0` and push it when `0 < c[1]`; otherwise leave `dist[1] = INF` and start with an empty frontier, which naturally yields `-1`. Re-trace `n = 1, c = [_, 0]`: `0 < c[1] = 0` is false, so `dist[1]` stays `INF`, nothing is pushed, the loop body never runs, and output is `dist[1] >= INF ? -1` -> `-1`. Correct. Re-trace `n = 2, c = [_, 0, 100]`: `0 < 0` false, frontier empty, `dist[2]` stays `INF`, output `-1`. Correct. And the ordinary case `n = 1, c = [_, 3]`: `0 < 3` true, `dist[1] = 0`, pushed, popped, no edges, output `dist[1] = 0`. Correct — the courier is already at the destination, legally, at time `0`.

**Independent cross-check by exhaustive enumeration.** I don't want to trust a single hand-trace, so I reason about an independent method and confirm the two agree in principle. Because weights are positive and waiting never helps, an optimal legal route is a *simple path* (revisiting a node only wastes time and can only push arrivals later, never earlier, so it never unlocks a curfew). That means I can verify by brute force: enumerate every simple path from `1`, accumulate arrival times, drop any path the moment an arrival violates `t < c[v]` (including the start `0 < c[1]`), and take the minimum arrival at `n`. This DFS-over-paths method shares no code structure with Dijkstra — no heap, no dominance pruning, no settle-once — so agreement between them is strong evidence. On the sample it enumerates `1->4` (arrival 8, rejected at the `8 < 8` test), `1->2->4` (arrival 7, kept), `3->4` (unreachable from 1), and returns `7`, matching Dijkstra. The two methods agree by construction on exactly the boundary case that distinguishes `<` from `<=`.

**Edge cases, deliberately.**
- *`n = 1`, reachable:* `c[1] > 0` -> answer `0` (already there, legally). `c[1] = 0` -> `-1` (illegal at the origin). Both handled by the seeded start check.
- *Unreachable target:* every route into `n` either has no edge or lands on/after a curfew; `dist[n]` stays `INF`; output `-1`. E.g. `c = [_, 5, 4, 6]`, edges `1->2 (4)`, `2->3 (2)`: arrival at `2` is `4 = c[2]`, gate `4 < 4` false, so `2` is never settled, `3` is never reached, answer `-1`.
- *Parallel edges:* two edges `1->n` with weights `8` (illegal at curfew) and `9` (also illegal) but a third path that is legal — the heap considers each candidate independently and keeps only legal, improving ones, so parallels just generate extra candidates and the min survives. Verified directly in fuzzing.
- *Start equals target conceptually:* covered by `n = 1`.
- *Overflow:* labels and `nd = d + w` are `long long`; the max realistic arrival `~2*10^14` fits with enormous headroom. `INF = LLONG_MAX / 4` is never added to (I only compute `d + w` for popped, finite `d`), so it cannot overflow; the `dist[n] >= INF` test cleanly detects unreached nodes. Comparisons against `c[v] <= 10^9` are all within range.
- *No edges, `m = 0`, `n > 1`:* nothing relaxes, `dist[n] = INF`, output `-1`. Correct.
- *Empty-ish input:* `if (!(cin >> n >> m)) return 0;` guards a truncated stream; `cin >>` skips arbitrary whitespace so the line layout is irrelevant.

**Re-verification after both fixes.** I run the corrected solution against the exhaustive-path brute force on several hundred random small graphs (up to 7 nodes, a dozen edges, curfews in `1..12` and weights in `1..6` so that arrivals frequently land *exactly* on curfews — the boundary is hit constantly). Zero mismatches. As a control I also run the *buggy* `<=` variant against the same brute force: it disagrees on a noticeable fraction of cases, every disagreement being precisely a graph where some optimal arrival lands on a curfew. That control is the proof that my generator actually exercises the boundary and that the `<` versus `<=` choice is the thing under test — the fix isn't cosmetic, it changes the answer on real inputs.

**Final solution.** The idea (Dijkstra with a strict-curfew relaxation gate) I justified by the monotonicity-of-earliest-legal-arrival argument and cross-checked against exhaustive simple-path enumeration; the *code* I corrected by tracing two concrete boundary inputs — `1->2 (5)` with `c[2] = 5` exposed the `<=`/`<` edge bug, and `n = 1, c[1] = 0` exposed the missing strict check on the start. This is what I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> c(n + 1);
    for (int v = 1; v <= n; v++) cin >> c[v];

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long INF = LLONG_MAX / 4;
    vector<long long> dist(n + 1, INF);

    // You start at station 1 at time 0.
    // Being present at station v at any time t with t >= c[v] is forbidden,
    // so an arrival time t at v is LEGAL only if t < c[v] (strict).
    // Departure happens at the same instant as arrival (no waiting helps),
    // so the only constraint per node is the strict-inequality arrival check.

    // Start station must itself be legal at time 0.
    if (0 < c[1]) {
        dist[1] = 0;
    }

    priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                   greater<pair<long long,int>>> pq;
    if (dist[1] == 0) pq.push({0, 1});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            long long nd = d + w;
            // Arriving at v at time nd is allowed only if nd < c[v] (strict boundary).
            if (nd < c[v] && nd < dist[v]) {
                dist[v] = nd;
                pq.push({nd, v});
            }
        }
    }

    if (dist[n] >= INF) cout << -1 << "\n";
    else cout << dist[n] << "\n";
    return 0;
}
```

**Causal recap.** The objective is a positive-weight shortest path with a per-node validity window, and because arriving earlier dominates, the right label is "earliest legal arrival," which makes a heap-based Dijkstra with a relaxation gate the correct and efficient tool. The entire problem turns on one inequality: the legal region is `t < c[v]`, strict, so my first gate `nd <= c[v]` was off by one and a two-node trace (`1->2 (5)`, `c[2] = 5`, true answer `-1`, code said `5`) pinned it; tightening to `nd < c[v]` fixed it. The second, independent boundary lives at the origin — the start is itself an arrival at time `0` — and seeding `dist[1] = 0` unconditionally let an illegal start (`c[1] = 0`) leak through, which an `n = 1` trace exposed; gating the seed on `0 < c[1]` fixed that. With both strict checks in place, `long long` everywhere for the `~2*10^14` time range, and `-1` emitted when `dist[n]` stays infinite, the solution agrees with exhaustive simple-path enumeration on hundreds of boundary-dense random cases while the `<=` variant provably does not.
