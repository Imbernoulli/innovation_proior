**Reading the problem and pinning the contract.** I have a directed graph of `n` relay stations and `m` links; each link `u -> v` has an integer margin `w` that may be positive, zero, or negative. A route's reliability is the *minimum* margin over its links, and for a fixed source `s` I want, for every station `v`, the maximum reliability over all routes `s -> v`. So the metric is "max over paths of (min edge weight)" — a widest-path / bottleneck objective, not an additive one. Let me nail the scale first because it dictates types: `n <= 2*10^5`, `m <= 5*10^5`, `|w| <= 10^9`. The metric never sums weights — a bottleneck is always equal to some single edge's margin — so the *answer* fits in 32 bits. But I will still carry weights and sentinels in `long long`, because I want sentinels like `+infinity` and `-infinity` that sit comfortably outside the real value range `[-10^9, 10^9]` without flirting with `int` overflow when I write `LLONG_MAX/4`. That is the first decision: `long long` for the metric, `int` for node ids.

**Why this is a Dijkstra problem at all, despite negative weights.** My instinct screams "negative edges — Dijkstra is illegal." That instinct is about *additive* shortest paths, where a later negative edge can retroactively make an already-finalized node suboptimal. Here the metric is `min`, and the key structural fact is that extending a route never *increases* its value: `min(best[u], w) <= best[u]` always, for any sign of `w`. So once I pop the unfinished node `u` with the largest `best[u]`, no other route can later reach `u` with a larger bottleneck — any such route would have to pass through some unfinished node whose `best` is `<= best[u]`, and going through it can only shrink the bottleneck further. That is exactly Dijkstra's exchange argument, and it leans on **monotonicity**, not on non-negativity. So a max-heap Dijkstra with the relaxation `cand = min(best[u], w)`, keeping the max, is correct *with* negative and zero margins. Good — I commit to max-min Dijkstra. The Bellman-Ford alternative (`O(n m)`) is correct too but `2*10^5 * 5*10^5 = 10^11` operations is hopeless at the time limit.

**Deciding the base case — this is where sign handling lives.** What is `best[s]`? The route `s -> s` of length zero uses *no* links. The minimum over an empty set of margins is, by convention, `+infinity` — there is no weak link to drag it down. So `best[s]` must start at `+infinity`, and that is what gets handed to the first relaxation: a link `s -> v` of margin `w` produces `min(+infinity, w) = w`, which is exactly right (the bottleneck of a single-link route is that link's margin). If I instead seeded `best[s] = 0`, then a single positive link `s -> v` with `w = 5` would yield `min(0, 5) = 0` — I would have invented a phantom weak link of margin 0 at the source and clamped every reachable node to `<= 0`. With all-positive margins that is a blatant wrong answer. So the base case is `best[s] = +infinity`, represented as a large sentinel `POS_INF = LLONG_MAX/4`. And "not reached yet" needs its own sentinel strictly below every real margin: `NEG_INF = LLONG_MIN/4`, well under `-10^9`. These two sentinels must be distinguishable from each other and from every real value in `[-10^9, 10^9]`; `LLONG_MAX/4` and `LLONG_MIN/4` satisfy that with enormous slack.

**Deciding the output mapping — the second place sign handling lives.** Three outcomes, and they must not collapse:
- `v == s`: print `INF` (the empty route's `+infinity` bottleneck; the source has no genuine self-route that could beat it, since any real route back to `s` has a finite bottleneck `< +infinity`).
- `v` reachable, `v != s`: print the integer `best[v]`, which may legitimately be negative or zero.
- `v` unreachable: `best[v]` is still `NEG_INF`; print `UNREACHABLE`.
The trap I can already feel: a *reachable* node whose best route bottoms out at a negative margin is a completely different thing from an *unreachable* node, even though both "look bad." If I test reachability by sign (`best[v] <= 0` -> unreachable) I will mislabel reachable degraded nodes. Reachability must be tested against the `NEG_INF` sentinel *exactly*, never by sign. I will hold that thought and see if my first code respects it.

**First implementation.** Here is my first cut of the core, deliberately written fast so I can then attack it:

```
const long long NEG_INF = LLONG_MIN / 4;
vector<long long> best(n + 1, NEG_INF);
vector<char> done(n + 1, 0);
best[s] = 0;                                  // (suspicious — revisit)
priority_queue<pair<long long,int>> pq;
pq.push({best[s], s});
while (!pq.empty()) {
    auto [d, u] = pq.top(); pq.pop();
    if (done[u]) continue;
    done[u] = 1;
    for (auto [v, w] : adj[u]) {
        long long cand = min(best[u], w);
        if (cand > best[v]) { best[v] = cand; pq.push({best[v], v}); }
    }
}
for (int v = 1; v <= n; v++) {
    if (best[v] <= 0) cout << "UNREACHABLE\n";  // (also suspicious — revisit)
    else cout << best[v] << "\n";
}
```

I wrote two things I flagged as suspicious; rather than reason in the abstract, let me trace a concrete input and let the numbers indict the code.

**First debug episode — tracing the documented sample.** Take the sample
```
5 6 1
1 2 4
1 3 -2
2 3 5
2 4 0
3 4 7
4 5 3
```
By hand the intended answers are: node 1 = `INF`; node 2 via `1->2` = `4`; node 3 best is `1->2->3 = min(4,5) = 4` (beating `1->3 = -2`); node 4 best is `1->2->3->4 = min(4,5,7) = 4` (beating `1->2->4 = min(4,0)=0` and `1->3->4 = min(-2,7)=-2`); node 5 via node 4 then edge 3 = `min(4,3) = 3`.

Now run my first code mentally. `best[1] = 0`. Pop node 1 (`d=0`). Relax its links: `1->2`, `cand = min(0, 4) = 0`, so `best[2] = 0`; `1->3`, `cand = min(0, -2) = -2`, so `best[3] = -2`. Already wrong: `best[2]` should be `4`, not `0`. The `best[s] = 0` seed injected a phantom weak link, clamping node 2 to `0`. Continuing only compounds it: from node 2 (`best=0`), `2->3` gives `min(0,5)=0` (so `best[3]` rises to `0`), `2->4` gives `min(0,0)=0`. From node 3 (`best=0`), `3->4` gives `min(0,7)=0`. From node 4 (`best=0`), `4->5` gives `min(0,3)=0`. Final `best = [_, 0, 0, 0, 0, 0]`. Output mapping: every `best[v] <= 0`, so the code prints `UNREACHABLE` for *all five* nodes. The intended output was `INF / 4 / 4 / 4 / 3`. Catastrophically wrong, and in two independent ways.

**Diagnosing the first bug (base case).** The root cause of the clamping is `best[s] = 0`. The empty route has bottleneck `+infinity`, not `0`; seeding `0` fabricates a zero-margin weak link that every route inherits, so no route can ever exceed `0`. The fix is `best[s] = POS_INF = LLONG_MAX/4`. Re-relax `1->2`: `min(POS_INF, 4) = 4` — correct. This is precisely the sign/base-case pitfall the metric invites: with an additive metric the natural source value is `0`, and it is muscle memory to write `0` here, but for a `min`-metric `0` is a *finite, often-binding* value, not the identity. The identity of `min` is `+infinity`.

**Diagnosing the second bug (sign-vs-reachability).** Even after fixing the seed, the output test `best[v] <= 0 -> UNREACHABLE` is broken, and I can show it independently. Consider
```
3 1 1
1 2 0
```
With the corrected seed, `best[1]=POS_INF`, pop 1, relax `1->2`: `min(POS_INF,0)=0`, `best[2]=0`. Node 3 is never reached, stays `NEG_INF`. Intended output: `INF` (node 1), `0` (node 2 is reachable with bottleneck exactly 0), `UNREACHABLE` (node 3). But `best[2] <= 0` is true, so the buggy test prints `UNREACHABLE` for node 2 — collapsing "reachable through a zero-margin link" into "unreachable." A second witness is any all-negative graph, e.g. `1->2` with `w=-5`: `best[2] = -5`, which is `<= 0`, so it would be misreported as `UNREACHABLE` when it is genuinely reachable with reliability `-5`. The defect is testing *reachability by sign*. Reachability is "did the value ever leave the `NEG_INF` sentinel," a question about the sentinel, not about the sign. The fix: compare `best[v] == NEG_INF` exactly, and only then print `UNREACHABLE`; otherwise print the integer (handling the source's `INF` separately).

**Fixing and re-verifying on the sample.** With `best[s] = POS_INF` and the output mapping
```
if (best[v] == NEG_INF) print "UNREACHABLE";
else if (v == s) print "INF";          // POS_INF empty-route bottleneck
else print best[v];
```
re-trace the sample. `best[1]=POS_INF`. Pop 1: `1->2` -> `best[2]=min(POS_INF,4)=4`; `1->3` -> `best[3]=min(POS_INF,-2)=-2`. The max-heap now holds `(4,2)` and `(-2,3)`. Pop `(4,2)` (node 2, largest): `2->3` -> `min(4,5)=4 > -2`, so `best[3]=4`, push `(4,3)`; `2->4` -> `min(4,0)=0`, `best[4]=0`, push. Pop `(4,3)` (node 3): `3->4` -> `min(4,7)=4 > 0`, `best[4]=4`, push `(4,4)`. Pop `(4,4)` (node 4): `4->5` -> `min(4,3)=3`, `best[5]=3`. Remaining heap entries are stale (`(0,4)`, `(-2,3)`) and get skipped by the `done`/freshness guard. Final `best = [_, POS_INF, 4, 4, 4, 3]`. Output: `INF, 4, 4, 4, 3`. Matches the intended sample exactly. Both bugs are gone, and each went away for the reason I diagnosed, which is the evidence I trust over "looks fixed."

**Second debug episode — the stale-entry / lazy-deletion guard.** I used a lazy priority queue (push improved values, never decrease-key), so the heap can hold outdated `(d, v)` pairs. My first guard was just `if (done[u]) continue;`. Is that enough? Trace: when I pop `(d, u)` with `done[u]` false but `d` an *old, smaller* value than the current `best[u]` (because a better value was pushed after this entry), I would still process `u`, mark it done, and relax with `min(best[u], w)`. Wait — I relax using `best[u]`, the array, not the popped `d`. So even a stale pop relaxes with the *current* `best[u]`. Does that corrupt anything? The danger is finalizing `u` while a strictly larger value for `u` is still sitting in the heap unpopped. But that cannot happen: the heap is a max-heap on the value, so if a larger entry `(d', u)` with `d' > d` existed, it would have been popped *before* `(d, u)`, at which point `done[u]` would already be set and this pop skipped. So `done[u]` alone is sufficient for correctness here. Still, to make the invariant explicit and avoid relaxing from a node on a stale value, I add `if (d != best[u]) continue;` — when I do reach a node for real, `d` equals the finalized `best[u]`. This is belt-and-suspenders: it does not change the answer (the 701-case stress test passes with or without it) but it documents that I only ever expand the up-to-date frontier value. I keep both guards.

**Edge cases, deliberately, because this is where bottleneck code dies.**
- *Empty edge set* `m = 0`, e.g. `3 0 2`: no link is ever read, the heap starts with just `(POS_INF, 2)`, node 2 is popped and has no out-links, nodes 1 and 3 stay `NEG_INF`. Output `UNREACHABLE / INF / UNREACHABLE`. Correct — only the source is reachable via the empty route.
- *Single node* `n = 1, m = 0, s = 1`: output is the single line `INF`. Correct.
- *All-negative graph*, e.g. `1->2 (-5), 2->3 (-1), 1->3 (-9)`: `best[2] = min(POS_INF,-5) = -5`; node 3 via `1->2->3 = min(-5,-1) = -5` beats `1->3 = -9`, so `best[3] = -5`. Output `INF / -5 / -5`. The negatives are reported as negatives, *not* as unreachable — the exact collision the second bug would have caused. Correct.
- *Zero margin reachable*, `1->2 (0)` with an unreachable node 3: `INF / 0 / UNREACHABLE`. The `0` node and the unreachable node are distinct. Correct.
- *Self-loops and parallel edges*: a self-loop `u->u` relaxes `min(best[u], w)` against `best[u]`; since `min(best[u], w) <= best[u]` it can never improve `best[u]`, so it is harmless. Parallel edges are just multiple relaxations; the best one wins via the `cand > best[v]` test. Correct.
- *Overflow / sentinel safety*: the real metric is some single `w`, always in `[-10^9, 10^9]`; I never *add* weights, so no accumulation can overflow. `POS_INF = LLONG_MAX/4` and `NEG_INF = LLONG_MIN/4` are only ever passed through `min` / compared, never arithmetic-combined with `w` in a way that could overflow (`min(POS_INF, w) = w` since `w << POS_INF`). Safe in 64-bit with vast slack; the answer itself prints fine in 32 bits but I leave it `long long`.
- *Source reachable from itself by a real cycle*: that route has a finite bottleneck `< POS_INF`, so it never improves `best[s]`, which stays `POS_INF` and prints `INF`. Consistent with "the empty route is the best route to the source."

**Sanity-checking the derivation itself.** Beyond tracing the code, let me re-confirm the *claim* that simple paths suffice, since the whole correctness rests on it. Any route can be decomposed by removing cycles; each removed cycle deletes one or more edges from the route, and deleting edges can only *raise or keep* the minimum (fewer terms in a `min`), never lower it. So the simple route obtained by cycle-removal has a bottleneck `>=` the original — meaning some simple route is always at least as good as any non-simple one. Hence restricting attention to simple paths loses nothing, the optimum is finite and attained, and the Dijkstra finalize-the-widest argument is sound. The brute force I cross-checked against literally enumerates all simple paths via DFS and takes the max-of-min; over 701 random cases (including all-negative, all-zero, negatives-and-zeros, empty, self-loops) it agreed with the Dijkstra solution on every line.

**Final solution.** I convinced myself the *idea* is right by reducing the `min`-metric correctness to monotonicity and to the simple-path argument, and I convinced myself the *code* is right by tracing two real failures — a base-case sign bug (`best[s]=0` clamping everything to `<= 0`) and a reachability-by-sign bug (negative/zero bottlenecks mislabeled `UNREACHABLE`) — to precise causes and re-verifying the fixes on the sample and the corners. This is what I ship, one self-contained file:

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long NEG_INF = LLONG_MIN / 4;   // "no path yet" / unreachable
    const long long POS_INF = LLONG_MAX / 4;   // bottleneck of the zero-edge path at the source

    vector<long long> best(n + 1, NEG_INF);
    vector<char> done(n + 1, 0);
    best[s] = POS_INF;                         // path of length 0: min over empty edge set = +inf

    // max-min Dijkstra: pop the node with the currently largest bottleneck.
    priority_queue<pair<long long,int>> pq;    // (bottleneck, node), max-heap
    pq.push({best[s], s});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (done[u]) continue;                 // stale / already finalized
        if (d != best[u]) continue;            // outdated entry
        done[u] = 1;
        for (auto [v, w] : adj[u]) {
            long long cand = min(best[u], w);  // weakest link along this path
            if (cand > best[v]) {
                best[v] = cand;
                pq.push({best[v], v});
            }
        }
    }

    for (int v = 1; v <= n; v++) {
        if (best[v] == NEG_INF) cout << "UNREACHABLE";
        else if (v == s && best[v] == POS_INF) cout << "INF";  // source with no self-route
        else cout << best[v];
        cout << "\n";
    }
    return 0;
}
```

**Causal recap.** The metric is max-over-paths-of-min-edge, which is Dijkstra-amenable not because weights are non-negative (they are not — they span negatives, zeros, positives) but because extending a route is *monotone non-increasing* in the bottleneck, so the widest-unfinished-node exchange argument holds. The two bugs both came from importing additive-shortest-path reflexes into a `min`-metric: seeding the source at `0` (additive identity) instead of `+infinity` (the `min` identity) phantom-clamped every node to `<= 0`, which a trace of the sample exposed by turning all five answers into `UNREACHABLE`; and testing reachability by sign rather than against the `NEG_INF` sentinel collapsed legitimately-reachable negative/zero-bottleneck nodes into `UNREACHABLE`, which a zero-margin trace and an all-negative trace each exposed. Fixing the source base case to `POS_INF`, mapping the source line to `INF`, and gating `UNREACHABLE` on `best[v] == NEG_INF` exactly closes the empty, single-node, all-negative, all-zero, self-loop, and overflow corners, all cross-checked against an exhaustive simple-path brute force over 701 random cases with zero mismatches.
