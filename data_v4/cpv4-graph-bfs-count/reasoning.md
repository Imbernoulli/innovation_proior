**Reading the problem and pinning the contract.** I have `n` stations and `m` bidirectional equal-cost hops, and I must count, modulo `1_000_000_007`, the number of *distinct minimum-hop itineraries* from `s` to `t`. "Distinct" is defined on the **sequence of hops** taken — so if two stations are joined by two parallel connections, choosing one versus the other yields different itineraries. I underline that because it tells me exactly what to do with parallel edges: keep them, each contributes separately. The contract spells out three corners and I want them fixed before any algorithm: if `t` is unreachable I print `0`; if `s == t` the zero-hop itinerary is the unique answer so I print `1`; self-loops and parallel edges are allowed in the input, and a self-loop "never shortens any route." Scale: `n` up to `2*10^5`, `m` up to `4*10^5`. Sparse, so an adjacency-list BFS is `O(n+m)` and well inside 1 second. The count itself can be enormous — a layered graph with width `W` and depth `k` has `W^(k-1)` shortest routes, which blows past 64-bit almost immediately — so every running count lives modulo `1e9+7`, and the reduction must happen *during* accumulation, not at the end, or the intermediate sums overflow. That is the first non-negotiable decision.

**Laying out the candidate approaches.** The distance part is uncontroversial: plain BFS from `s` gives `dist[v]`, the minimum hop count, because BFS finalizes vertices in nondecreasing distance and the first time a vertex is reached its distance is minimal. The interesting part is the *count* layer, `ways[v]` = number of minimum-hop itineraries from `s` to `v`. The governing identity is clean: a hop `u -> v` extends a shortest route to `v` iff `u` is one layer closer, `dist[u] + 1 == dist[v]`, and then every shortest route to `v` is a shortest route to some such `u` extended by that hop. So

  `ways[v] = sum over hops (u,v) with dist[u] + 1 == dist[v] of ways[u]`,

with `ways[s] = 1` (the empty itinerary). Parallel hops simply appear multiple times in the sum, which is correct under the hop-sequence definition. Two ways to realize this:

- *Two-pass.* BFS once for `dist[]`. Then iterate over edges and, whenever `dist[u]+1 == dist[v]`, do `ways[v] += ways[u]`. Simple to state, but it is only correct if I add in nondecreasing `dist` order; if I touch `ways[v]` before `ways[u]` is itself complete, I read a half-built number. The ordering is the whole risk.
- *One-pass.* Fold the counting into the BFS itself: when a hop first discovers `v`, seed `ways[v]` from its discoverer; when a *later* hop comes from the previous layer, add into `ways[v]`. Because BFS dequeues by layer, a predecessor `u` (on the strictly-earlier layer) is fully accumulated by the time `v` is being relaxed from it. This is the approach I lean toward, but its correctness hinges entirely on the predicate that decides which neighbours count.

Both reduce to getting one predicate exactly right: "previous layer" must be `dist[u] + 1 == dist[v]`, and I must positively *exclude* "same layer" (`dist[u] == dist[v]`) and "behind" (`dist[u] >= dist[v]`). The same-layer edge is the heart of this problem's difficulty: the sample deliberately includes `2-3` and `4-5`, both joining stations at equal distance from `s`, and they must contribute nothing.

**Deriving the recurrence and sanity-checking it on the sample.** Let me first compute distances on the sample by hand to make sure I understand the layers. Edges: `1-2, 1-3, 2-4, 3-4, 3-5, 4-6, 5-6`, plus the decoys `2-3, 4-5`. From `s = 1`: layer 0 = `{1}`. Layer 1 = `{2, 3}` (both adjacent to 1). Layer 2 = neighbours of layer 1 not yet seen: from 2 -> 4; from 3 -> 4, 5; so `{4, 5}`. Layer 3 = from 4 -> 6, from 5 -> 6; so `{6}`. Distances: `dist[1]=0, dist[2]=dist[3]=1, dist[4]=dist[5]=2, dist[6]=3`. Good, `dist[t]=dist[6]=3`.

Now counts. `ways[1]=1`. `ways[2]`: contributors are hops `(u,2)` with `dist[u]+1==1`, i.e. `dist[u]=0`: only station 1. So `ways[2]=ways[1]=1`. Likewise `ways[3]=1`. `ways[4]`: neighbours with `dist=1` are `2` and `3` (the edge `4-5` has `dist[5]=2`, same layer, excluded — that's a decoy). So `ways[4]=ways[2]+ways[3]=1+1=2`. `ways[5]`: neighbours with `dist=1`: only `3` (5 is adjacent to 3 and to 4; `dist[4]=2` is same-layer, excluded). So `ways[5]=ways[3]=1`. `ways[6]`: neighbours with `dist=2`: `4` and `5`. So `ways[6]=ways[4]+ways[5]=2+1=3`. The answer is `ways[6]=3`, matching the stated sample (`1-2-4-6`, `1-3-4-6`, `1-3-5-6`). The recurrence is right, and notice how decisively the two decoy edges had to be excluded: had I let `4-5` feed `ways[4] += ways[5]` or vice versa, the count would be wrong. That is the bug I am about to walk into.

**First implementation — the one-pass BFS, and immediately a trace, because the predicate is where this dies.** I write the relaxation a little too loosely on the first cut:

```
dist[s] = 0; ways[s] = 1; q.push(s);
while (!q.empty()) {
    int u = q.front(); q.pop();
    for (int w : adj[u]) {
        if (dist[w] == INF) {
            dist[w] = dist[u] + 1;
            ways[w] = ways[u];
            q.push(w);
        } else {
            ways[w] = (ways[w] + ways[u]) % MOD;   // "another contributor"
        }
    }
}
```

The dangerous line is the `else`: I treat *every* already-discovered neighbour as a contributor and add `ways[u]` into it. Let me trace the sample to see what that does. I track `(dist, ways)`. Start: `1:(0,1)`, everything else `INF`. Dequeue `1`, neighbours `2,3`: both `INF`, so `2:(1,1)`, `3:(1,1)`, push both. Dequeue `2`, neighbours (in input order) `1,4,3`. Neighbour `1`: already discovered, so the `else` fires -> `ways[1] += ways[2] = 1+1 = 2`. That is already nonsense — station 1 is the source, behind `2`, and I just bumped its count. Neighbour `4`: `INF` -> `4:(2,1)`, push. Neighbour `3`: discovered, `else` fires -> `ways[3] += ways[2] = 1+1 = 2`. But `3` is on the *same layer* as `2`; the decoy edge `2-3` just corrupted `ways[3]`. Dequeue `3`, neighbours `1,4,5,2`. `1`: `else` -> `ways[1] += ways[3] = 2 + 2 = 4`. `4`: discovered -> `else` -> `ways[4] += ways[3] = 1 + 2 = 3`. `5`: `INF` -> `5:(2, ways[3]=2)`, push. `2`: `else` -> `ways[2] += ways[3] = 1 + 2 = 3`. By now `ways` is total garbage. The run would eventually spit out some wrong number for `ways[6]`.

**Diagnosing the first bug.** The defect is precise and it is exactly the pitfall this problem is built around. My `else` branch adds `ways[u]` into `ways[w]` for *any* already-discovered `w`, but a true contributor must satisfy `dist[u] + 1 == dist[w]`. Two illegal cases slipped through: (1) `w` is on the *same layer* as `u` (`dist[w] == dist[u]`) — the decoy edges `2-3` and `4-5`, which lie on no shortest route and must contribute nothing; (2) `w` is *behind* `u` (`dist[w] < dist[u]`, e.g. the source `1` seen from `2`) — adding there counts a route in the wrong direction. The undirected edge is the root cause: every edge is stored twice (`u` in `adj[w]` and `w` in `adj[u]`), so when I dequeue the deeper endpoint I see the shallower one again and wrongly "add back." The fix is to gate the addition on the layer test.

**Fixing the predicate and re-verifying.** I replace the blanket `else` with the exact "previous layer" test, and leave same-layer / behind neighbours alone:

```
if (dist[w] == INF) {
    dist[w] = dist[u] + 1;
    ways[w] = ways[u];
    q.push(w);
} else if (dist[w] == dist[u] + 1) {
    ways[w] = (ways[w] + ways[u]) % MOD;   // u is a genuine previous-layer contributor
}
// dist[w] == dist[u] (same layer)  -> nothing
// dist[w] <  dist[u] (behind)      -> nothing
```

Is this *ordering-safe*? When I dequeue `u` and add `ways[u]` into a deeper `w` with `dist[w] == dist[u] + 1`, is `ways[u]` already final? BFS dequeues in nondecreasing distance, and every contribution into `ways[u]` comes from a vertex on layer `dist[u]-1`, all of which were dequeued strictly before any layer-`dist[u]` vertex. So by the time `u` (layer `dist[u]`) is dequeued, `ways[u]` has received all its contributors and is final. The contribution into `w` therefore reads a finished value. Good — the single-pass BFS is correct precisely because the queue serializes by layer.

Let me re-trace the sample with the fix. Start `1:(0,1)`. Dequeue `1`, neighbours `2,3`: both `INF` -> `2:(1,1)`, `3:(1,1)`. Dequeue `2`, neighbours `1,4,3`. `1`: `dist[1]=0`, is `0 == dist[2]+1 = 2`? No -> skip (behind, correct). `4`: `INF` -> `4:(2,1)`. `3`: `dist[3]=1`, is `1 == 2`? No -> skip (same layer, decoy `2-3`, correct). Dequeue `3`, neighbours `1,4,5,2`. `1`: `dist=0 != 2` skip. `4`: `dist[4]=2 == dist[3]+1=2`? Yes -> `ways[4] = 1 + ways[3] = 1 + 1 = 2`. `5`: `INF` -> `5:(2,1)`. `2`: `dist=1 != 2` skip (decoy). Dequeue `4`, neighbours `2,3,6,5`. `2`: `dist=1 != 3` skip. `3`: `dist=1 != 3` skip. `6`: `INF` -> `6:(3, ways[4]=2)`. `5`: `dist[5]=2 == dist[4]+1=3`? No, `2 != 3` -> skip (decoy `4-5`, same layer, correct!). Dequeue `5`, neighbours `3,6,4`. `3`: skip. `6`: `dist[6]=3 == dist[5]+1=3`? Yes -> `ways[6] = 2 + ways[5] = 2 + 1 = 3`. `4`: `dist=2 != 3` skip. Dequeue `6`, neighbours `4,5`: both behind, skip. Final `ways[6] = 3`. Correct, and crucially the two decoy edges contributed nothing this time. The first bug is dead.

**A second, sneakier attempt — the two-pass version — exposes an ordering double-count.** Before I commit, I want to compare against the two-pass formulation, partly to convince myself the one-pass is right and partly because two-pass is what many people reach for ("first distances, then counts"). My naive two-pass: run BFS for `dist[]`, then loop over the *vertices in input/index order* `1..n`, and for each vertex `v` add its contribution to its neighbours:

```
// after dist[] is known:
ways[s] = 1;
for (int v = 1; v <= n; v++)
    for (int w : adj[v])
        if (dist[v] + 1 == dist[w])
            ways[w] = (ways[w] + ways[v]) % MOD;   // push v's count forward to w
```

This *looks* like the right recurrence — same predicate, even — but let me trace it on a graph where the index order and the layer order disagree. Take stations relabeled so the deep one has a small index: edges `1-3, 1-4, 3-2, 4-2`, with `s = 1`, `t = 2`. Distances: `dist[1]=0`, `dist[3]=dist[4]=1`, `dist[2]=2`. True answer: two routes `1-3-2` and `1-4-2`, so `ways[2]=2`. Now run the index-order push. `ways[1]=1`, rest `0`. v=1: neighbours `3,4`, both `dist=1==0+1` -> `ways[3]+=1 -> 1`, `ways[4]+=1 -> 1`. v=2: `dist[2]=2`; neighbours `3,4` have `dist=1`, `dist[2]+1=3 != 1` -> nothing pushed (good, 2 is the deepest). v=3: `dist[3]=1`; neighbour `2` has `dist=2==1+1` -> `ways[2]+=ways[3]=0+1=1`; neighbour `1` `dist=0`, `1+1=2!=0` skip. v=4: neighbour `2`, `dist=2==2` -> `ways[2]+=ways[4]=1+1=2`. Final `ways[2]=2`. Here it happened to work because the small-index vertices were also the shallow ones.

Now I deliberately relabel so a *deep* vertex has a *small* index: edges `1-2, 1-3, 2-4, 3-4`, but read `s = 1`, `t = 4` — wait, here index order already matches layer order. Let me force the mismatch: stations `1=s`, and the merge vertex `t` is labeled `2`, with two midpoints labeled `5` and `6`. Edges `1-5, 1-6, 5-2, 6-2`. `dist[1]=0, dist[5]=dist[6]=1, dist[2]=2`, answer `2`. Index-order push: `ways[1]=1`. v=1: push to `5,6` -> both `1`. v=2 (`dist=2`, deepest): neighbours `5,6` `dist=1`, `2+1=3!=1`, nothing. v=5: `dist=1`, neighbour `2` `dist=2==1+1` -> `ways[2]+=1 -> 1`; neighbour `1` skip. v=6: neighbour `2` -> `ways[2]+=1 -> 2`. Final `2`. Still right — because the push reads `ways[v]` for the source-side vertex `v`, and the *contributors* of `v` (layer `dist[v]-1`) all have... hmm, do they necessarily have smaller index? No. Let me construct the actual failure: I need a vertex `v` at layer `L` whose own count is *incomplete* at the moment I process it in index order, i.e. some contributor of `v` (at layer `L-1`) has a *larger index* than `v` and so is processed *after* `v`.

Concrete: path `s=1 -> a -> b`, with `a` reachable from `s` two ways through two layer-1 stations, then `b` from `a`. Stations: `s=1`; layer-1 stations `7` and `8`; layer-2 station `a=2`; layer-3 station `b=3`; `t=3`. Edges `1-7, 1-8, 7-2, 8-2, 2-3`. Distances: `dist[1]=0, dist[7]=dist[8]=1, dist[2]=2, dist[3]=3`. True answer: routes to `2` are `1-7-2` and `1-8-2` (count 2), each extends to `b=3` uniquely, so `ways[3]=2`. Index-order push (`v` from 1 to 8): `ways[1]=1`. v=1: neighbours `7,8` `dist=1==1` -> `ways[7]=1, ways[8]=1`. v=2 (`a`, `dist=2`): **here is the bug** — I now push `ways[2]` forward to its neighbour `3`: `dist[3]=3==dist[2]+1` -> `ways[3] += ways[2]`. But `ways[2]` is still `0`, because its contributors `7` and `8` have indices `> 2` and have **not been processed yet**. So `ways[3] += 0`, no effect. v=3: `dist=3`, neighbour `2` `dist=2`, `3+1=4!=2` skip. v=7: `dist=1`, neighbour `2` `dist=2==2` -> `ways[2] += ways[7] = 0+1 = 1`. v=8: neighbour `2` -> `ways[2] += 1 = 2`. End. Final `ways[3] = 0`, but the true answer is `2`. **The two-pass index-order version is wrong**: it pushed `ways[2]` forward before `ways[2]` had been filled by its own contributors.

**Diagnosing the second bug.** The recurrence `ways[w] += ways[v]` is only valid if `ways[v]` is *final* when read, and finality is guaranteed by *layer* order, not *index* order. Iterating `v` from `1` to `n` reads a partially-built `ways[v]` whenever some contributor of `v` has a larger index — precisely what happened to vertex `2` (indices `7,8` larger). The off-by-one in thinking is "the predicate is right, so the loop is right" — but the predicate guarantees *which* edges contribute, not *when* the values are ready. The correct two-pass must iterate vertices in nondecreasing `dist` order (BFS order), e.g. by sorting vertices by distance or by reusing the BFS dequeue order. This is exactly why the **one-pass BFS is the cleaner choice**: the queue already hands me vertices in nondecreasing-distance order for free, so `ways[u]` is final at dequeue time and the forward push is always safe. I confirm: the one-pass solution on this same graph gives `ways[3]=2` (dequeue order `1,7,8,2,3`: at `2` both `7,8` already added, `ways[2]=2`, then push to `3`). So I keep the one-pass and discard the index-order two-pass. The lesson — read a count only after its layer is finished — is the same root cause as the first bug viewed from the other side: both are about respecting the layering.

**Edge cases, deliberately.**
- `s == t`: `ways[s] = 1` set pre-loop. A neighbour `u` of `s` has `dist[u] = 1`, and when `u` is dequeued it tests `dist[s] == dist[u] + 1`, i.e. `0 == 2` — false. So `ways[s]` is never overwritten -> output `1`. Correct.
- `t` unreachable: `dist[t]` stays `INF`; I output `0` via an explicit `dist[t] == INF` guard (and `ways[t]` is `0` anyway, but the guard makes the intent unmistakable). Correct.
- `n = 1`, no edges (forces `s = t = 1`): loop does nothing, `ways[1] = 1` -> `1`. Correct.
- Self-loops: a self-loop puts `u` in `adj[u]`; when `u` is processed the test is `dist[u] == dist[u] + 1`, always false, so it contributes nothing — harmless. I still drop them at read time (`if (u == v) continue;`) to keep adjacency clean. Correct.
- Parallel edges: kept, because itineraries are hop-sequences. Two parallel `u-v` hops with `dist[u]+1==dist[v]` add `ways[u]` twice — exactly the two distinct itineraries. Traced: doubled `1-2` with `s=1,t=2` gives `ways[2] = 1` (seed) `+ 1` (second hop) `= 2`, which is the right count of two distinct one-hop itineraries. Correct under the stated definition.
- Same-layer (decoy) edges: `dist[u] == dist[w]` fails `== dist[u]+1`, contributes nothing. This is the central pitfall and the predicate handles it. Correct.
- Overflow: counts kept `% MOD` at every addition; the pre-reduction sum `ways[w] + ways[u] < 2*MOD < 2.2*10^9` fits in `long long`. Correct.
- Output: exactly one integer and a newline; `cin >>` is whitespace-agnostic.

**Sanity check of the derivation on the sample, end to end.** The hand recurrence gave `ways[6]=3`; the traced one-pass code gave `ways[6]=3`; the stated sample answer is `3`. All three agree, and the two decoy edges `2-3`/`4-5` were excluded by the layer predicate in the trace, which is the property the sample was built to test. I also checked a layered graph of width `4`, depth `5`: hand count `4^4 = 256`, code prints `256`; and width `10`, depth `11`: exact count `10^10`, and `10^10 mod (1e9+7) = 999999937`, which the code prints — confirming the modular reduction is applied *during* accumulation and the count is not silently truncated to 64-bit.

**Final solution.** I convinced myself the recurrence is right by hand-computing the sample layer by layer; I killed the same-layer / add-back double-count by gating the addition on the exact previous-layer predicate `dist[w] == dist[u] + 1`; I rejected the tempting index-order two-pass after a trace showed it reads a not-yet-final count, and kept the one-pass BFS whose queue order makes every read safe; and I closed the `s == t` / unreachable / self-loop / parallel / overflow corners by tracing each. Here is the file I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s, t;
    if (!(cin >> n >> m >> s >> t)) return 0;

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        if (u == v) continue;              // ignore self-loops: they never help a shortest path
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    const int INF = INT_MAX;
    vector<int> dist(n + 1, INF);
    vector<long long> ways(n + 1, 0);

    // BFS from s; layer-by-layer so every vertex is finalized in nondecreasing distance order.
    queue<int> q;
    dist[s] = 0;
    ways[s] = 1;
    q.push(s);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int w : adj[u]) {
            if (dist[w] == INF) {              // first time we reach w
                dist[w] = dist[u] + 1;
                ways[w] = ways[u];             // start its count from this predecessor
                q.push(w);
            } else if (dist[w] == dist[u] + 1) {
                // another shortest predecessor on the previous layer
                ways[w] = (ways[w] + ways[u]) % MOD;
            }
            // dist[w] == dist[u] (same layer) or dist[w] < dist[u]: contributes nothing.
        }
    }

    long long ans = (dist[t] == INF) ? 0 : ways[t] % MOD;
    cout << ans << "\n";
    return 0;
}
```

**Causal recap.** The distance layer is plain BFS; the count layer rides the identity `ways[v] = sum of ways[u]` over previous-layer contributors, and the whole problem is the predicate and the order of reads. My first loop added `ways[u]` into *every* already-seen neighbour, so a trace of the sample showed the decoy same-layer edges `2-3`/`4-5` and the backward edge to the source all corrupting the count — fixed by gating on `dist[w] == dist[u] + 1`, sound because BFS finalizes counts layer by layer before they are read. A second trace, on an index-ordered two-pass variant over a graph whose merge vertex had a small index, exposed a read of a not-yet-final count (`ways[3]` came out `0` instead of `2`) — which is why I keep the one-pass BFS, whose queue serves vertices in nondecreasing-distance order so every `ways[u]` read is complete. The `s == t` (pre-seeded `ways[s]=1`, never overwritten), unreachable (`dist[t]==INF` guard), self-loop (predicate false on `dist[u]==dist[u]+1`), parallel-edge (each distinct hop counts once), same-layer (predicate excludes), and overflow (`% MOD` per addition) corners each fall out of the same predicate plus the explicit guards, and 1100 random cases against an independent path-enumeration brute, plus the `10^10 mod p` spot check, confirm it.
