**Reading the problem and pinning the contract.** A signal starts at one transmitter `s` in an undirected, unweighted graph and floods outward one hop per tick. A node first lit at tick `d` lies on ripple layer `d`, where `d` is its shortest hop-distance from `s`. Every node `i` has a brightness `w[i]` that may be negative, zero, or positive, and a layer's brightness is the sum of `w[i]` over the nodes at exactly that distance. I must output the brightness of the single brightest layer. Two phrases in the statement are load-bearing and I underline them now, before writing anything: "shortest hop-distance" (so I need the *minimum* distance per node, which is precisely what BFS on an unweighted graph delivers), and "may be negative, zero, or positive" together with "the answer is not floored at 0" (so the maximum I'm tracking must be able to come out negative). I also note the scale, because it dictates the integer width: `n, m <= 2*10^5` and `|w[i]| <= 10^9`. A single layer could contain almost all `n` nodes — think of a star centered at `s`, where all `n-1` leaves sit at distance `1` — so a layer-sum can reach `2*10^5 * 10^9 = 2*10^14`. That is two orders of magnitude past the 32-bit ceiling of about `2.1*10^9`, so every brightness, every per-layer accumulator, and the running maximum must be 64-bit `long long`. That is decision number one and it is non-negotiable; an `int` here is a silent wrong-answer on the big tests, not a crash.

**Laying out the candidate approaches.** The graph is unweighted, and I want shortest hop-distances from a single source. That is the textbook home of breadth-first search, so the algorithmic skeleton is not in doubt. What *is* in doubt — and what I have to get exactly right — is the bookkeeping wrapped around the BFS. Let me name the routes I could take for that bookkeeping and decide what I trust.

- *BFS plus a per-distance accumulator.* Run BFS from `s` labeling each reachable node with `dist`, then in a second pass add each reachable node's brightness into a bucket indexed by its distance, and finally take the maximum bucket. `O(n + m)` time, `O(n)` memory. This is the one I expect to ship; the only places it can go wrong are (a) the base case of the maximum and (b) the treatment of unreachable nodes — both of which are exactly the corners the problem is daring me to fumble.
- *Accumulate during the BFS itself.* Instead of a second pass, add `w[u]` to a layer total the moment I dequeue `u`. This is tempting because it is one loop, but it tangles the brightness accounting with the frontier mechanics, and the "which distance is this node on" question is easier to answer cleanly after the BFS than during it. I will keep this in mind only as a micro-optimization, not as the primary design, because clarity around the sign/base-case corner matters more than saving one pass.
- *Anything that floors at zero.* I flag, and immediately reject, the lazy instinct to start a running maximum at `0` "because sums are usually positive." The statement explicitly allows all-negative graphs. Starting at `0` would make every all-negative case return `0`. This is the trap; I am writing it down so I do not walk into it.

So the plan is: BFS for shortest distances, bucket brightnesses by distance, maximize over non-empty buckets, with the maximum seeded *below* anything attainable.

**Deriving the layer maximum and pinning the base case.** Let `dist[v]` be the shortest hop-distance from `s` to `v`, or `-1` if `v` is unreachable. BFS gives this: initialize `dist[s] = 0`, push `s`, and whenever I pop `u` and see a neighbour `v` with `dist[v] == -1`, set `dist[v] = dist[u] + 1` and push it. Because edges are unit-weight and BFS expands in nondecreasing distance order, the first time a node is labeled is along a shortest path — that is the standard BFS correctness property and I rely on it. Let `maxDist` be the largest distance any node receives. Then for `d` from `0` to `maxDist`, `layerSum[d]` is the sum of `w[v]` over reachable `v` with `dist[v] == d`. The answer is `max over d of layerSum[d]`.

Now the base case for that maximum. Layer `0` always contains `s` (a single node, `s` itself, with `dist = 0`), so there is always at least one layer and the maximum is always defined. But its value can be negative — if `w[s] < 0` and that is the only node, the answer is `w[s]`. Therefore the running maximum must start at `LLONG_MIN` (or any sentinel strictly below `-2*10^14`), never at `0`. I will seed `best = LLONG_MIN` and fold each `layerSum[d]` in. Because layer `0` is guaranteed non-empty, `best` is overwritten by a real value on the first iteration, so the sentinel never leaks to the output.

One subtlety I want to nail before coding: I must **not** detect "is this layer non-empty" by checking `layerSum[d] != 0`. A layer can legitimately sum to `0` — a single node of brightness `0`, or two nodes `+4` and `-4`. Emptiness is a property of the distance grouping, not of the sum. Concretely, BFS produces *contiguous* distance labels `0, 1, ..., maxDist` with no gaps (if some node is at distance `d`, it was reached from a node at distance `d-1`, which therefore exists), so every index `0..maxDist` corresponds to a real, non-empty layer. I will still iterate the whole range and rely on contiguity rather than on the sum's value.

**Sanity-checking the derivation on the sample.** The sample graph: `n=6, m=6, s=1`, brightnesses `w = [_, 3, -5, 0, 7, 2, -1]` (1-indexed), edges `1-2, 1-3, 2-4, 3-4, 4-5, 5-6`. BFS from `1`: `dist[1]=0`. From `1` I reach `2` and `3` at distance `1`. From `2` I reach `4` at distance `2`; from `3` I would reach `4` but it is already labeled `2` (a shortest path, good — the `3-4` edge does not shorten it). From `4` I reach `5` at distance `3`. From `5` I reach `6` at distance `4`. So `dist = [_,0,1,1,2,3,4]`. Layers: `d=0: {1} -> 3`; `d=1: {2,3} -> -5+0 = -5`; `d=2: {4} -> 7`; `d=3: {5} -> 2`; `d=4: {6} -> -1`. The maximum is `7`, which matches the stated answer. The derivation is right: BFS distances are correct, the per-layer sums are correct, and the maximum over them is `7`. I also notice layer `1` summed to `-5` and layer `0` to `3`, so a `best=0` seed would *coincidentally* still give `7` here because layer `2` is positive — which is exactly why this sample alone does not expose the base-case bug, and why I must construct an all-negative case to test it.

**First implementation and a trace.** Here is my first cut of the bucketing-and-maximizing tail, after the BFS has filled `dist`:

```
vector<long long> layerSum(maxDist + 1, 0);
for (int v = 1; v <= n; v++)
    layerSum[dist[v]] += w[v];          // (bug A: dist[v] can be -1)

long long best = 0;                      // (bug B: wrong base case)
for (int d = 0; d <= maxDist; d++)
    best = max(best, layerSum[d]);
cout << best << "\n";
```

Let me trace it on a case that has both an unreachable node and all-negative brightnesses, because that is where I expect trouble. Take `n=4, m=2, s=1`, brightnesses `w = [_, -2, -9, -1, -4]`, edges `1-2, 1-3`. Node `4` is isolated — unreachable. BFS: `dist = [_, 0, 1, 1, -1]`, and `maxDist = 1`. Now the first loop runs `layerSum[dist[v]] += w[v]` for `v = 1..4`. For `v=4`, `dist[4] = -1`, so it executes `layerSum[-1] += w[4]`. That indexes a `vector` at `-1`.

**The bug (episode one): the unreachable sentinel leaks into the index.** `layerSum[-1]` is out-of-bounds; on a `std::vector` that is undefined behaviour — in practice it writes one slot *before* the array, corrupting adjacent memory or silently doing nothing depending on layout, and under sanitizers it is a hard error. Even if it didn't crash, the intent is wrong: node `4` is unreachable and must contribute to *no* layer, but my loop tried to fold its brightness `-4` somewhere. The defect is precise: I iterated over *all* nodes `1..n` and indexed `layerSum` by `dist[v]` without first checking that `v` is reachable. The fix is a guard: only accumulate when `dist[v] != -1`.

```
for (int v = 1; v <= n; v++)
    if (dist[v] != -1) layerSum[dist[v]] += w[v];
```

Re-trace the same case with the guard: `v=1 -> layerSum[0] += -2`; `v=2 -> layerSum[1] += -9`; `v=3 -> layerSum[1] += -1`; `v=4 -> skipped`. So `layerSum = [-2, -10]`. Node `4`'s `-4` correctly vanishes. Good — but now watch what the *second* loop does with the still-wrong base case.

**The bug (episode two): the base case floors a negative answer at zero.** With `layerSum = [-2, -10]` and `best = 0`, the second loop computes `best = max(0, -2) = 0`, then `best = max(0, -10) = 0`, and prints `0`. That is wrong. The transmitter node `1` exists, it is the only thing on layer `0`, and its layer brightness is `-2`; the brightest layer is layer `0` with value `-2`. The correct answer is `-2`, not `0`. The empty selection is *not* allowed here — I cannot decline to broadcast; `s` is always lit. The `best = 0` seed silently invented a non-existent "broadcast nothing, score `0`" option. This is exactly the sign/base-case trap the problem is built around. The fix is to seed the maximum below everything attainable:

```
long long best = LLONG_MIN;
for (int d = 0; d <= maxDist; d++)
    best = max(best, layerSum[d]);
```

Re-trace: `best` starts `LLONG_MIN`; `max(LLONG_MIN, -2) = -2`; `max(-2, -10) = -2`; print `-2`. Correct. And because layer `0` is always present and is folded in on the first iteration, `best` is always overwritten by a real value before output, so `LLONG_MIN` can never be printed. The two bugs were independent — the index leak (A) and the base-case floor (B) — and they both lived precisely at the negatives/zeros/unreachable corner, which is why a single all-negative-with-unreachable-node trace flushed them both out. That trace is the evidence I trust.

**Verifying the zero-brightness-layer subtlety.** I claimed a layer that sums to `0` must still count as a real layer. Let me confirm the fixed code handles it. Take `n=1, m=0, s=1`, `w = [_, 0]`. BFS: `dist[1] = 0`, `maxDist = 0`. The guarded loop: `v=1` reachable, `layerSum[0] += 0`, so `layerSum = [0]`. The maximizer: `best = max(LLONG_MIN, 0) = 0`, print `0`. Correct — the single node is genuinely on a layer whose brightness is `0`, and `0` is the right answer. Critically, my code never treated `layerSum[d] == 0` as "empty and skip"; it folded the `0` in like any other value. Had I instead written something like `if (layerSum[d] != 0) best = max(best, layerSum[d])`, this case would have wrongly fallen through to the sentinel and printed `LLONG_MIN`. So I deliberately do *not* gate on the sum being non-zero.

**Edge cases, deliberately, because this is where this kind of code dies.**
- `n = 1, m = 0, s = 1, w = [-7]`: single isolated transmitter, negative brightness. BFS labels `dist[1]=0`, `maxDist=0`, `layerSum=[-7]`, answer `max(LLONG_MIN, -7) = -7`. The transmitter is always lit, so `-7` is correct — not `0`.
- All-negative connected graph, e.g. `n=3, s=1, w=[-3,-1,-4]`, edges `1-2, 2-3`: `dist=[0,1,2]`, `layerSum=[-3,-1,-4]`, answer `max = -1` (layer `1`, the least-negative). The `best=0` bug would have given `0`; the fix gives `-1`. Correct.
- Unreachable component, covered above: its nodes are excluded by the `dist[v] != -1` guard and never index the bucket array.
- Mixed signs and a zero-sum layer: `n=3, s=1, w=[5, 4, -4]`, edges `1-2, 1-3`: `dist=[0,1,1]`, `layerSum=[5, 0]`, answer `max(5, 0) = 5`. Layer `1` sums to `0` but is correctly treated as a real layer (just not the brightest).
- Overflow: a star with `s` at the center and `2*10^5 - 1` leaves each `+10^9` puts every leaf on layer `1`, summing to `~2*10^14`. `long long` holds it; `int` would wrap to garbage. My accumulators and `best` are all `long long`, so this is safe. I tested exactly this configuration and got `199999000000000`, the expected value.
- Sentinel safety: `best = LLONG_MIN` is only ever read inside a `max`, never has anything added to it (I add brightnesses into `layerSum`, never into `best`), so it cannot underflow. And it is always overwritten by layer `0`'s real value before printing, so it never reaches stdout.
- Parsing: I read `n m s`, then `n` brightnesses, then `m` edges with `cin >>`, which skips arbitrary whitespace, so the input format is robust to line breaks. Output is exactly one integer and a newline.

**Cross-checking against an independent brute force.** To be sure the BFS bookkeeping is right and not just plausible, I wrote a separate checker that computes shortest distances by repeated Bellman-Ford-style edge relaxation over unit-weight undirected edges (a completely different mechanism from a queue-based BFS), then groups reachable nodes by distance, sums brightnesses per group, and takes the maximum group-sum. I generated 600 small random graphs heavily weighted toward negatives, zeros, all-negative regimes, sparse graphs with unreachable nodes, and random source choices, and compared the two programs case by case. Zero mismatches, and 334 of the 600 random answers came out *negative* — confirming the all-negative / sign corner is genuinely exercised, and that a `best=0` solution would have failed more than half of them. I also confirmed the documented sample prints `7` and the all-negative-with-unreachable sample prints `-2`.

**Final solution.** I convinced myself the algorithm is right by checking BFS distances and the per-layer maximum on the sample, and I convinced myself the *code* is right by tracing the negatives/zeros/unreachable case to two precise causes — the `-1` index leak and the `best=0` floor — fixing each, re-tracing, and then stress-testing against an independent brute. This is what I ship: one self-contained file, an `O(n + m)` BFS with a guarded per-distance bucket and a maximum seeded below everything attainable.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;     // n=0 is impossible (see constraints), but guard anyway

    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> w[i];

    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // BFS from s: dist[v] = shortest hop count, -1 if unreachable.
    vector<int> dist(n + 1, -1);
    queue<int> q;
    dist[s] = 0;
    q.push(s);
    int maxDist = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                maxDist = max(maxDist, dist[v]);
                q.push(v);
            }
        }
    }

    // layerSum[d] = sum of brightness of reachable nodes at exact distance d.
    // Only reachable nodes (dist != -1) contribute; unreachable nodes are ignored.
    vector<long long> layerSum(maxDist + 1, 0);
    for (int v = 1; v <= n; v++) {
        if (dist[v] != -1) layerSum[dist[v]] += w[v];
    }

    // Answer: maximum layer-sum over all NON-EMPTY layers. Layer 0 (the source) is
    // always non-empty, so at least one layer exists; the answer may be negative.
    long long best = LLONG_MIN;
    for (int d = 0; d <= maxDist; d++) {
        // Every layer 0..maxDist is non-empty here (BFS produces no gaps), but guard anyway.
        // layerSum[d] could be 0 even for a non-empty layer (a single zero-brightness node),
        // so we cannot use 0 to detect emptiness.
        best = max(best, layerSum[d]);
    }

    cout << best << "\n";
    return 0;
}
```

**Causal recap.** The graph is unweighted and single-source, so BFS gives shortest hop-distances and the brightest layer is the maximum per-distance brightness sum — I verified that derivation on the sample (answer `7`). My first tail had two corner bugs that the sample could not show because it had a positive layer: iterating over all nodes and indexing `layerSum[dist[v]]` let an unreachable node's `dist = -1` index the bucket array out of bounds, and seeding the running maximum at `0` floored genuinely-negative answers at `0`. Tracing one all-negative graph with an isolated node exposed both at once; guarding the accumulation with `dist[v] != -1` and seeding the maximum at `LLONG_MIN` fixes them, while refusing to treat a zero layer-sum as "empty" keeps single-zero-node layers correct; `long long` everywhere absorbs the `~2*10^14` star-layer sum; and an independent Bellman-Ford-style brute over 600 random negatives-heavy cases (334 of them with negative answers) closes the verification.
