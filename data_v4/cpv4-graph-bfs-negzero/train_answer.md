**Problem.** A signal broadcasts from one transmitter `s` in an undirected, unweighted graph of `n`
nodes and `m` cables, flooding outward one hop per tick. A node first lit at tick `d` lies on ripple
layer `d` (its shortest hop-distance from `s`). Each node `i` has a brightness `w[i]` that may be
negative, zero, or positive; a layer's brightness is the sum of `w[i]` over the nodes at exactly that
distance. Unreachable nodes belong to no layer. Read `n m s`, the brightnesses, and the `m` edges from
stdin; print the brightness of the brightest ripple layer. Layer `0` always contains `s`, so an answer
always exists and it may be negative.

**Key idea — BFS layers plus a per-distance bucket.** The graph is unweighted and single-source, so a
breadth-first search from `s` labels every reachable node with its shortest hop-distance `dist[v]`
(`-1` if unreachable). Then bucket brightnesses by distance and take the maximum bucket:

- `dist[s] = 0`; BFS relaxes `dist[v] = dist[u] + 1` the first time each `v` is seen.
- For each *reachable* node `v`, add `w[v]` into `layerSum[dist[v]]`.
- Answer = `max over d of layerSum[d]`.

BFS distance labels are contiguous (`0, 1, ..., maxDist`, no gaps), so every index is a real,
non-empty layer.

**Correctness.** On an unweighted graph BFS labels each node along a shortest path (it expands in
nondecreasing distance order), so `dist[v]` is the true minimum hop-distance and the partition into
layers is exactly the ripple structure. Summing brightnesses per distance and maximizing gives the
brightest layer by definition. Layer `0 = {s}` is always present, so the maximum is well-defined.

**Pitfalls.**
1. *Base case / sign handling.* The running maximum must start at `LLONG_MIN`, not `0`. Brightnesses
   can all be negative, in which case the brightest layer is itself negative; a `0` seed silently
   invents a non-existent "broadcast nothing, score 0" option and returns `0`. On an all-negative
   graph like `w=[-3,-1,-4]` along a path from `s`, the answer is `-1`, not `0`.
2. *Unreachable sentinel leaking into the index.* Accumulate only when `dist[v] != -1`. Iterating all
   nodes and writing `layerSum[dist[v]]` unconditionally lets an unreachable node's `dist = -1` index
   the bucket array at `-1` — out of bounds (undefined behaviour) and semantically wrong, since
   unreachable nodes must contribute to no layer.
3. *Zero-sum layer is still a real layer.* Never infer emptiness from `layerSum[d] == 0`; a single
   zero-brightness node, or `+4` and `-4` together, is a genuine layer. Iterate the full distance range
   and rely on BFS contiguity instead.
4. *Overflow.* A star centered at `s` puts up to `n-1` leaves on layer `1`, summing to `~2*10^14`; use
   `long long` for brightnesses, buckets, and the maximum. `int` is a silent wrong-answer on large
   tests.

**Edge cases.** `n = 1, m = 0` (lone transmitter): answer is `w[s]`, possibly negative or zero. All
negatives: least-negative layer. Unreachable component: excluded by the `dist != -1` guard. Zero layer:
counted, value `0`. The `LLONG_MIN` seed is only read inside `max` and always overwritten by layer `0`
before output, so it never leaks.

**Complexity.** `O(n + m)` time, `O(n)` space.

**Code.**

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
