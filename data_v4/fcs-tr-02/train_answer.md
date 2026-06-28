**Problem.** Given a tree on `n` vertices with positive integer edge weights and a target `L`, count
the unordered pairs of distinct vertices `(u, v)` whose unique tree path has total weight exactly `L`.
Read `n`, `L`, and the `n - 1` weighted edges from stdin; print the count. With `n` up to `2*10^5`
there are ~`2*10^10` pairs, so per-pair enumeration is impossible, and the count itself can exceed
32 bits.

**Why the obvious approaches are wrong / too slow.** BFS from every vertex and tally `dist == L` is
`O(n^2)` — about `4*10^10` operations at the limit, roughly a hundred times over a 2-second budget. It
is correct, so it is the oracle, not the solution. Rooting the tree and merging per-subtree
distance→count maps (small-to-large) is `O(n log n)` in the number of insertions but pays a large
hash-map constant and is shape-sensitive; it is fiddly and risky at this scale.

**Key idea — centroid decomposition with a frequency array.** Pick the **centroid** `c` (removing it
leaves every component with at most half the vertices) and split all paths into: those **through `c`**
(endpoints in different components, or one endpoint is `c`) and those **inside one component** (solved
by recursion). A path through `c` from `a` to `b` has weight `dist(a,c) + dist(c,b)`, so if `freq[d]`
counts vertices at distance `d` from `c`, the partners of a vertex at distance `d` are `freq[L - d]`.
The half-size guarantee bounds the recursion depth to `O(log n)`, giving `O(n log n)` overall. Every
path is counted exactly once — at the highest centroid lying on it.

To avoid counting *same-branch* pairs (which do not pass through `c`), process `c`'s branches one at a
time with the **add-then-insert** rule: for the current branch first add `freq[L - d]` for each of its
distances (against previously inserted branches plus the centroid), then insert the branch's distances.
Seed `freq[0] = 1` to represent `c` itself, which counts paths with one endpoint equal to `c`.

**Pitfalls.**
1. *Centroid counted twice.* The centroid must live in `freq` exactly once — only via the `freq[0]=1`
   seed. Each branch gather must start at `c`'s neighbor with base distance equal to that edge's
   weight and never re-emit `c`. (A 3-vertex path `1-2-3`, `L=4`, returns `2` instead of `1` if `c` is
   double-represented.)
2. *Same-branch double counting.* Use add-then-insert per branch so only cross-branch pairs are
   counted here; within-branch pairs are counted in the recursion.
3. *Frequency-array size.* Distances can reach the diameter (`~2*10^{11}`), far too large to index.
   But positive weights mean only distances `<= L` can form an exact-`L` path, so cap the array at
   `L + 1` and prune any subtree whose running distance exceeds `L`.
4. *Overflow.* A `2*10^5`-leaf star at `L = 2w` yields `~2*10^{10}` pairs; the answer accumulator must
   be `long long`.
5. *Recursion depth.* Run the decomposition iteratively (explicit component stack) so chains do not
   risk the system stack.

**Edge cases.** `n = 1` -> `0` (no pairs); `n = 2` -> `1` iff the edge weight equals `L`; `L = 0` ->
`0` (positive weights cannot sum to 0); chain at `L` reachable -> `n - L` pairs for unit weights;
diameter `L` -> `1`. All confirmed against the `O(n^2)` oracle.

**Complexity.** `O(n log n)` time (centroid depth `O(log n)`, linear work per level), `O(n + L)`
memory.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;

    // Adjacency: for each vertex, list of (neighbor, weight).
    vector<vector<pair<int,int>>> adj(n + 1);
    for (int e = 0; e < n - 1; e++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }

    // Frequency array indexed by distance value in [0, L].
    // Edge weights are positive, so only distances <= L can ever combine to
    // exactly L; we never index outside [0, L].
    vector<long long> freq(L + 1, 0);
    freq[0] = 1; // the centroid itself sits at distance 0 (counts centroid-endpoint paths)

    vector<char> removed(n + 1, 0); // vertices already used as a centroid
    vector<int> sub(n + 1, 0);       // subtree sizes (recomputed per component)
    vector<int> par(n + 1, 0);       // parent in the current rooted traversal

    long long answer = 0;

    // Compute subtree sizes of the component containing `root`, return component size.
    // Fills sub[] and par[] for vertices in this component (rooted at root).
    auto computeSizes = [&](int root) -> int {
        vector<int> ord;
        ord.reserve(64);
        vector<pair<int,int>> stk;
        stk.push_back({root, 0});
        while (!stk.empty()) {
            auto [u, p] = stk.back(); stk.pop_back();
            par[u] = p;
            ord.push_back(u);
            for (auto [v, w] : adj[u]) {
                if (v != p && !removed[v]) stk.push_back({v, u});
            }
        }
        for (int u : ord) sub[u] = 1;
        for (int i = (int)ord.size() - 1; i >= 0; i--) {
            int u = ord[i], p = par[u];
            if (p != 0) sub[p] += sub[u];
        }
        return (int)ord.size();
    };

    // Find the centroid of a component, given precomputed sub[]/par[] rooted at `root`.
    auto findCentroid = [&](int root, int total) -> int {
        int cur = root, prev = 0;
        while (true) {
            int nxt = -1;
            for (auto [v, w] : adj[cur]) {
                if (v == prev || removed[v]) continue;
                // size of the component "behind" v when we cut edge (cur,v):
                // if v is a child of cur in the root-tree, that side is sub[v];
                // otherwise (v == par[cur]) that side is total - sub[cur].
                int sz = (par[v] == cur) ? sub[v] : (total - sub[cur]);
                if (sz > total / 2) { nxt = v; break; }
            }
            if (nxt == -1) break;
            prev = cur;
            cur = nxt;
        }
        return cur;
    };

    // Gather distances of all nodes in the branch rooted at `start` (parent `parent0`),
    // starting from baseDist; keep only distances <= L.
    vector<long long> dists;
    auto gatherDists = [&](int start, int parent0, long long baseDist) {
        dists.clear();
        vector<tuple<int,int,long long>> stk;
        stk.push_back({start, parent0, baseDist});
        while (!stk.empty()) {
            auto [u, p, d] = stk.back(); stk.pop_back();
            dists.push_back(d); // d <= L guaranteed by the push guard below / base check
            for (auto [v, w] : adj[u]) {
                if (v == p || removed[v]) continue;
                long long nd = d + w;
                if (nd <= L) stk.push_back({v, u, nd});
            }
        }
    };

    // Iterative centroid decomposition over an explicit stack of component entry points.
    vector<int> compStack;
    compStack.push_back(1);

    while (!compStack.empty()) {
        int entry = compStack.back(); compStack.pop_back();
        if (removed[entry]) continue;
        int total = computeSizes(entry);
        int c = findCentroid(entry, total);

        // Count paths through centroid c. freq holds only freq[0] = 1 right now.
        // For each branch of c: first ADD over already-inserted branches (and the
        // centroid via freq[0]), then insert this branch's distances. This counts
        // every cross-branch pair exactly once and centroid-endpoint paths once.
        vector<long long> inserted;
        for (auto [child, w] : adj[c]) {
            if (removed[child]) continue;
            long long base = (long long)w;
            if (base > L) continue; // whole branch starts beyond L; no node can match
            gatherDists(child, c, base);
            for (long long d : dists) {
                long long need = L - d;
                if (need >= 0) answer += freq[need]; // need <= L since d >= 0
            }
            for (long long d : dists) {
                freq[d] += 1;
                inserted.push_back(d);
            }
        }
        for (long long d : inserted) freq[d] -= 1; // restore freq to {freq[0]=1}

        // Remove the centroid and recurse into the pieces around it.
        removed[c] = 1;
        for (auto [v, w] : adj[c]) {
            if (!removed[v]) compStack.push_back(v);
        }
    }

    cout << answer << "\n";
    return 0;
}
```
