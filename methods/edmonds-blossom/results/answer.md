# Edmonds' Blossom Algorithm

Given a graph `G` and a matching `M`, repeat:

1. Grow an alternating forest from every exposed vertex.
2. Scan edges incident to outer vertices.
3. If an edge reaches an unreached vertex that is matched by `M`, add the scanned edge and that vertex's matched edge to the forest.
4. If an edge reaches an inner vertex already in the forest, mark the edge examined and do not grow the forest.
5. If an edge joins outer vertices, trace their parent paths. Different roots give an augmenting path; replace `M` by the symmetric difference of `M` and that path.
6. If those outer vertices have the same root, the two tree paths plus the scanned edge form an odd alternating circuit with a base. Shrink that circuit into one pseudovertex and continue the same search in the reduced graph.
7. If an augmenting path is found in a reduced graph, expand pseudovertexes in dependency order from outermost to inner stored circuits. Each stored odd circuit has a unique maximum internal matching compatible with the external vertex used by the lifted path; in the tree-path view this is the unique alternating route from that vertex to the circuit base.
8. If the dense forest search exhausts all possible extensions and no augmenting path is found, stop.

The core invariant is:

`G` has an augmenting path with respect to `M` if and only if the graph produced by shrinking a blossom has an augmenting path with respect to the contracted matching.

The forward direction says shrinking does not hide a real improvement. The backward direction says every contracted augmenting path can be expanded through the stored odd circuit by selecting the unique compatible internal matching, or equivalently the correct alternating route to the base in the search tree.

Therefore each successful iteration increases `|M|` by one, and when the search reports no augmenting path, Berge's theorem proves that `M` is maximum. Edmonds's conceptual cardinality algorithm has the stated upper bounds `n^4` time and `n^2` memory for `n` vertices. The current NetworkX implementation is the weighted Galil/Edmonds primal-dual variant, so its dual and slack machinery is extra, but its `scanBlossom`, `addBlossom`, `expandBlossom`, `augmentBlossom`, and `augmentMatching` routines match the search, nested-blossom storage, expansion, and alternating-flip structure above.

Edmonds's paper also gives the matching-duality theorem: the maximum size of a matching equals the minimum capacity-sum of an odd-set cover. A singleton has capacity `1`; a set of `2k + 1` vertices has capacity `k` and covers edges with both endpoints inside it. This is the dual face of the same obstruction that the algorithm handles by odd circuits.

The landing is a single self-contained C++17 program. It reads from stdin a first line `n m` (`n` vertices labeled `0..n-1`, `m` edges) followed by `m` lines `u v`, and prints the matching size, then one matched edge `u v` per line.

```cpp
// Maximum-cardinality matching in a general undirected graph (Edmonds' blossom).
// Reads from stdin: first line "n m" (n vertices labeled 0..n-1, m edges),
// then m lines each "u v". Prints the matching size, then one "u v" per matched
// edge (each undirected edge once).
#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<int>> g;
vector<int> match_, base_, parent_;
vector<char> used, blossom;

int lca(int a, int b) {
    vector<char> used_path(n, false);
    while (true) {
        a = base_[a];
        used_path[a] = true;
        if (match_[a] == -1) break;
        a = parent_[match_[a]];
    }
    while (true) {
        b = base_[b];
        if (used_path[b]) return b;
        b = parent_[match_[b]];
    }
}

void mark_path(int v, int b, int children) {
    while (base_[v] != b) {
        blossom[base_[v]] = true;
        blossom[base_[match_[v]]] = true;
        parent_[v] = children;
        children = match_[v];
        v = parent_[match_[v]];
    }
}

bool find_path(int root) {
    fill(used.begin(), used.end(), false);
    fill(parent_.begin(), parent_.end(), -1);
    for (int i = 0; i < n; i++) base_[i] = i;
    queue<int> q;
    q.push(root);
    used[root] = true;
    while (!q.empty()) {
        int v = q.front(); q.pop();
        for (int to : g[v]) {
            if (base_[v] == base_[to] || match_[v] == to) continue;
            if (to == root || (match_[to] != -1 && parent_[match_[to]] != -1)) {
                int cur_base = lca(v, to);
                fill(blossom.begin(), blossom.end(), false);
                mark_path(v, cur_base, to);
                mark_path(to, cur_base, v);
                for (int i = 0; i < n; i++) {
                    if (blossom[base_[i]]) {
                        base_[i] = cur_base;
                        if (!used[i]) {
                            used[i] = true;
                            q.push(i);
                        }
                    }
                }
            } else if (parent_[to] == -1) {
                parent_[to] = v;
                if (match_[to] == -1) {
                    // augmenting path found, lift it
                    int cur = to;
                    while (cur != -1) {
                        int prev = parent_[cur];
                        int nxt = (prev != -1) ? match_[prev] : -1;
                        match_[cur] = prev;
                        match_[prev] = cur;
                        cur = nxt;
                    }
                    return true;
                } else {
                    used[match_[to]] = true;
                    q.push(match_[to]);
                }
            }
        }
    }
    return false;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int m;
    if (!(cin >> n >> m)) return 0;
    g.assign(n, {});
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        g[u].push_back(v);
        g[v].push_back(u);
    }
    match_.assign(n, -1);
    base_.assign(n, 0);
    parent_.assign(n, -1);
    used.assign(n, false);
    blossom.assign(n, false);

    for (int v = 0; v < n; v++)
        if (match_[v] == -1) find_path(v);

    int sz = 0;
    for (int v = 0; v < n; v++)
        if (match_[v] != -1 && v < match_[v]) sz++;
    cout << sz << "\n";
    for (int v = 0; v < n; v++)
        if (match_[v] != -1 && v < match_[v])
            cout << v << " " << match_[v] << "\n";
    return 0;
}
```
