The problem is to find a maximum-cardinality matching in a general undirected graph. A matching is a set of edges without shared endpoints, and the goal is to include as many edges as possible. The natural local improvement is to look for an augmenting path: an alternating path that starts and ends at exposed vertices, so flipping which edges are matched along it increases the matching size by one. Berge's theorem says a matching is maximum exactly when no augmenting path exists, so the whole problem reduces to searching for such a path or certifying that none remains.

The difficulty is that the obvious alternating search works cleanly only in bipartite graphs. There, every cycle is even, so vertices can be labeled by parity as the forest grows, and an edge between two outer vertices always connects different trees and immediately yields an augmenting path. In a general graph, an edge can join two outer vertices of the same tree. Those two tree paths plus the new edge form an odd alternating circuit, or blossom. This blossom is neither a direct augmentation nor something that can be safely ignored, because an augmenting path may enter the blossom from outside and exit through the right boundary vertex. Branching over every possibility inside the blossom would defeat the purpose of a polynomial search, so the algorithm needs a way to treat the blossom as a single structural unit without losing any valid augmenting path.

The method is Edmonds' blossom algorithm. It grows an alternating forest from every exposed vertex, scanning edges incident to outer vertices, and whenever it finds a blossom it shrinks the odd circuit into a single pseudovertex. The search then continues in the reduced graph. The key invariant is that the original graph has an augmenting path with respect to the current matching if and only if the reduced graph has an augmenting path with respect to the contracted matching. If the search in the reduced graph finds an augmenting path that avoids all pseudovertices, the same path works in the original graph. If the path uses a pseudovertex, the stored blossom is expanded and the unique internal alternating route that makes the lifted path valid is chosen. Each successful augmentation increases the matching size by one, and when the search exhausts all possibilities without finding an augmenting path, Berge's theorem certifies optimality.

This contraction-and-expansion machinery is what makes the algorithm polynomial. Shrinking collapses the local ambiguity of an odd cycle into a single vertex, preserving the global search structure. Expanding restores the original vertices only when an augmenting path actually passes through the blossom, using the fact that an odd cycle has a unique near-perfect matching compatible with whichever boundary vertex the outside path needs exposed. The dual certificate, the matching polytope odd-set constraints, and the blossoms handled during search are all the same odd-set obstructions in different guises.

As a single self-contained C++17 program, the deliverable reads the graph from stdin -- a first line `n m` (`n` vertices labeled `0..n-1`, `m` edges) followed by `m` lines `u v` -- and prints the matching size, then one matched edge `u v` per line. The search routines are free functions over shared arrays rather than a library object: `lca` finds the blossom base by walking the two tree paths until they meet, `mark_path` records the odd-circuit vertices to be contracted, and `find_path` grows the alternating forest from one exposed root, shrinking blossoms in place by rewriting `base_` and lifting an augmenting path as soon as it reaches an exposed vertex.

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