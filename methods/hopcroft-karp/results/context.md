# Context

## Problem

Given a bipartite graph with left part `L` of size `n`, right part `R` of size `m`, and edges only between `L` and `R`, find a maximum-cardinality matching: the largest possible set of edges such that no two chosen edges share an endpoint.

The deliverable is a single self-contained C++17 program. It reads from standard input: `n` (left size), `m` (right size), `e` (edge count), followed by `e` edges `u v` with `u` in `[0, n)` and `v` in `[0, m)`. It writes to standard output the matching size, then one matched `(left, right)` pair per line.

## Code framework

The parser and output harness are already fixed. The missing work is the implementation that fills `match_l` and `match_r` with a largest valid set of pairs.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    long long e;
    if (!(cin >> n >> m >> e)) return 0;

    vector<vector<int>> adj(n);
    for (long long i = 0; i < e; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
    }

    vector<int> match_l(n, -1), match_r(m, -1);

    // TODO: fill the required partner arrays.

    int size = 0;
    for (int u = 0; u < n; ++u) {
        if (match_l[u] != -1) ++size;
    }

    cout << size << '\n';
    for (int u = 0; u < n; ++u) {
        if (match_l[u] != -1) {
            cout << u << ' ' << match_l[u] << '\n';
        }
    }

    return 0;
}
```
