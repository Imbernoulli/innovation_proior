## Problem

You are given a connected undirected graph on $V$ vertices and $E$ edges ($V \le 200$, $E \le 10000$). Every edge $e$ carries two positive integer weights: a *time* $t_e$ and a *cost* $c_e$, with $1 \le t_e, c_e < 256$.

For any spanning tree $T$ of the graph, define its value to be the product of its total time and its total cost:

$$\Big(\sum_{e \in T} t_e\Big)\cdot\Big(\sum_{e \in T} c_e\Big).$$

Find a spanning tree $T$ that minimizes this value, and output its edges.

## Code framework

Write a single-file C++17 program that reads from stdin and writes to stdout.
The input begins with `V E`, followed by `E` lines `u v t c` for 0-indexed
edge endpoints, time, and cost. The output should be the chosen tree's total
time and total cost as `X Y`, followed by the `V-1` selected edges, one `u v`
per line.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int u, v, t, c;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int V, E;
    if (!(cin >> V >> E)) return 0;

    vector<Edge> edges(E);
    for (int i = 0; i < E; ++i) {
        cin >> edges[i].u >> edges[i].v >> edges[i].t >> edges[i].c;
    }

    long long X = 0, Y = 0;
    vector<pair<int, int>> tree_edges;

    // TODO:

    cout << X << ' ' << Y << '\n';
    for (auto [u, v] : tree_edges) {
        cout << u << ' ' << v << '\n';
    }

    return 0;
}
```
