# Context

## Problem

You are given a connected undirected graph on $N$ vertices, numbered
$0, 1, \dots, N-1$, with $M$ edges. You must answer $Q$ independent queries.

Each query is four integers $(S, E, L, R)$ with $L \le R$. A query asks whether
there is a walk $v_0, v_1, \dots, v_k$ from $v_0 = S$ to $v_k = E$ that proceeds
in two phases separated by a single *switch* at some index $s$ (with
$0 \le s \le k$):

- Every vertex up to and including the switch is "big": $L \le v_0, v_1, \dots, v_s$.
- Every vertex from the switch onward is "small": $v_s, v_{s+1}, \dots, v_k \le R$.

The switch vertex $v_s$ therefore must satisfy $L \le v_s \le R$. Consecutive
vertices on the walk must be joined by an edge. The walk may revisit vertices.
Decide each query independently: output yes if such a walk exists, no otherwise.

Both $N$ and $Q$ can be large (each up to a few hundred thousand).

The deliverable is a single self-contained C++17 program that reads from
standard input and writes to standard output. It reads `N M Q`, then `M` lines
`u v` for 0-based undirected edges, then `Q` lines `S E L R`, and prints one
line per query: `YES` if such a walk exists and `NO` otherwise.

## Code framework

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Query {
    int S, E, L, R;
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, q;
    if (!(cin >> n >> m >> q)) return 0;

    vector<pair<int, int>> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        edges[i] = {u, v};
    }

    vector<Query> queries(q);
    for (int i = 0; i < q; i++) {
        cin >> queries[i].S >> queries[i].E >> queries[i].L >> queries[i].R;
    }

    vector<char> answer(q, 0);

    // TODO: Compute the yes/no result for each query.

    for (int i = 0; i < q; i++) {
        cout << (answer[i] ? "YES" : "NO") << '\n';
    }
    return 0;
}
```
