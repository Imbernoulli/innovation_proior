# Context

## Problem

Given a rooted tree of $n$ nodes, answer $q$ queries each asking the Lowest Common Ancestor of two nodes $u$ and $v$, with $O(n \log n)$ preprocessing and $O(1)$ time per query. ($n, q$ up to $\sim 10^5 / 10^6$.)

The tree is rooted at node $0$. Input is 1-based: `n q`, then `n - 1` undirected tree edges, then `q` query pairs. The program may convert node ids to 0-based internally and must print 1-based answers.

## Code framework

Deliver a single self-contained C++17 program that reads from stdin and writes to stdout. The skeleton below parses the tree into a 0-based adjacency list and stores the 0-based query pairs; fill the algorithm body and print one answer per line.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<vector<int>> adj(n);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        --u;
        --v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<pair<int, int>> queries(q);
    for (int i = 0; i < q; ++i) {
        int u, v;
        cin >> u >> v;
        --u;
        --v;
        queries[i] = {u, v};
    }

    vector<int> answers(q);

    // TODO: algorithm body

    for (int ans : answers) {
        cout << ans + 1 << '\n';
    }

    return 0;
}
```
