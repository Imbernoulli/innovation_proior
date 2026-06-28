# Context

## Problem

Given a tree of $n$ nodes each holding a value, support two online operations:
update the value at a node, and query the sum (or maximum) of values on the path
between two given nodes $u$ and $v$ — fast enough for $n, q$ up to $\sim 10^5$.

The operations arrive online: each must be answered before the next is read, so
the queries cannot be reordered or batched offline. The path between $u$ and $v$
is the unique simple path in the tree, and it can contain anywhere from one to
$n$ nodes.

## Code framework

The deliverable is a single self-contained C++17 program that reads from stdin
and writes to stdout. It follows the judged input format: read `n`, then the
`n-1` one-based tree edges, then the `n` node weights, then `q` operations.
`CHANGE u t` updates node `u`, while `QMAX u v` and `QSUM u v` print one line
with the path maximum or path sum respectively.

```cpp
#include <bits/stdc++.h>
using namespace std;

int n;
vector<vector<int>> adj;
vector<long long> wt;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n)) return 0;
    adj.assign(n + 1, {});
    wt.assign(n + 1, 0);
    for (int i = 0; i < n - 1; i++) {
        int a, b; cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    for (int i = 1; i <= n; i++) cin >> wt[i];

    // TODO

    int q; cin >> q;
    string op; int u, v;
    while (q--) {
        cin >> op >> u >> v;
        if (op == "CHANGE") updateNode(u, v);
        else if (op == "QMAX") cout << pathMax(u, v) << '\n';
        else if (op == "QSUM") cout << pathSum(u, v) << '\n';
    }
    return 0;
}
```
