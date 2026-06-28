# Context

## Problem

You are given a rooted tree of `n` nodes, rooted at node `0`. Each node has a
color. For every node `v`, look at the multiset of colors in the subtree of
`v`. A color is dominating in that subtree if no other color appears more times
there. If several colors tie for the maximum frequency, all of them are
dominating.

Return one value for every node: the sum of all dominating colors in that
node's subtree.

The input size can be as large as `n = 100000`, and every color is an integer in
`[1, n]`.

## Code framework

The required artifact is a single self-contained C++ program that reads from
stdin and writes to stdout. The input gives `n`, then `n` colors for vertices
`1..n`, then `n - 1` undirected edges with one-based endpoints; the tree is
rooted at vertex `1`. Output `n` space-separated sums, one per vertex in vertex
order, followed by a newline.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<int> color(n);
    for (int i = 0; i < n; ++i) cin >> color[i];

    vector<vector<int>> g(n);
    for (int i = 0; i < n - 1; ++i) {
        int x, y;
        cin >> x >> y;
        --x;
        --y;
        g[x].push_back(y);
        g[y].push_back(x);
    }

    vector<long long> ans(n, 0);

    // TODO: compute the answer for each vertex.

    for (int i = 0; i < n; ++i) {
        if (i) cout << ' ';
        cout << ans[i];
    }
    cout << '\n';
    return 0;
}
```
