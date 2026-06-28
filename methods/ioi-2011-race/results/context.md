## Problem

You are given a tree on $n$ vertices. Each edge has a non-negative integer
length, and you are given an integer $K$.

Among all simple paths between two distinct vertices, consider those whose total
edge length is exactly $K$. Output the minimum possible number of edges on such
a path, or $-1$ if no path has total length exactly $K$.

## Code framework

The deliverable is a single self-contained C++17 program that reads `n K`
followed by `n-1` lines `u v w` from stdin and writes the answer to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    if (!(cin >> n >> K)) return 0;

    vector<vector<pair<int, long long>>> g(n);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        g[u].emplace_back(v, w);
        g[v].emplace_back(u, w);
    }

    long long answer = -1;
    // TODO:

    cout << answer << "\n";
    return 0;
}
```
