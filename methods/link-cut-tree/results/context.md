# Context

## Problem

Maintain a forest of $n$ nodes (each with a value) under online operations:
`link(u, v)` add an edge $(u, v)$ given they are in different trees; `cut(u, v)`
remove edge $(u, v)$; `connected(u, v)`; and path aggregate (e.g. sum or max of
node values) on the path between $u$ and $v$. $n$, ops up to $\sim 10^5$.

Each operation arrives online and must be answered before the next is read, so
the operation stream cannot be reordered or batched. Crucially the *shape* of the
forest changes over time: `link` and `cut` insert and delete edges, so the set of
trees and the paths inside them are not fixed in advance. The path between $u$ and
$v$ (when they are connected) is the unique simple path; it can contain anywhere
from one to $n$ nodes.

## Code framework

The deliverable is a single self-contained C++ program reading from stdin and
writing to stdout. It reads `n q`, then `n` integer node values, then `q`
operations `op a b` with `op` in {`link`, `cut`, `conn`, `path`}. For each
`conn` operation it prints `0` or `1`, and for each `path` operation it prints
the path aggregate, one answer per line. Node ids are `1..n`; index `0` may be
reserved internally as a null sentinel.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> value(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> value[i];
    }

    ostringstream output;
    for (int i = 0; i < q; ++i) {
        string op;
        int a, b;
        cin >> op >> a >> b;

        // TODO: maintain the forest state and emit answers for query operations.
    }

    cout << output.str();
    return 0;
}
```
