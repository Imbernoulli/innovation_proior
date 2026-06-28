## Problem

There are $n$ projects. Project $i$ yields integer profit $p_i$, which may be negative (for
example, a project might require purchasing a machine, so undertaking it costs money rather
than earning it). Some projects depend on others: a project may be selected only if **all**
of its prerequisites are also selected. (Prerequisites may chain, and a project may be a
prerequisite of several others.)

Choose a subset of the projects, respecting every prerequisite, so as to maximize the total
profit $\sum_{i \in \text{chosen}} p_i$.

## Code framework

The deliverable is a single self-contained C++17 program. It reads `n m`, then the
`n` station costs, then `m` lines `a_j b_j c_j` describing a user group that uses
stations `a_j` and `b_j` (1-indexed) and yields revenue `c_j`. It writes the maximum
achievable net profit (total revenue minus total build cost) to stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<long long> p(n);
    for (int i = 0; i < n; i++) {
        cin >> p[i];
    }

    vector<int> a(m), b(m);
    vector<long long> c(m);
    for (int j = 0; j < m; j++) {
        cin >> a[j] >> b[j] >> c[j];
        --a[j];
        --b[j];
    }

    long long answer = 0;
    // TODO: compute answer.

    cout << answer << "\n";
    return 0;
}
```
