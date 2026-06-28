# Context

## Research question

Maintain an integer array `a[1..n]` under online range operations:

1. `chmin l r x`: for every `i` in `[l, r]`, set `a[i] = min(a[i], x)`;
2. `max l r`: report `max(a[l..r])`;
3. `sum l r`: report `sum(a[l..r])`.

The array and the operation stream can both be very large, so scanning a range for
each operation is too slow.

The deliverable is a single self-contained C++17 program reading from stdin and
writing to stdout.

## Baseline

A segment tree gives the right traversal shape. Range `max` and range `sum`
queries split across children and merge answers. For range updates, the standard
lazy-propagation pattern stores a pending tag at each node and pushes it down
before visiting children.

## Target

Design a segment tree that supports all three operations efficiently, with an
amortized complexity guarantee over a long sequence of operations.

## Input-output contract

The program reads `T` test cases. Each test case gives `n m`, then the array
`a[1..n]`, then `m` operations in numeric form: `0 l r x` applies
`a[i] = min(a[i], x)` on `[l, r]`, `1 l r` prints the range maximum, and
`2 l r` prints the range sum. Indices are 1-based. It prints one line per query
operation. The fixed input loop and output formatting are ordinary
infrastructure; the open part is the data-structure logic.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    if (!(cin >> T)) return 0;

    ostringstream output;
    while (T--) {
        int n, m;
        cin >> n >> m;

        vector<long long> a(n + 1);
        for (int i = 1; i <= n; ++i) {
            cin >> a[i];
        }

        // TODO: maintain the array state and append answers for query operations.
        for (int i = 0; i < m; ++i) {
            int op, l, r;
            cin >> op >> l >> r;

            if (op == 0) {
                long long x;
                cin >> x;
            } else if (op == 1) {
                long long answer = 0;
                output << answer << '\n';
            } else if (op == 2) {
                long long answer = 0;
                output << answer << '\n';
            }
        }
    }

    cout << output.str();
    return 0;
}
```
