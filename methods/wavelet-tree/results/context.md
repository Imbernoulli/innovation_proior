# Context

## Problem

Given a static array of $n$ integers, answer queries: the $k$-th smallest value in
`a[l..r]`, and the number of elements in `a[l..r]` that are $\le x$ (range rank).
($n, q$ up to $\sim 10^5$.)

The array is fixed after input; queries do not update it. Indices are $1$-based with
$1 \le l \le r \le n$, and for the order-statistic query $k$ is $1$-based with
$1 \le k \le r - l + 1$. Values may be negative and may repeat; equal values count
with multiplicity.

## Code framework

The deliverable is a single self-contained C++17 program reading from stdin and
writing to stdout. The scaffold fixes the input/output shape and value types;
fill in the query-processing logic.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (cin >> n >> q) {
        vector<long long> a(n);
        for (int i = 0; i < n; ++i) {
            cin >> a[i];
        }

        // TODO:
        (void)a;

        for (int i = 0; i < q; ++i) {
            int t;
            cin >> t;
            long long answer = 0;
            if (t == 1) {
                int l, r, k;
                cin >> l >> r >> k;
                (void)l;
                (void)r;
                (void)k;
            } else {
                int l, r;
                long long x;
                cin >> l >> r >> x;
                (void)l;
                (void)r;
                (void)x;
            }
            cout << answer << '\n';
        }
    }
}
```
