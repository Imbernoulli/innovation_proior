# Context

## Problem

Given a static array of `n` integers and `q` queries `(l, r, k)`, report the
`k`-th smallest value in `a[l..r]`. The array is fixed after input; queries do
not update it. Indices are 1-based, `1 <= l <= r <= n`, and `k` is 1-based with
`1 <= k <= r - l + 1`. Equal values count with multiplicity.

## Research question / Input-output contract

The deliverable is a single self-contained C++17 program that reads from
standard input and writes to standard output. Input starts with two integers
`n` and `q`, followed by `n` array values. Then follow `q` queries, each given as
three integers `l`, `r`, and `k`. For every query, print the `k`-th smallest
value in `a[l..r]` on its own line.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<long long> a(n);
    for (int i = 0; i < n; ++i) {
        cin >> a[i];
    }

    struct Query {
        int l;
        int r;
        long long k;
    };

    vector<Query> queries(q);
    for (int i = 0; i < q; ++i) {
        cin >> queries[i].l >> queries[i].r >> queries[i].k;
    }

    vector<long long> answers(q);
    // TODO: implement query processing.

    for (int i = 0; i < q; ++i) {
        cout << answers[i] << '\n';
    }

    return 0;
}
```
