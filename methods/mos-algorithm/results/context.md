# Context

## Problem

Given a static array of `n` integers and `q` offline queries `(l, r)`, report the
number of distinct values in `a[l..r]`. (`n, q` up to ~10^5 / 10^6.)

All `q` queries are available before any answer has to be printed, and the output
must still follow the original query order. The deliverable is a single
self-contained C++17 program that reads from stdin and writes to stdout. The
input query endpoints are 1-based inclusive; the program may convert them to
0-based inclusive indices internally.

## Code framework

The program reads `n`, then the `n` array values, then `q`, then `q` query pairs
`l r` from stdin. It prints one integer per query to stdout, in the same order as
the queries were given.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) cin >> a[i];

    int q;
    cin >> q;
    vector<int> ql(q), qr(q);
    for (int i = 0; i < q; ++i) {
        int l, r;
        cin >> l >> r;
        ql[i] = l - 1;
        qr[i] = r - 1;
    }

    vector<int> answers(q);

    // TODO: implement the solution

    string out;
    for (int i = 0; i < q; ++i) {
        out += to_string(answers[i]);
        if (i + 1 < q) out += '\n';
    }
    cout << out;
    return 0;
}
```
