# Mo's algorithm

## Problem

Given a static array `a[0..n-1]` and inclusive queries `(l, r)`, return the
number of distinct values in `a[l..r]` for each query, in the original query
order.

## Method

Maintain an inclusive range `[cur_l, cur_r]`, a frequency table `cnt`, and a
running count `distinct`. When index `i` enters the range, increment
`cnt[a[i]]`; if the count becomes `1`, increment `distinct`. When index `i`
leaves the range, decrement `cnt[a[i]]`; if the count becomes `0`, decrement
`distinct`. Each endpoint step costs `O(1)` expected time.

Choose `block = max(1, floor(sqrt(n)))`. Sort query indices lexicographically by

```text
(l // block, r)
```

where `(l, r)` is the query for that index. Within one left-endpoint block, the
right endpoint moves monotonically except for the reset between blocks, giving
`O(n^2 / block)` right-endpoint movement. The left endpoint moves `O(block)` per
query, plus lower-order cross-block movement, giving `O(q * block + n)`. With
`block = sqrt(n)`, total pointer movement is `O((n + q) sqrt(n))`; sorting costs
`O(q log q)`.

## Code

Single-file C++17. Reads from stdin: `n`, then `n` array values, then `q`, then
`q` query pairs `l r` (1-based inclusive). Prints one distinct-count per query,
in the original query order.

```cpp
// Mo's algorithm for offline distinct-value range queries.
// Reads from stdin: n, then n array values, then q, then q query pairs (l r),
// 1-based inclusive. Prints, one per line in original query order, the number of
// distinct values in a[l..r].
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) cin >> a[i];

    // Coordinate-compress values so cnt can be a dense array, valid for
    // arbitrary integer values (the dictionary-backed count in the reference).
    vector<int> sorted_vals(a.begin(), a.end());
    sort(sorted_vals.begin(), sorted_vals.end());
    sorted_vals.erase(unique(sorted_vals.begin(), sorted_vals.end()),
                      sorted_vals.end());
    for (int i = 0; i < n; ++i)
        a[i] = int(lower_bound(sorted_vals.begin(), sorted_vals.end(), a[i]) -
                   sorted_vals.begin());

    int q;
    cin >> q;
    vector<int> ql(q), qr(q), order(q);
    for (int i = 0; i < q; ++i) {
        int l, r;
        cin >> l >> r;             // 1-based inclusive
        ql[i] = l - 1;
        qr[i] = r - 1;
        order[i] = i;
    }

    int block = max(1, (int)sqrt((double)n));
    sort(order.begin(), order.end(), [&](int x, int y) {
        if (ql[x] / block != ql[y] / block) return ql[x] / block < ql[y] / block;
        return qr[x] < qr[y];
    });

    vector<int> cnt(max(1, (int)sorted_vals.size()), 0);
    int distinct = 0;
    vector<int> answers(q);

    auto add = [&](int i) {
        if (++cnt[a[i]] == 1) ++distinct;
    };
    auto remove = [&](int i) {
        if (--cnt[a[i]] == 0) --distinct;
    };

    int cur_l = 0, cur_r = -1;
    for (int idx : order) {
        int l = ql[idx], r = qr[idx];
        while (cur_l > l) add(--cur_l);
        while (cur_r < r) add(++cur_r);
        while (cur_l < l) remove(cur_l++);
        while (cur_r > r) remove(cur_r--);
        answers[idx] = distinct;
    }

    string out;
    for (int i = 0; i < q; ++i) {
        out += to_string(answers[i]);
        if (i + 1 < q) out += '\n';
    }
    cout << out;
    return 0;
}
```

## Complexity

Time is `O(q log q + (n + q) sqrt(n))`. Space is `O(q + d)`, where `d` is the
number of distinct values in `a`.
