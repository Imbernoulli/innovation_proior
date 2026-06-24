# Best contiguous-block sum on a range, with point updates

## Research question

A logistics dashboard tracks the hourly net cash flow of a single warehouse as a row of `n`
integers `a[1..n]` (a value can be negative: a loss). Two kinds of requests arrive online:

- **Update** `1 p v`: the net flow at hour `p` is corrected to `v`.
- **Query** `2 l r`: over the hour window `[l, r]`, report the largest total net flow obtainable by
  picking a **contiguous** block of hours inside that window. You may also pick **no** hours at all
  (an empty block), whose total is `0`, so the answer is never negative.

In other words, each query asks for the maximum-sum contiguous subarray restricted to `a[l..r]`,
where the empty subarray (sum `0`) is allowed, and updates change single entries between queries.
This is the canonical "maximum subarray on a range with point updates" task; getting the segment-tree
merge exactly right — and resisting the wrong but tempting greedy — is the whole point.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `q`
  (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`). The second line has `n` integers
  `a[i]` (`-10^9 <= a[i] <= 10^9`). Then `q` lines, each either
  - `1 p v` with `1 <= p <= n` and `-10^9 <= v <= 10^9` (point update), or
  - `2 l r` with `1 <= l <= r <= n` (query).
- Output (stdout): for every query of type `2`, one line with the maximum contiguous-block sum
  in `a[l..r]` (empty block allowed, so the answer is `>= 0`).
- Time limit: 2 seconds. Memory: 256 MB.

Example. For the array `[3, -2, 5, -1, -6, 4, 2]`:

```
7 5
3 -2 5 -1 -6 4 2
2 1 7
2 5 7
1 5 4
2 1 7
2 5 7
```

produces

```
6
6
15
10
```

The first query over the whole array: the best contiguous block is `[3,-2,5]` (or `[4,2]`), sum `6`
— note that the positive values `3,5,4,2` sum to `14` but they are **not** contiguous. After the
update `a[5] = 4` the array is `[3,-2,5,-1,4,4,2]`, and `[3,-2,5,-1,4,4,2]` itself sums to `15`.

## Background

Each query is a maximum-subarray (Kadane) problem, but confined to an arbitrary window and
interleaved with point updates, so a per-query linear scan is `O(nq)` and too slow. Two routes are
on the table before committing:

- **Greedy: sum the positives in the window.** For a query `[l, r]`, add up every `a[i] > 0`. It is
  trivial and `O(window)` per query (or `O(log n)` with a positive-sum Fenwick tree). The open
  question is whether "take all the gains, drop all the losses" is even a valid block — the answer
  must be a single *contiguous* run, and a positive can be marooned behind a deep negative.
- **Segment tree of merged statistics.** Store at each node four numbers — total sum, best prefix,
  best suffix, best inner block — and merge children in `O(1)`. Build is `O(n)`, each update and each
  query is `O(log n)`. The open question is the exact merge formula and how the empty-block rule is
  encoded so all four fields stay consistent.

## Evaluation settings

Judged on hidden tests covering: all-positive windows (greedy looks right here), windows whose only
gains are split by a large loss (greedy fails), all-negative windows (answer `0`), single-element
windows and `n = 1`, heavy update/query interleavings, and large `n = q = 2*10^5` with values near
`10^9` so a window sum can reach `2*10^14` and overflow 32 bits.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: build a segment tree supporting point updates and, per query [l,r],
    //       the maximum-sum contiguous block (empty block allowed, answer >= 0).

    for (int i = 0; i < q; i++) {
        int type; cin >> type;
        if (type == 1) {
            int p; long long v; cin >> p >> v;
            // apply update
        } else {
            int l, r; cin >> l >> r;
            // print answer
        }
    }
    return 0;
}
```
