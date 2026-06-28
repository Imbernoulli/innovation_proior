# Online range-rank and range-quantile on a static array

## Research question

You are given a static array `a[1..n]` of nonnegative integers. You must answer `q` queries of
two kinds, where every query is restricted to a contiguous index range `[l, r]`:

- **Type 1 (range-rank):** given `l, r, x`, report how many positions `i` with `l <= i <= r`
  satisfy `a[i] <= x`.
- **Type 2 (range-quantile):** given `l, r, k`, report the `k`-th smallest value among
  `a[l], a[l+1], ..., a[r]` (`1`-indexed, so `k = 1` is the minimum of the range).

The catch that defines this problem: the queries are **online**. Each query's three integer
parameters arrive XOR-encoded with the previous answer, so you cannot read all the queries up
front, sort them, and sweep — you are forced to answer query `t` before you can even decode the
parameters of query `t+1`. The array itself never changes.

This is the core "range rank / quantile" primitive that sits underneath order-statistics queries,
median-of-range problems, and many computational-geometry counting tasks. Getting the *online*
version right — at `n, q` up to `2*10^5` — is what separates a structure that merely works offline
from one that answers each query independently in logarithmic time.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `q` (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`).
  - The second line has `n` integers `a[1..n]` (`0 <= a[i] <= 10^9`).
  - Each of the next `q` lines has four integers `type A B C`. Let `last` be the answer printed
    for the previous query (`last = 0` before the first query). Decode:
    `l = A XOR last`, `r = B XOR last`, and the third parameter
    (`x` for type 1, `k` for type 2) `= C XOR last`.
  - After decoding it is guaranteed that `1 <= l <= r <= n`; for type 2, additionally
    `1 <= k <= r - l + 1`. For type 1, the decoded `x` may be any integer (it can fall below `0`
    or above every array value). Every answer this problem produces is nonnegative, so the running
    XOR key `last` is always well defined.
- Output (stdout): for each query, one line with its integer answer.
- Time limit: 1 second. Memory: 256 MB.

### Worked sample

Input:

```
8 6
2 7 1 8 2 8 1 8
1 1 8 5
2 5 12 7
1 1 4 0
2 6 5 0
1 3 10 10
2 10 13 12
```

Output:

```
4
2
2
2
8
8
```

Decoding trace (`last` starts at `0`):

- `1 1 8 5` -> `last=0` -> type 1, `l=1, r=8, x=5`: values `<= 5` in the whole array
  `{2,7,1,8,2,8,1,8}` are `2,1,2,1` -> **4**. Now `last = 4`.
- `2 5 12 7` -> XOR by `4` -> type 2, `l=1, r=8, k=3`: sorted array is
  `1,1,2,2,7,8,8,8`, the 3rd smallest is **2**. Now `last = 2`.
- `1 1 4 0` -> XOR by `2` -> type 1, `l=3, r=6, x=2`: subarray `{1,8,2,8}`, values `<= 2`
  are `1,2` -> **2**. Now `last = 2`.
- `2 6 5 0` -> XOR by `2` -> type 2, `l=4, r=7, k=2`: subarray `{8,2,8,1}` sorted `1,2,8,8`,
  2nd smallest is **2**. Now `last = 2`.
- `1 3 10 10` -> XOR by `2` -> type 1, `l=1, r=8, x=8`: every value is `<= 8` -> **8**.
  Now `last = 8`.
- `2 10 13 12` -> XOR by `8` -> type 2, `l=2, r=5, k=4`: subarray `{7,1,8,2}` sorted `1,2,7,8`,
  4th smallest is **8**. Now `last = 8`.

## Background

Two angles are worth weighing before committing to a structure.

- **Offline sweep with a Fenwick tree (BIT).** If all queries were known in advance, range-rank is
  textbook: sort the queries, sweep `x` (or sweep the array by value), and maintain a BIT over
  indices so each "count of positions `<= x` in `[l, r]`" becomes two prefix queries. This is
  `O((n + q) log n)` and very fast in practice. Its hard requirement is that the *entire* query
  list be available so it can be reordered. The XOR-encoding deliberately removes that: query `t+1`
  cannot even be parsed until query `t` is answered, so no global reordering exists. The offline
  BIT is therefore inapplicable, not merely slower.

- **Merge-sort tree (segment tree of sorted lists).** Build a segment tree where each node stores
  the sorted multiset of its segment. A range-rank is then `O(log^2 n)` (a `log n` decomposition,
  each node binary-searched). This *is* online and correct, but range-quantile on top of it needs a
  binary search over the value domain wrapping the range-rank, pushing quantiles to `O(log^3 n)`,
  and the structure costs `O(n log n)` space with heavy constants.

The question is whether a single structure can answer **both** rank and quantile **online** in a
clean `O(log sigma)` per query, where `sigma` is the number of distinct values.

## Evaluation settings

Judged on hidden tests covering: arrays with massive value ties (forcing coordinate compression),
strictly increasing arrays (quantiles hit every leaf of the value domain), `x` below the minimum
and at/above the maximum, single-element arrays (`n = 1`), full-range queries `[1, n]`, and the
maximum scale `n = q = 2*10^5` with values up to `10^9`. Correctness is exact (every line must
match); the online XOR-encoding is enforced, so any attempt to batch or reorder queries fails.

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
    for (auto &v : a) cin >> v;

    long long last = 0;
    for (int t = 0; t < q; ++t) {
        long long type, A, B, C;
        cin >> type >> A >> B >> C;
        long long l = A ^ last, r = B ^ last, c = C ^ last;

        // TODO: answer online -- type 1 = count of a[l..r] <= c (rank);
        //                        type 2 = k-th smallest of a[l..r] with k = c (quantile).
        long long ans = 0;

        cout << ans << "\n";
        last = ans;
    }
    return 0;
}
```
