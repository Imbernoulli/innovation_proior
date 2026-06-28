# Online subset-XOR queries: maximum, membership, and k-th smallest

## Research question

You maintain a growing collection of non-negative integers under four kinds of operations, processed
online (each must be answered before the next is read):

- `1 x` — add the integer `x` to the collection.
- `2` — report the **maximum** value obtainable as the XOR of some subset of the current collection
  (the empty subset gives `0`, so the answer is at least `0`).
- `3 x` — report whether `x` is **representable** as the XOR of some subset of the current
  collection (`YES` / `NO`). The empty subset XORs to `0`, so `0` is always representable.
- `4 k` — report the **k-th smallest distinct value** (1-indexed) among all values obtainable as a
  subset-XOR. The smallest such value is always `0` (empty subset). If `k` exceeds the number of
  distinct obtainable values, report `-1`.

Every value fits in 60 bits. The collection only grows; numbers are never removed. The challenge is
that all three query families must be answered together, repeatedly, as the collection changes — so
the data structure has to support insertion, an extremal query, a membership query, and an order
query, all efficiently.

## Input / output contract

- Input (stdin): the first token is `q` (`0 <= q <= 3*10^5`), the number of operations. Then `q`
  operations follow, one per line in the formats above. All `x` satisfy `0 <= x < 2^60`. For
  operation `4`, `k` satisfies `1 <= k <= 2^60`.
- Output (stdout): for each operation of type `2`, `3`, or `4`, one line with its answer (a
  non-negative integer for type `2`; `YES`/`NO` for type `3`; an integer, possibly `-1`, for
  type `4`). Operations of type `1` produce no output.
- Time limit: 1 second. Memory: 256 MB.

Example:

```
9
1 3
1 5
2
3 6
3 4
4 1
4 3
4 4
4 5
```

After inserting `3` and `5`, the obtainable subset-XOR values are `{0, 3, 5, 6}`. So `2` prints `6`
(the max). `3 6` is `YES` (`3 xor 5 = 6`); `3 4` is `NO`. Sorted ascending the distinct values are
`[0, 3, 5, 6]`, so `4 1 -> 0`, `4 3 -> 5`, `4 4 -> 6`, and `4 5 -> -1` (only four distinct values).
Output:

```
6
YES
NO
0
5
6
-1
```

## Background

The set of values reachable as subset-XORs is exactly the linear span of the inserted numbers over
the field `GF(2)`, where each 60-bit number is a vector in `GF(2)^60` and XOR is vector addition.
That reframing is what makes the three query families tractable at once:

- A naive enumeration of all `2^m` subsets (for `m` inserted numbers) answers every query but is
  exponential — only viable for `m` up to about 20, far below the constraints here.
- Sorting the numbers and greedily grabbing the largest is a tempting answer for the max query, but
  "largest value first" is not the same as "highest leading bit first," and it gives no help at all
  for membership or for the k-th-smallest query.

The intended structure is a **linear basis** over `GF(2)` — the span's basis maintained by online
Gaussian elimination — from which the max query is a high-bit-first greedy walk, membership is a
reduction-to-zero test, and the k-th-smallest query reads `k` in binary against a row-reduced basis.

## Evaluation settings

Judged on hidden tests covering: queries issued before any insertion (empty span), dependent and
duplicate insertions (rank does not grow), membership of `0` and of out-of-span values, k-th-smallest
at the boundaries (`k = 1`, `k = 2^rank`, and `k` just past the range giving `-1`), values using the
top bit (bit 59), and large instances (`q = 3*10^5`) including adversarial patterns that interleave
insertions with order queries.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    if (!(cin >> q)) return 0;

    for (int i = 0; i < q; ++i) {
        int type;
        cin >> type;
        if (type == 1) {
            unsigned long long x; cin >> x;
            // TODO: add x to the collection
        } else if (type == 2) {
            // TODO: output maximum subset-XOR
        } else if (type == 3) {
            unsigned long long x; cin >> x;
            // TODO: output YES if x is representable as a subset-XOR, else NO
        } else { // type == 4
            unsigned long long k; cin >> k;
            // TODO: output the k-th smallest distinct subset-XOR value, or -1
        }
    }
    return 0;
}
```
