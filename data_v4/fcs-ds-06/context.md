# Dynamic 2D rectangle-sum on a known point set (forced online)

## Research question

You are given `n` points in the plane. Point `i` has integer coordinates `(x_i, y_i)` and an integer
weight `w_i` (which may be negative). The set of point *locations* is fixed for the whole run, but the
weights change over time. You must process `q` operations of two kinds, **in order**:

- **Update** `1 i d`: add `d` to the weight of point `i` (`0`-indexed).
- **Query** `2 X1 Y1 X2 Y2`: report the sum of the current weights of all points whose coordinates
  satisfy `X1 <= x <= X2` and `Y1 <= y <= Y2` (an axis-aligned closed rectangle).

The queries are **forced online**: the four rectangle coordinates of every query are XOR-encoded with
the answer to the previous query, so you cannot read all queries, sort them, and answer offline — each
query's real coordinates are only known once you have produced the previous answer. Let `lastAns` be the
answer printed for the most recent query (and `lastAns = 0` before the first query). A query line
`2 A B C D` decodes to `X1 = A xor lastAns`, `Y1 = B xor lastAns`, `X2 = C xor lastAns`,
`Y2 = D xor lastAns`.

This is the dynamic version of 2D dominance / rectangle counting: the classic offline trick (sort
events, sweep one axis with a Fenwick tree over the other) is unavailable because the next query is
hidden behind the current answer.

## Input / output contract

- Input (stdin):
  - Line 1: two integers `n q` (`1 <= n <= 10^5`, `1 <= q <= 10^5`).
  - Next `n` lines: `x_i y_i w_i` (`-10^9 <= x_i, y_i <= 10^9`, `-10^9 <= w_i <= 10^9`). These are the
    initial weights; point `i` is the `i`-th of these lines, `0`-indexed.
  - Next `q` lines: each is either `1 i d` (`0 <= i < n`, `-10^9 <= d <= 10^9`) or
    `2 A B C D` (an encoded query as described above).
- Output (stdout): for each query (type `2`), one line containing the rectangle-sum answer.
- A query whose decoded rectangle is empty (`X1 > X2` or `Y1 > Y2`) has answer `0`.
- Time limit: 1 second. Memory: 256 MB.

Worked example:

```
1 2
2 2 5
2 0 0 3 3
2 5 5 6 6
```

The single point is `(2,2)` with weight `5`. The first query (`lastAns = 0`) decodes to the rectangle
`[0,3] x [0,3]`, which contains the point, so the answer is `5`. Now `lastAns = 5`, and the second
query `2 5 5 6 6` decodes to `X1 = 5 xor 5 = 0`, `Y1 = 0`, `X2 = 6 xor 5 = 3`, `Y2 = 3` — again the
rectangle `[0,3] x [0,3]`, answer `5`. Output:

```
5
5
```

## Background

Sums over an axis-aligned rectangle on a static, offline workload are a solved problem: compress
coordinates, sweep the `x` axis in sorted order, and keep a Fenwick tree (BIT) over `y`; each query
becomes two prefix events `(X2, ...) - (X1-1, ...)`, each split again over `y`. The whole thing is
`O((n+q) log n)` and needs only a one-dimensional BIT.

Two things here break that template. First, weights change, so a query at time `t` must see exactly the
updates that happened before it — a precomputed prefix structure over a fixed weight vector is wrong.
Second, and decisively, the **forced-online** encoding means a query's rectangle depends on the previous
answer, so there is no legal order in which to process events other than the given one. We are forced to
keep a structure that supports, interleaved and online, both a point weight-update and a rectangle-sum.

The coordinate *locations* never change, though — every `(x_i, y_i)` that will ever be touched is known
from the start. That is the structural fact the intended solution exploits.

## Evaluation settings

Judged on hidden tests covering: all-positive weights; mixed and negative weights (sums that cancel to
zero or go negative); duplicate coordinates (several points sharing an `(x, y)`); points and queries
with coordinates near `+-10^9`; weight sums exceeding the 32-bit range (so accumulators must be 64-bit);
empty decoded rectangles; queries that touch every point (full-plane sums) at `n = q = 10^5`; and runs
that are update-heavy versus query-heavy. Correctness is exact (the judge compares the full output
stream byte-for-byte after the contract's line format).

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

    vector<long long> px(n), py(n), pw(n);
    for (int i = 0; i < n; i++) cin >> px[i] >> py[i] >> pw[i];

    long long lastAns = 0;
    for (int t = 0; t < q; t++) {
        int type; cin >> type;
        if (type == 1) {
            long long i, d; cin >> i >> d;
            // TODO: apply the weight delta to point i
        } else {
            long long A, B, C, D; cin >> A >> B >> C >> D;
            long long X1 = A ^ lastAns, Y1 = B ^ lastAns;
            long long X2 = C ^ lastAns, Y2 = D ^ lastAns;
            // TODO: compute the rectangle sum over [X1,X2] x [Y1,Y2]
            long long ans = 0;
            lastAns = ans;
            cout << ans << "\n";
        }
    }
    return 0;
}
```
