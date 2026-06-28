# Functional-graph cycle arithmetic: where am I after t steps?

## Research question

You are given a **functional graph** on `n` nodes labelled `0..n-1`: every node `i` has exactly one
out-edge `f(i)`. Starting from a node `s` and repeatedly following the out-edge, after `t` steps you
land on some node; write that node as `f^t(s)` (the `t`-fold composition of `f` applied to `s`). You
must answer `q` independent queries, each asking for `f^t(s)` where `t` can be as large as `10^18`.

The point of interest is the scale of `t`. You cannot simulate `10^18` steps, and at `n, q <= 2*10^5`
you also cannot afford a per-query data structure that is wasteful in memory. The question is what
structure of the functional graph lets every query be answered essentially in constant time after a
linear-time preprocessing pass.

## Input / output contract

- Input (stdin):
  - line 1: integer `n` (`1 <= n <= 2*10^5`);
  - line 2: `n` integers `f(0) f(1) ... f(n-1)`, each in `[0, n-1]` (the out-edges);
  - line 3: integer `q` (`1 <= q <= 2*10^5`);
  - next `q` lines: two integers `s` and `t` per line (`0 <= s <= n-1`, `0 <= t <= 10^18`).
- Output (stdout): `q` lines; line `i` is the node `f^t(s)` for the `i`-th query (`t = 0` means the
  start node itself).
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
6
1 2 0 2 3 4
4
5 0
5 3
5 4
5 1000000000000000000
```

Here `f = [1,2,0,2,3,4]`. Nodes `0->1->2->0` form a 3-cycle; node `3->2`, `4->3`, `5->4` form a tail
that feeds into the cycle at node `2`. From node `5`: `0` steps stays at `5`; `3` steps reach
`5->4->3->2`, landing on `2` (the cycle entry); `4` steps reach `0`; and `10^18` steps land somewhere
on the cycle. The expected output is:

```
5
2
0
0
```

## Background

A functional graph (out-degree exactly 1 everywhere) has a rigid shape. Following edges from any
node is a deterministic walk on a finite set, so it must eventually repeat a node, and from the first
repeat onward it loops forever. Each weakly-connected component therefore looks like a **"rho" (ρ)**:
a number of trees ("tails") whose edges all point toward a single directed **cycle**. Once the walk
reaches the cycle it circulates with a fixed period equal to the cycle length.

Two families of approach are on the table before committing to one:

- **Binary lifting / jump pointers.** Precompute `up[k][v] = f^{2^k}(v)` for `k` up to about `60`
  (since `2^60 > 10^18`). A query then decomposes `t` into its binary digits and follows `O(log t)`
  jumps. This is the textbook "k-th successor" structure; the open question is its cost: the table is
  `n * 60` entries, around `1.2*10^7` 32-bit integers, and each query is `O(log t)`.
- **Rho decomposition with modular arithmetic on the cycle.** Find each node's cycle and its distance
  to that cycle (the "tail depth"), then split a query into a bounded tail part plus a *modular* jump
  around the cycle. The open question is exactly how to make the bounded tail part also `O(1)` per
  query without rebuilding binary lifting.

## Evaluation settings

Judged on hidden tests covering: a single self-loop (`n = 1`); one large cycle covering all nodes;
a single very long chain feeding into a self-loop (maximal tail depth); many disjoint rho components;
queries with `t = 0`, with `t` exactly equal to the tail depth (landing on the cycle entry), with `t`
just below the depth (staying in the tail), and with `t = 10^18` (deep into the cyclic regime); and
the largest case `n = q = 2*10^5`. Node answers must be exact.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> f(n);
    for (int i = 0; i < n; i++) cin >> f[i];

    int q;
    cin >> q;
    vector<int> qs(q);
    vector<long long> qt(q);
    for (int i = 0; i < q; i++) cin >> qs[i] >> qt[i];

    // TODO: decompose the functional graph into cycles + tails, then answer
    // each query f^t(s) using the rho structure.

    for (int i = 0; i < q; i++) {
        cout << 0 << "\n";
    }
    return 0;
}
```
