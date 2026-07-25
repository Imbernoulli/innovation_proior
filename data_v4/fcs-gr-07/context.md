# Functional-graph cycle arithmetic: where am I after t steps?

## Research question

You are given a **functional graph** on `n` nodes labelled `0..n-1`: every node `i` has exactly one
out-edge `f(i)`. Starting from a node `s` and repeatedly following the out-edge, after `t` steps you
land on some node; write that node as `f^t(s)` (the `t`-fold composition of `f` applied to `s`). You
must answer `q` independent queries, each asking for `f^t(s)` where `t` can be as large as `10^18`.

The point of interest is the scale of `t`. You cannot simulate `10^18` steps, and at `n, q <= 2*10^5`
you also cannot afford a per-query data structure that is wasteful in memory.

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

## Evaluation settings

Judged on hidden tests covering: a single self-loop (`n = 1`); one large cycle covering all nodes;
a single very long chain feeding into a self-loop; many disjoint components; queries with `t = 0`,
with `t` exactly equal to the tail depth, with `t` just below the depth, and with `t = 10^18`; and
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

    // TODO: answer each query f^t(s) efficiently, given t up to 10^18.

    for (int i = 0; i < q; i++) {
        cout << 0 << "\n";
    }
    return 0;
}
```
