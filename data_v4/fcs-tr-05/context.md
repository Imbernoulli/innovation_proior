# Path XOR-assign and path sum on a tree

## Research question

You are given a tree with `n` nodes (numbered `1..n`). Node `i` carries a non-negative
integer value `v[i]` with `0 <= v[i] < 2^30`. You must process `q` online operations of
two kinds, each acting on the **unique simple path** between two nodes `u` and `v`
(inclusive of both endpoints, and of the LCA):

- `1 u v x` — for **every** node `w` on the path from `u` to `v`, replace `v[w]` by
  `v[w] XOR x` (a path XOR-assign update), with `0 <= x < 2^30`.
- `2 u v` — output the **sum** of `v[w]` over every node `w` on the path from `u` to `v`.

The task is to answer every type-2 query. This is the path case of an aggregate
data-structure problem on trees: updates and queries are both along arbitrary tree paths,
and the update operator (bitwise XOR) is not the usual "add a constant", so the
aggregate (a plain integer sum) and the update do not commute in the naive way.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `q`
    (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`).
  - The second line has `n` integers `v[1] ... v[n]` (`0 <= v[i] < 2^30`).
  - The next `n-1` lines each have two integers `a b` (`1 <= a, b <= n`), an edge of the
    tree. The edges describe a connected acyclic graph on the `n` nodes.
  - The next `q` lines each describe one operation:
    - `1 u v x` for a path XOR-assign (`1 <= u, v <= n`, `0 <= x < 2^30`), or
    - `2 u v` for a path-sum query (`1 <= u, v <= n`).
- Output (stdout): for each type-2 query, a single line containing the path sum.
- Time limit: 2 seconds. Memory: 256 MB.

Example. Tree edges `1-2`, `1-3`, `2-4`; initial values `v = [5, 3, 6, 1]`
(so `v[1]=5, v[2]=3, v[3]=6, v[4]=1`). Operations:

```
2 4 3      -> path 4-2-1-3, sum 1+3+5+6 = 15
1 2 3 7    -> XOR 7 onto path 2-1-3: v=[5^7, 3^7, 6^7, 1]=[2,4,1,1]
2 4 3      -> path 4-2-1-3, sum 1+4+2+1 = 8
1 3 4 1    -> XOR 1 onto path 3-1-2-4: v=[3,5,0,0]
2 4 4      -> path is the single node 4, sum 0
```

Output:

```
15
8
0
```

## Background

Two ingredients have to be combined, and each, taken alone, is standard:

- **Path decomposition.** A query/update on an arbitrary tree path can be reduced to a set
  of operations on contiguous segments of a linearised array if we lay the tree out by
  **heavy-light decomposition (HLD)**. HLD writes the tree into one array so that every
  root-to-node path — and hence every `u`-`v` path, split at the LCA — is covered by
  `O(log n)` contiguous index ranges. A range data structure then handles each range.
- **A range structure under XOR-assign.** On a flat array we need a structure that supports
  "XOR a constant `x` onto a contiguous range" and "sum a contiguous range". A segment tree
  with lazy propagation is the natural home, but the lazy tag must compose with the
  aggregate. With a *sum* aggregate, an additive update is trivial; the XOR update is the
  part that is not obvious, because `sum(a XOR x)` is not `sum(a) +/- (something simple)`.

The interaction between these two — what exactly the segment tree stores so that a lazy XOR
tag both updates the stored sum and pushes correctly to children — is the heart of the
problem. The naive route of LCA + prefix sums (which works when there are no updates, or
when updates are additive and you keep a Fenwick/BIT on Euler tour) does not directly carry
the XOR-assign update, so a different invariant is needed.

## Evaluation settings

Judged on hidden tests covering: single-node trees (`n = 1`); two-node trees; long paths
(a degenerate chain) and stars (a hub with many leaves), which stress the two extremes of
HLD chain structure; queries where `u = v` (a single-node path); XOR by `0` (a no-op
update) and XOR by the full 30-bit mask; double XOR by the same value (must cancel); and
large random trees with `n = q = 2*10^5` and values near `2^30` (so a path sum can reach
about `2*10^5 * 2^30 ~= 2.1*10^14`, well past 32 bits — every accumulator must be 64-bit).

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

    vector<long long> v(n + 1);
    for (int i = 1; i <= n; i++) cin >> v[i];

    // read n-1 tree edges
    for (int i = 0; i < n - 1; i++) {
        int a, b;
        cin >> a >> b;
        // TODO: store adjacency
    }

    // TODO: build a structure that supports
    //   path XOR-assign (op type 1) and path sum (op type 2)
    // and answer every type-2 query.

    for (int i = 0; i < q; i++) {
        int type;
        cin >> type;
        if (type == 1) {
            int u, vv, x;
            cin >> u >> vv >> x;
            // TODO: XOR x onto every node on path(u, vv)
        } else {
            int u, vv;
            cin >> u >> vv;
            // TODO: output sum of values on path(u, vv)
        }
    }
    return 0;
}
```
