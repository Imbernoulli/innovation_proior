# Counting star clusters after range-union bridge builds

## Research question

There are `n` stars in a line, numbered `1..n`. You are given `m` bridge-building operations. Each
operation is a pair `(l, r)` with `1 <= l <= r <= n` and means **build bridges so that every star in
the inclusive range `l, l+1, ..., r` becomes mutually connected**. Bridges are permanent and
operations may overlap, repeat, or be single points (`l == r`, which connects nothing new).

After all `m` operations have been applied, the stars partition into connected clusters (two stars are
in the same cluster if there is a chain of bridges between them; a star touched by no operation is its
own singleton cluster). **Output the number of connected clusters.**

This is the "union an entire contiguous range" flavour of disjoint-set union. The subtlety is entirely
on the boundary: a range `[l, r]` introduces exactly `r - l` adjacency links (between `i` and `i+1`
for `l <= i < r`), so a one-element range adds zero links, and ranges that merely *touch* at an
endpoint (`[1,3]` then `[3,5]`) merge while ranges separated by a gap of one (`[1,3]` then `[4,6]`) do
not. Getting the inclusive/exclusive boundary of the union loop exactly right is the whole game.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`). Then `m` lines follow, each with two integers `l` and `r`
  (`1 <= l <= r <= n`) describing one range-union operation.
- Output (stdout): a single line with the number of connected clusters after all operations.
- Time limit: 1 second. Memory: 256 MB.

Example: with `n = 8` and operations `(2,4), (4,6), (7,7)`, the answer is `4`. Operation `(2,4)` joins
`{2,3,4}`; `(4,6)` joins `{4,5,6}` and so merges with the previous group into `{2,3,4,5,6}`; `(7,7)`
is a single point and joins nothing. The clusters are `{1}`, `{2,3,4,5,6}`, `{7}`, `{8}` — four of them.

## Background

The naive way to apply one operation `(l, r)` is to union each adjacent pair in the range — `unite(l,
l+1)`, `unite(l+1, l+2)`, ..., `unite(r-1, r)` — which is correct but costs `O(r - l)` per operation,
so total cost is `O(sum of range lengths)`. With `m` ranges each of length up to `n` that is
`O(n*m) = 4*10^{10}` link attempts, far too slow.

Two ideas are on the table before committing:

- **Plain DSU, link every pair in every range.** Trivially correct, easy to write, but quadratic in
  the worst case (many long overlapping ranges). It will time out at the stated bounds; it is only
  useful as a reference oracle on small inputs.
- **DSU with a "next unconsumed index" skip pointer.** Keep a second pointer structure `nxt[]` so
  that once stars `i` and `i+1` have been linked, future range-walks jump straight past `i` instead
  of re-touching it. Each link is then created at most once across the entire run, giving near-linear
  total time. The open question is the exact loop boundary: the walk must add links up to and
  including `(r-1) -> r` but must never create the out-of-range link `r -> r+1`.

## Evaluation settings

Judged on hidden tests covering: single-point ranges (`l == r`) that must connect nothing; ranges that
touch at a shared endpoint (must merge) versus ranges separated by a one-index gap (must not merge);
fully overlapping and repeated ranges; the no-operation case (`m = 0`, answer `n`); `n = 1`; and large
adversarial inputs with `n = m = 2*10^5` and many long overlapping ranges (where a quadratic solution
times out and the skip pointer is required).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // DSU over stars 1..n.
    // TODO: apply each range-union (l, r) efficiently using a skip pointer,
    //       then count the connected clusters among stars 1..n.

    int comps = 0;

    cout << comps << "\n";
    return 0;
}
```
