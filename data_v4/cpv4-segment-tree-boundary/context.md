# Longest strictly-increasing run inside a window, with point updates

## Research question

You maintain an array `a[1..n]` of integers. You must support two kinds of operations, interleaved
and online:

- **Set** `1 p x` — assign `a[p] = x`.
- **Query** `2 l r` — among the positions `l, l+1, ..., r` (inclusive, `1 <= l <= r <= n`), report
  the **length of the longest contiguous block that is strictly increasing and lies entirely inside
  `[l, r]`**. Formally, the largest `len` for which there is a start `s` with `l <= s` and
  `s + len - 1 <= r` such that `a[s] < a[s+1] < ... < a[s+len-1]`.

A single position is a valid block of length `1`, so every query answer is at least `1`.

This is the canonical "structural segment tree" task: each node summarizes a contiguous block, and a
parent is reconstructed from its two children by gluing across the seam between them. The whole
difficulty is the seam — both inside the tree (when merging children) and at the query window edges
(`l` and `r`), where a run that continues *outside* the window must be clipped. It is exactly the
setting where inclusive/exclusive boundary errors hide.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `q`
  (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`). The second line has `n` integers `a[1..n]`
  (`-10^9 <= a[i] <= 10^9`). Then `q` lines follow, each either
  `1 p x` (`1 <= p <= n`, `-10^9 <= x <= 10^9`) or `2 l r` (`1 <= l <= r <= n`).
- Output (stdout): for each query of type `2`, one line with the requested length.
- Time limit: 2 seconds. Memory: 256 MB.

Example input:

```
8 5
1 3 2 4 5 6 1 7
2 1 8
2 2 6
1 7 8
2 5 8
2 3 5
```

Example output:

```
4
4
3
3
```

Walkthrough: the array is `1 3 2 4 5 6 1 7`. Query `[1,8]` has the run `2<4<5<6` of length `4`.
Query `[2,6]` is `3 2 4 5 6`; the run `2<4<5<6` again gives `4`. The update sets `a[7]=8`, so the
array becomes `1 3 2 4 5 6 8 7`. Query `[5,8]` is `5 6 8 7`: the run `5<6<8` has length `3` — note
the longer run `4<5<6<8` would start at position `4`, which is *outside* the window, so it must not
count. Query `[3,5]` is `2 4 5`, giving `3`.

## Background

The brute force — for each query, scan `l..r` resetting a counter on every non-ascent — is `O(n)`
per query and `O(nq)` overall, far too slow at `2*10^5` of each. The standard accelerator is a
segment tree where each node, covering a block of positions, stores enough to answer "longest
increasing run inside this block" and to be merged with a neighbouring block in `O(1)`:

- `best` — the longest increasing run fully inside the block;
- `pre` — the longest increasing run starting at the block's left end;
- `suf` — the longest increasing run ending at the block's right end;
- the boundary values `lval`, `rval` and the block length `len`.

Two adjacent blocks `L` then `R` merge by deciding whether a run may cross the seam. The seam is a
valid step of an increasing run **iff `L.rval < R.lval`** (a *strict* ascent). If and only if it is,
`L.suf + R.pre` is a candidate for the merged `best`, and the prefix/suffix can grow across the seam
when a child is entirely one run. Point updates rebuild one leaf-to-root path in `O(log n)`; queries
combine `O(log n)` canonical blocks. The open questions are the exact merge formula and — the part
that bites — how the query restricts the combination to exactly `[l, r]` without letting a run leak
past `l` or `r`.

## Evaluation settings

Judged on hidden tests covering: strictly increasing arrays (one run spanning everything, so window
clipping at `l`/`r` is the whole game), strictly decreasing and all-equal arrays (answer `1`
everywhere — equal neighbours are *not* an ascent), single-element windows `l == r`, alternating
up/down patterns, heavy update streams that flip a single boundary repeatedly, large random values
near `±10^9`, and maximal `n, q = 2*10^5` for time.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Node {
    long long lval, rval;
    int pre, suf, best, len;
};

int n;
vector<long long> a;     // 1-indexed values, a[1..n]
vector<Node> tr;         // segment tree, size 4n

Node makeLeaf(long long v) { return Node{v, v, 1, 1, 1, 1}; }

Node merge(const Node &L, const Node &R) {
    Node res;
    res.len  = L.len + R.len;
    res.lval = L.lval;
    res.rval = R.rval;
    // TODO: combine the two children, gluing a run across the seam exactly when
    //       the seam is a STRICT ascent, and update pre / suf / best accordingly.
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int q;
    if (!(cin >> n >> q)) return 0;
    a.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> a[i];
    tr.assign(4 * (n + 1), Node{});
    // build, then process q operations: "1 p x" set, "2 l r" query longest run.
    return 0;
}
```
