# Welding a row of slabs at minimum total heat

## Research question

A foundry has `n` metal slabs laid out left to right in a single row. Slab `i` has integer
**width** `w[i]`. A welding robot fuses the row into one piece by repeatedly performing **adjacent
welds**: it picks two pieces that are currently next to each other in the row and fuses them into a
single piece. Each weld deposits an amount of heat equal to the **combined width** of the two pieces
being joined (the sum of widths of all original slabs inside them). After a weld, the fused piece
occupies the contiguous span of the slabs it contains and can itself be welded to a neighbour later.

The robot keeps welding until the whole row is a single piece. Because welds may only join pieces
that are currently adjacent, the *order* of welds is a sequence of binary joins over the line, and
different orders deposit different total heat. Output the **minimum total heat** over all valid weld
orders.

This is the line-merge / optimal-parenthesization family of interval dynamic programming: it has the
same skeleton as ordering matrix-chain multiplications or building an optimal alphabetic tree, where
a contiguous range is split at an internal boundary and the two halves are solved independently. The
whole difficulty of getting it right lives at that boundary — which slabs a closed range actually
contains, where the split index is allowed to sit, and which prefix index measures a range's width.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 400`); then `n` integers `w[i]`
  (`1 <= w[i] <= 10^6`), whitespace-separated.
- Output (stdout): a single line with the minimum total heat to fuse the whole row into one piece.
- A row with `0` or `1` slab needs no welds, so its answer is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `w = [3, 1, 4, 1]` the answer is `18`.

## Background

The cost of one weld is the combined width of the two pieces joined, and that combined width is just
the sum of the original widths of every slab inside the merged span. So if you ever fully fuse the
closed range of slabs `[i, j]`, the *last* weld in that range pays `w[i] + w[i+1] + ... + w[j]`
regardless of how the two halves were formed, and the two halves are themselves fully-fused
sub-ranges. That recursive structure is what makes interval DP apply.

Two approaches are on the table before committing to one:

- **Greedy by smallest piece.** Always weld the currently-cheapest adjacent pair first (a
  Huffman-like instinct). It is fast and short. The open question is whether "join the smallest
  neighbours first" is actually optimal once the *adjacency* restriction is imposed — Huffman is
  free to combine any two items, but here only neighbours may fuse.
- **Interval dynamic programming.** Define `dp[i][j]` as the minimum heat to fuse the closed range
  `[i, j]` into one piece, split the range at an internal boundary `k`, and add the width of the
  whole range for the final weld. This is `O(n^3)`. The open question is purely the *boundaries*: the
  exact set of slabs `[i, j]` covers, the legal positions for the split `k`, and the prefix index
  that yields the range's total width.

## Evaluation settings

Judged on hidden tests covering: tiny rows (`n = 0`, `n = 1`, `n = 2`) where the answer is `0`, `0`,
and `w[0] + w[1]` respectively; small rows where every weld order can be enumerated by brute force;
rows engineered so a smallest-first greedy is strictly suboptimal; and large rows up to `n = 400`
with widths near `10^6`, where the total heat exceeds the 32-bit range and an `O(n^3)` algorithm must
finish within the limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    if (n <= 1) { cout << 0 << "\n"; return 0; }

    // TODO: compute the minimum total heat to fuse the whole row into one
    // piece, where each weld costs the combined width of the two adjacent
    // pieces it joins. Mind the inclusive/exclusive range boundaries.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
