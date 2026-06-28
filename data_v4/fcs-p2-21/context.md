# Box stacking for maximum height (rotations allowed)

## Research question

You are given `n` box **types**. Box type `i` has three positive integer side lengths
`x_i, y_i, z_i`. You may build a vertical stack of boxes subject to these rules:

- **Unlimited supply, free rotation.** You may use each box type any number of times, and each
  time you place a box you may rotate it so that *any one* of its three dimensions is the vertical
  height; the other two dimensions form the box's rectangular **base**.
- **Strictly shrinking base.** A box may rest on top of another box only if **both** of its base
  dimensions are *strictly* smaller than the corresponding base dimensions of the box directly
  below it. (Equality in either base dimension is not allowed.) Two rectangles `(w1, d1)` and
  `(w2, d2)` may be matched in either orientation, so `(w1, d1)` fits strictly inside `(w2, d2)`
  iff, after sorting each pair, the smaller-of-one is `<` the smaller-of-the-other and likewise for
  the larger.

A stack's total height is the sum of the heights of the boxes in it. Compute the **maximum
achievable total height** over all legal stacks. The empty stack (height `0`) is always allowed.

This is the classic *box stacking* problem. It appears as a packing/scheduling kernel and is a clean
instance where the temptingly simple heuristics are wrong and a provably correct ordered DP is
cheap.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 200`). Then `n` lines (or simply `3*n` further
  whitespace-separated tokens) each giving three integers `x_i y_i z_i`
  (`1 <= x_i, y_i, z_i <= 10^6`).
- Output (stdout): a single line with the maximum total stack height.
- Time limit: 1 second. Memory: 256 MB.

Example: for the three box types `(6, 6, 10)`, `(5, 9, 9)`, `(4, 8, 8)` the answer is `23`.

## Background

Because each box may be rotated and reused, the natural move is to expand every box type into its
distinct orientations and then choose a strictly-nesting sequence of bases that maximizes the summed
height. Two families of approach are on the table before committing to one:

- **A greedy / heuristic pick.** Sort boxes by some local score — base area, volume, or "use each
  box once standing on its tallest face" — and then take them in that order while each still fits.
  These are `O(n log n)` and short to write; the open question is whether any such local rule is
  actually optimal once the strict-nesting constraint couples the choices globally.
- **An ordered dynamic program.** Expand each type into its orientations, put them in an order in
  which any box that can sit on top of another comes *after* it, and run a longest-increasing-
  subsequence-style DP where `dp[j]` is the tallest stack whose top box is `j`. The number of
  oriented boxes is `m = 3n <= 600`, so an `O(m^2)` DP is trivially fast; the open questions are the
  correct ordering, the exact nesting test, and the base-orientation normalization.

## Evaluation settings

Judged on hidden tests covering: `n = 0` (answer `0`), `n = 1` (a single type, which can still stack
several of its own orientations), cubes and equal-dimension boxes that create base-area ties,
instances engineered so that area/volume/tallest-face greedy is strictly suboptimal, and the largest
case `n = 200` with dimensions near `10^6` (so the summed height can exceed a 32-bit integer).

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
    // Read n box types; each has three positive integer dimensions.
    // TODO: expand each type into its orientations (base = two non-height dims),
    //       order them so a box can only rest on one that comes earlier,
    //       and compute the maximum total height via a strict-nesting DP.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
