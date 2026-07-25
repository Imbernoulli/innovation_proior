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

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 400`); then `n` integers `w[i]`
  (`1 <= w[i] <= 10^6`), whitespace-separated.
- Output (stdout): a single line with the minimum total heat to fuse the whole row into one piece.
- A row with `0` or `1` slab needs no welds, so its answer is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `w = [3, 1, 4, 1]` the answer is `18`.

## Evaluation settings

Judged on hidden tests covering: tiny rows (`n = 0`, `n = 1`, `n = 2`) where the answer is `0`, `0`,
and `w[0] + w[1]` respectively; small rows where every weld order can be enumerated by brute force;
and large rows up to `n = 400` with widths near `10^6`, testing both correctness and performance
within the time limit.

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
