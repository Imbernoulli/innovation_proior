# Counting axis-aligned right triangles in a point set

## Research question

You are given `n` points with integer coordinates on the plane. Count the number of **axis-aligned
right triangles** whose three corners are all points of the set. A triangle is *axis-aligned right*
if it has one right angle whose two legs are parallel to the coordinate axes — that is, one leg is a
vertical segment and the other is a horizontal segment, meeting at the right-angle vertex. Two
triangles are the same if and only if they use the same set of three corner points; report each
distinct triple once. Points may be repeated in the input, but **coincident points are one
location**: a triangle's three corners must occupy three distinct positions.

This is the planar-counting subproblem that shows up inside grid-geometry, sweep, and incidence
problems. The whole difficulty is in the counting bookkeeping: deciding what to count once, how to
attribute each triangle to a unique anchor, and how to collapse duplicate input points — the kind of
place where an off-by-one or a double-count silently corrupts the answer.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` pairs `x_i y_i`
  (`-10^9 <= x_i, y_i <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the number of distinct axis-aligned right triangles.
- Time limit: 1 second. Memory: 256 MB.

Example: for the 5 points `(0,0) (0,2) (0,3) (2,0) (3,0)` the answer is `4`.
All four triangles have their right angle at `(0,0)`: a vertical leg up to `(0,2)` or `(0,3)`,
combined with a horizontal leg right to `(2,0)` or `(3,0)`, giving `2 * 2 = 4`.

## Background

The defining feature of an axis-aligned right triangle is its right-angle vertex `P`: from `P` one
leg goes straight up or down (a second point sharing `P`'s x-coordinate) and the other goes straight
left or right (a third point sharing `P`'s y-coordinate). Crucially, that right-angle vertex is
**unique** to the triangle — the other two corners are not right angles — so every qualifying
triangle is "owned" by exactly one of its corners. Two families of approach are on the table before
committing to one:

- **Per-pair / per-edge counting.** Enumerate structures like "pairs of points in the same column"
  and "pairs in the same row" and try to glue them. This invites double-counting, because a single
  triangle can be reached through more than one pair, so a correction or division is needed and is
  easy to get wrong.
- **Per-anchor counting.** Attribute each triangle to its unique right-angle vertex `P`, and for each
  candidate `P` multiply the number of other points in `P`'s column by the number in `P`'s row.
  Because the anchor is unique, summing over all `P` counts every triangle exactly once with no
  division — provided the "exclude `P` itself" subtraction and the duplicate-point collapse are done
  correctly.

## Evaluation settings

Judged on hidden tests covering: the empty set (`n = 0`), fewer than three distinct points (answer
`0`), inputs with many exact-duplicate points (must collapse to one location), tiny coordinate
ranges where rows and columns collide heavily (the dense counting regime), full `k x k` grids where
the count grows like `k^2 (k-1)^2` and overflows 32-bit integers, and large `n = 2*10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<pair<long long,long long>> raw(n);
    for (auto &p : raw) cin >> p.first >> p.second;

    // TODO: count distinct axis-aligned right triangles. Collapse coincident
    // points to one location, then attribute each triangle to its unique
    // right-angle vertex; beware double-counting and the exclude-self off-by-one.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
