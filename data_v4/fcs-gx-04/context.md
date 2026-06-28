# Weighted Manhattan meeting point (median minimizes total L1)

## Research question

There are `n` groups of people scattered on an integer grid. Group `i` sits at the
integer point `(x[i], y[i])` and contains `w[i]` people. We want to pick **one integer
meeting point** `(X, Y)` and bring everyone there, where the effort of moving one person
from `(x[i], y[i])` to `(X, Y)` is the **Manhattan (L1) distance** `|X - x[i]| + |Y - y[i]|`.

Choose `(X, Y)` to **minimize the total people-weighted travel**

```
cost(X, Y) = sum over i of  w[i] * ( |X - x[i]| + |Y - y[i]| )
```

and output that minimum total cost. The meeting point may coincide with a group's
location or be any other integer lattice point; it is not restricted to the input points.

This is the weighted L1 facility-location ("meeting point" / "post office") problem on the
plane. The one-dimensional, equal-weight version is folklore; the load-bearing subtlety is
*which* central statistic minimizes total L1, and how the two axes interact.

## Input / output contract

- Input (stdin):
  - the first token is `n` (`0 <= n <= 2*10^5`);
  - then `n` lines (whitespace-separated triples) `x[i] y[i] w[i]` with
    `-10^9 <= x[i], y[i] <= 10^9` and `1 <= w[i] <= 10^9`.
- Output (stdout): a single line with the minimum total weighted Manhattan distance.
- Time limit: 1 second. Memory: 256 MB.

Note on magnitude: the total weight can reach `2*10^5 * 10^9 = 2*10^14` and a single-axis
displacement can reach `2*10^9`, so the answer can reach `~4*10^23`. This **does not fit in
a signed 64-bit integer** (max `~9.2*10^18`); the accumulator and the printed value must be
128-bit (or an equivalent big integer).

### Sample

Input:

```
4
0 0 1
4 0 1
0 4 1
4 4 1
```

Output:

```
16
```

The four unit-weight corners of a 4x4 square. Any integer point inside the square (e.g.
`(2, 2)`, or in fact any `(X, Y)` with `0 <= X <= 4`, `0 <= Y <= 4`) is optimal: each of the
four people walks `2 + 2 = 4`, for a total of `16`.

## Background

The instinct is to send everyone to the **centroid (mean)** `(x̄, ȳ)`. That is the right
answer for *squared* Euclidean distance (the mean minimizes `sum (X - x[i])^2`), but it is
the **wrong** statistic for L1. A concrete one-axis example: positions `0, 0, 0, 100` with
unit weights have mean `25`; total L1 to `25` is `25+25+25+75 = 150`, whereas to the median
`0` it is `0+0+0+100 = 100`. The mean is dragged toward the outlier; the median is not.

Two structural facts are in play before committing to an algorithm:

- **Separability of L1 across axes.** Because `|X - x[i]| + |Y - y[i]|` is a *sum* of an
  X-only term and a Y-only term, the total cost splits as
  `(sum_i w[i]|X - x[i]|) + (sum_i w[i]|Y - y[i]|)`. The two brackets share no variable, so
  `X` and `Y` can be optimized **independently** — the 2D problem is two 1D problems. (This
  is exactly what fails for L2, where the cross term couples the axes.)

- **The 1D weighted median.** Minimizing `f(X) = sum_i w[i]|X - x[i]|` over a single axis is
  a convex, piecewise-linear problem whose minimizer is a **weighted median** of the
  coordinates: the smallest coordinate at which the cumulative weight reaches half the total.
  Not the mean.

Naive alternatives and why they fail at the limits:

- **Scan every candidate point** in the bounding box: the box can be `2*10^9` wide on each
  side, so a grid scan is astronomically infeasible (and even an `O(n)` cost evaluation per
  candidate over many candidates is hopeless).
- **Try every input point as the meeting point** and evaluate cost in `O(n)`: that is
  `O(n^2) = 4*10^10` — far too slow for `n = 2*10^5` in 1 second.

## Evaluation settings

Judged on hidden tests covering: the empty input (`n = 0`, answer `0`); a single group
(answer `0`); all groups at one location (answer `0`); collinear groups (one axis
degenerate); even-vs-odd total weight (the weighted-median "flat region" / tie); heavy skew
where one group's weight dwarfs the rest (the median sits on that group); and large
`n = 2*10^5` with extreme coordinates and weights so the answer overflows 64 bits.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;            // n = 0 (or empty input) -> answer 0

    vector<long long> x(n), y(n), w(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i] >> w[i];

    // TODO: choose an integer meeting point (X, Y) minimizing
    //   sum_i w[i] * (|X - x[i]| + |Y - y[i]|)
    // and print that minimum total cost (it can exceed 64 bits).

    cout << 0 << "\n";
    return 0;
}
```
