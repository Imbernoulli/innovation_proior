# Closest pair of points (minimum squared distance)

## Research question

You are given `n` points with integer coordinates in the plane. Find the pair of distinct points whose
Euclidean distance is smallest, and output the **square** of that distance. Reporting the squared
distance keeps the answer an exact integer, so the judge can compare it without any floating-point
tolerance.

This is the classic *closest pair* problem. It is a building block in computational geometry,
clustering, collision detection, and nearest-neighbour queries, so the one-shot static version — all
points known up front, one query — has to be both correct on the degenerate inputs (duplicates,
collinear runs) and fast enough at the upper size limit.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` pairs of integers `x_i y_i`
  (`-10^9 <= x_i, y_i <= 10^9`), whitespace-separated. Points may coincide.
- Output (stdout): a single line with the minimum squared distance over all pairs of distinct
  *indices* `i != j`. If `n < 2` there is no pair; print `-1`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the six points `(0,0), (7,6), (3,4), (1,1), (8,7), (10,2)` the closest pair is
`(7,6)`–`(8,7)` (also `(0,0)`–`(1,1)`), at squared distance `1^2 + 1^2 = 2`, so the answer is `2`.

## Background

Two coordinates can each reach `10^9` in magnitude, so a coordinate difference can be `2*10^9` and a
squared distance can reach `(2*10^9)^2 + (2*10^9)^2 = 8*10^18`. That fits inside a signed 64-bit
integer (max about `9.2*10^18`) but overflows 32-bit by nine orders of magnitude, so every coordinate,
difference, and squared distance has to be carried in `long long`.

Two algorithmic families are on the table before committing:

- **Divide and conquer.** Sort by `x`, split at the median, recurse on each half, then merge by
  checking points within the `delta`-strip around the split line, sorted by `y`. This is the textbook
  `O(n log n)` method, but the strip merge is famously fiddly to get exactly right (how many `y`
  neighbours to compare, how to thread the sort through the recursion).
- **Sweepline with an ordered active set.** Process points left to right by `x`, maintaining a balanced
  BST (an ordered set) of the points whose `x` lies within the current best distance `d` behind the
  sweep, keyed by `y`. For each new point, only the `O(1)` amortized points inside the `[y-d, y+d]`
  band can possibly beat `d`. This is also `O(n log n)` and is generally simpler to implement
  correctly than the divide-and-conquer merge.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1` (print `-1`); two points; coincident points
(answer `0`); collinear runs (all same `x`, or all same `y`); dense small-coordinate clouds with many
ties; regular grids (the classic adversary where the best distance stays small while many points
crowd the band); large coordinates near `+-10^9` so the squared distance approaches `8*10^18`; and
`n = 2*10^5` worst cases including a single vertical line where `x` never prunes the active set.

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
    vector<pair<long long, long long>> p(n); // (x, y)
    for (auto &q : p) cin >> q.first >> q.second;

    if (n < 2) {                 // fewer than two points: no pair exists
        cout << -1 << "\n";
        return 0;
    }

    // TODO: compute the minimum squared distance over all pairs of distinct points.
    long long best = 0;

    cout << best << "\n";
    return 0;
}
```
