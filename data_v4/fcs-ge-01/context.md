# Farthest pair of points (planar diameter), squared distance

## Research question

You are given `n` points with integer coordinates in the plane. Consider every unordered pair of
**distinct** points and the squared Euclidean distance between them. Output the **maximum** squared
distance over all pairs — the squared **diameter** of the point set.

Squared distance (not the distance itself) is asked for on purpose: it keeps the whole computation in
exact integer arithmetic, so there is a single well-defined correct answer with no floating-point
tie-breaking. The diameter of a point set is the kind of primitive that sits underneath clustering,
collision bounds, minimum-enclosing-circle pre-passes, and shape-descriptor pipelines, so the planar
case has to be exactly right — including the degenerate configurations (all points equal, all points
collinear, heavy duplication) where naive geometry code tends to crack.

## Input / output contract

- Input (stdin): the first token is `n` (`2 <= n <= 2*10^5`). Then `n` lines (or just `2n` further
  whitespace-separated tokens) each give two integers `x y` (`-10^9 <= x, y <= 10^9`). Points are
  **not** guaranteed distinct.
- Output (stdout): a single line with the maximum squared Euclidean distance over all pairs of points.
- Time limit: 1 second. Memory: 256 MB.

Example: for the four points `(0,0), (3,0), (3,4), (0,4)` the answer is `25` — the longest pair is a
diagonal of length 5, e.g. `(3,0)`–`(0,4)`.

## Background

The brute-force reading is immediate: try all `O(n^2)` pairs and keep the largest squared distance.
That is obviously correct and is exactly what a checker would use on small inputs, but at `n = 2*10^5`
it is `~2*10^10` pair evaluations — far beyond a 1-second budget. The task is to get the same exact
answer while only doing near-linear work.

Two structural facts about the diameter frame the faster approach, and both need to be earned rather
than assumed:

- **The farthest pair lies on the convex hull.** If a point is strictly interior to the hull, it
  cannot be an endpoint of the diameter, so we may discard all interior points first. Building the
  hull is `O(n log n)`.
- **The hull has few "antipodal" pairs.** The diameter is realised by two vertices that admit parallel
  supporting lines; sweeping a pair of supporting lines around a convex polygon visits only `O(h)`
  such candidate pairs for a hull of `h` vertices. The open questions are the exact sweep rule and
  how it behaves when the hull degenerates (two points, or a straight segment of collinear inputs).

## Evaluation settings

Judged on hidden tests covering: generic random clouds; the two-point minimum (`n = 2`); all points
identical (answer `0`); all points collinear, including a vertical line and a single repeated segment;
dense small lattices that force many duplicate and collinear triples (the case that breaks fragile hull
code); coordinates at the `±10^9` extremes so the squared distance reaches `8*10^18` and must be held
in 64-bit; and full-size `n = 2*10^5` inputs that must finish within the time limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<ll> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    // TODO: compute the maximum squared Euclidean distance over all pairs.
    ll answer = 0;

    cout << answer << "\n";
    return 0;
}
```
