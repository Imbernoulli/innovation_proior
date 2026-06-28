**Problem.** Given `n` groups of people on an integer grid — group `i` at `(x[i], y[i])` with `w[i]` people — choose one integer meeting point `(X, Y)` to minimize the total people-weighted Manhattan travel `sum_i w[i] * (|X - x[i]| + |Y - y[i]|)`, and print that minimum. `n <= 2*10^5`, `|x[i]|, |y[i]| <= 10^9`, `1 <= w[i] <= 10^9`.

**Why the obvious answer is wrong.** The instinct is the **centroid (mean)**. That minimizes *squared* distance, not L1. On one axis with positions `0, 0, 0, 100` (unit weights) the mean `25` costs `150`, while the point `0` costs `100`: the mean is dragged toward the outlier, the median is not. So the mean is the wrong statistic. Brute alternatives also fail at the limits: scanning every lattice point in the bounding box is hopeless (the box is up to `2*10^9` wide), and trying every input point with an `O(n)` cost evaluation is `O(n^2) = 4*10^10` — too slow.

**Key idea (the insight).** Two structural facts collapse the problem:

1. **L1 separates across axes.** Because `|X - x[i]| + |Y - y[i]|` is a sum of an X-only and a Y-only term, the total cost splits as `(sum_i w[i]|X - x[i]|) + (sum_i w[i]|Y - y[i]|)`. The two brackets share no variable, so `X` and `Y` are optimized **independently** — the 2D problem becomes two 1D problems. (This is special to L1; for Euclidean distance the square root couples the axes and no such split exists.)
2. **The 1D minimizer is the weighted median.** `f(X) = sum_i w[i]|X - x[i]|` is convex and piecewise linear; its slope just right of `X` is `(left weight) - (right weight)`, which turns non-negative at the **weighted median** — the smallest coordinate whose cumulative weight reaches `ceil(W/2)` where `W = sum_i w[i]`. *Not* the mean.

Algorithm: build the X list and Y list of `(coord, weight)`; for each, sort, find the weighted median `m`, sum `w * |coord - m|`; add the two axis costs. `O(n log n)` time, `O(n)` space.

**Pitfalls.**
1. *Mean vs median.* The centroid is the L2 answer, not the L1 answer; using it is a wrong-answer.
2. *Overflow.* The total weight reaches `2*10^14` and a displacement reaches `2*10^9`, so the answer reaches `~4*10^23` — far past signed 64-bit (`~9.2*10^18`). A single product `|coord - m| * weight` already reaches `~2*10^18` and can wrap in `long long`. Compute the product and the running cost in `__int128` and print with a digit emitter. Coordinates, weights, and the total weight stay in `long long`.
3. *Empty / degenerate.* `n = 0` (or `total = 0`) must short-circuit to `0`; never call `v.back()` on an empty vector.
4. *Median index.* Use `ceil(W/2) = (W + 1) / 2` so the first coordinate crossing it is where the slope turns non-negative; for even `W` there is a flat optimum, but a site coordinate is always a valid representative.

**Edge cases.** `n = 0` -> `0`; single group -> `0`; all groups at one point -> `0`; collinear groups (one axis degenerate) -> the live axis is the pure 1D median; even total weight (tie / flat region) -> any site median is valid, the cost is unique; heavy-weight skew -> the median sits on the dominant group; max-magnitude inputs -> answer needs 128 bits.

**Complexity.** `O(n log n)` time (two sorts), `O(n)` extra space.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

// Print a (possibly large) non-negative 128-bit integer.
static void print128(__int128 x) {
    if (x == 0) { cout << "0\n"; return; }
    string s;
    while (x > 0) { s += char('0' + (int)(x % 10)); x /= 10; }
    reverse(s.begin(), s.end());
    cout << s << "\n";
}

// Minimum total weighted Manhattan distance from a chosen integer meeting point
// to all weighted sites. L1 separates across axes, so we solve each axis with a
// weighted median: sort the (coordinate, weight) pairs, walk the prefix of weight
// until it reaches half the total, and that coordinate is an optimal meeting value.
// The total cost can reach ~4e23, so accumulate the answer in __int128.
static __int128 solveAxis(vector<pair<long long,long long>> &v, long long total) {
    sort(v.begin(), v.end());
    // Weighted median coordinate m: smallest coord whose prefix weight reaches
    // ceil(total/2). Any optimum can be taken at such a site coordinate.
    long long need = (total + 1) / 2;
    long long acc = 0;
    long long m = v.empty() ? 0 : v.back().first;
    for (auto &p : v) {
        acc += p.second;
        if (acc >= need) { m = p.first; break; }
    }
    __int128 cost = 0;
    for (auto &p : v) {
        long long d = llabs(p.first - m);
        cost += (__int128)d * (__int128)p.second;
    }
    return cost;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;

    vector<pair<long long,long long>> xs(n), ys(n);
    long long total = 0;
    for (int i = 0; i < n; i++) {
        long long x, y, w;
        cin >> x >> y >> w;
        xs[i] = {x, w};
        ys[i] = {y, w};
        total += w;
    }

    if (n == 0 || total == 0) {
        cout << 0 << "\n";
        return 0;
    }

    __int128 ans = solveAxis(xs, total) + solveAxis(ys, total);
    print128(ans);
    return 0;
}
```
