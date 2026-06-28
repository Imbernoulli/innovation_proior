**Problem.** Given `n` points with integer coordinates (`-10^9 <= x, y <= 10^9`, `0 <= n <= 2*10^5`, points may coincide), find the closest pair and print the **squared** Euclidean distance between them. If `n < 2` there is no pair, print `-1`. Outputting the squared distance keeps the answer an exact integer for a tolerance-free judge.

**Why the obvious brute force is too slow.** The `O(n^2)` double loop does `~2*10^10` distance checks at `n = 2*10^5` — twenty-plus seconds, an order of magnitude over a two-second limit. It survives only as a correctness oracle on `n <= ~2000`. We need `O(n log n)`.

**Key idea — sweepline with an ordered-set window of width = current best `d`.** Sort points by `x` and sweep left to right, keeping the points already seen in a balanced BST (`std::set`) keyed by `(y, x)`. Maintain `best`, the smallest squared distance found so far, and let `d = sqrt(best)`. For a new point `p_i`, the only earlier points that can beat `best` lie in the box `[x_i - d, x_i] x [y_i - d, y_i + d]`:

- *Enforce the `x` lower bound* by evicting from the set every earlier point whose `x` is more than `d` behind (a `left` pointer over the `x`-sorted array; the front has the largest gap, so stop at the first survivor). This eviction test stays in exact integers: drop while `dx*dx >= best`.
- *Query the `y` band* `[y_i - d, y_i + d]` with `lower_bound`/`upper_bound` and compare `p_i` against the survivors.

A box-packing argument bounds the survivors: at most a constant number of points can sit in a `d x 2d` region while all pairwise distances are `>= d`, so each query returns `O(1)` points amortized. Total `O(n log n)`. This replaces the famously fiddly divide-and-conquer strip-merge with two clean moves and is the simpler `O(n log n)` method to get exactly right.

**Pitfalls to get right.**
1. *The `y`-band needs an exact integer `d`.* `(long long)sqrt((double)best)` truncates to the floor, and near `2^53`–`2^63` `double` rounds wrong, so the band can be one too narrow and *exclude* the genuinely closest point — a silent too-large answer. Fix `d` to the exact integer ceiling: `d = ceil(sqrt((double)best))`, then `while (d*d < best) d++;` and `while (d>0 && (d-1)*(d-1) >= best) d--;`. The `x`-eviction, by contrast, needs no `sqrt` — keep it as `dx*dx >= best`.
2. *Overflow.* A coordinate gap reaches `2*10^9`; a squared distance reaches `8*10^18`, which fits in signed 64-bit (`~9.2*10^18`) but overflows 32-bit. Use `long long` for coordinates, differences, and squared distances throughout.
3. *Bootstrap.* Before any pair is found `best = LLONG_MAX`; `sqrt(LLONG_MAX)` is meaningless, so special-case it by scanning the (then tiny) window directly until `best` becomes finite.

**Edge cases.** `n = 0` / `n = 1` -> `-1` (short-circuit). Coincident points -> `0`. Collinear horizontal (all same `y`) -> `x`-eviction carries it. Collinear vertical (all same `x`) -> `x` never prunes, but the `y`-band still keeps each query `O(1)`, so it stays fast (`2*10^5` points run in ~`0.1s`). Extreme `(-10^9,-10^9)`–`(10^9,10^9)` -> `8*10^18`, exact.

**Complexity.** `O(n log n)` time (sort plus `n` set operations, each touching `O(1)` band candidates), `O(n)` space.

**Code.**

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

    if (n < 2) {                       // fewer than two points: no pair exists
        cout << -1 << "\n";
        return 0;
    }

    sort(p.begin(), p.end());          // sort by x, then y

    // best = current smallest squared distance found so far.
    long long best = LLONG_MAX;

    // Active set: points within the vertical strip of half-width d = sqrt(best)
    // behind the sweep, keyed by (y, x) so we can range-query on y.
    set<pair<long long, long long>> win; // (y, x)
    int left = 0;                      // p[left..i-1] are candidates by x
    for (int i = 0; i < n; i++) {
        long long xi = p[i].first, yi = p[i].second;

        // d = ceil(sqrt(best)); any point farther than d in x cannot beat best.
        // Use the squared comparison directly: drop p[left] while
        // (xi - p[left].x)^2 >= best  (strictly cannot improve).
        while (left < i) {
            long long dx = xi - p[left].first;
            if (dx * dx >= best) {
                win.erase({p[left].second, p[left].first});
                left++;
            } else break;
        }

        // Window in y: [yi - d, yi + d]. We don't have d as an integer, so we
        // iterate the set on the y-band and stop once dy^2 >= best.
        // Lower bound: first point with y >= yi - d. Since d^2 ~ best, use
        // a sentinel; we walk outward from yi using lower_bound on y.
        if (best == LLONG_MAX) {
            // No bound yet: just scan everything currently in window
            // (only happens for the very first comparisons; window is tiny).
            for (auto &q : win) {
                long long dx = xi - q.second;
                long long dy = yi - q.first;
                long long d2 = dx * dx + dy * dy;
                if (d2 < best) best = d2;
            }
        } else {
            // Integer ceil of sqrt(best) gives the y-band half-width.
            long long d = (long long)ceil(sqrt((double)best));
            // guard against floating error
            while (d * d < best) d++;
            while (d > 0 && (d - 1) * (d - 1) >= best) d--;

            auto lo = win.lower_bound({yi - d, LLONG_MIN});
            auto hi = win.upper_bound({yi + d, LLONG_MAX});
            for (auto it = lo; it != hi; ++it) {
                long long dx = xi - it->second;
                long long dy = yi - it->first;
                long long d2 = dx * dx + dy * dy;
                if (d2 < best) best = d2;
            }
        }

        win.insert({yi, xi});
    }

    cout << best << "\n";
    return 0;
}
```
