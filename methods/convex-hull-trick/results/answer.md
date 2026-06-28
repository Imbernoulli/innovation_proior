# Monotone Convex Hull Trick

## Problem

Compute

$$
dp[i] = \min_{0 \le j < i}\big(dp[j] + b[j]\cdot a[i]\big), \qquad dp[0]=0,
$$

when `b[j]` is non-increasing and `a[i]` is non-decreasing. The direct transition
is $O(n^2)$.

## Method

Each finished state `j` defines the line

$$
L_j(x) = b[j]x + dp[j].
$$

Then `dp[i]` is the minimum stored line value at `x = a[i]`. With slopes inserted
in non-increasing order, useful lines occur along the lower envelope in that same
order. With query points arriving in non-decreasing order, the best line moves
only forward.

The implementation stores the active lines in a deque:

- `register(slope, value)` first removes equal-slope duplicates by keeping only
  the smaller intercept, then pops redundant lines from the back.
- `query(x)` pops from the front while the next line is no higher at `x`, then
  returns the front line's value.

For three strict-slope lines $(m_1,c_1)$, $(m_2,c_2)$, $(m_3,c_3)$ in decreasing
slope order, the middle line is redundant exactly when

$$
(c_2-c_3)(m_1-m_3) \ge (m_3-m_2)(c_3-c_1).
$$

This is the crossing comparison with the division removed. Python integers avoid
overflow; a fixed-width implementation should evaluate this cross-product test in
a widened integer type.

Each line is appended once and removed at most once from either end, so the DP is
$O(n)$ time and $O(n)$ space under the two monotonicity assumptions.

This linear form requires **both** monotone slopes and monotone queries: that is
what lets every new line go straight to the back and the query pointer only move
forward. With arbitrary slopes one inserts into the hull and binary-searches it
($O(n\log n)$); when the query points can be arbitrary or arrive online, the Li
Chao tree answers point queries against an arbitrary set of inserted lines in
$O(\log C)$ each and is the standard structure for that non-monotone case.

## Code

```cpp
// Reads n, then arrays a[0..n-1] and b[0..n-1] from stdin; prints dp[n-1].
// dp[i] = min_{0<=j<i}(dp[j] + b[j]*a[i]), dp[0]=0, with b non-increasing,
// a non-decreasing. Solved in O(n) by the monotone convex hull trick.
#include <bits/stdc++.h>
using namespace std;

struct Transition {
    // Lower-envelope of lines (slope, intercept) kept in a deque.
    deque<pair<long long, long long>> lines;

    // Make one finished state available to later transitions.
    void register_line(long long slope, long long value) {
        if (!lines.empty() && lines.back().first == slope) {
            if (lines.back().second <= value) return;
            lines.pop_back();
        }
        pair<long long, long long> nw{slope, value};
        while (lines.size() >= 2) {
            long long m1 = lines[lines.size() - 2].first;
            long long c1 = lines[lines.size() - 2].second;
            long long m2 = lines.back().first;
            long long c2 = lines.back().second;
            long long m3 = nw.first, c3 = nw.second;
            // Keep the middle line only if it dips below the envelope of the
            // outer two: (c2-c3)(m1-m3) < (m3-m2)(c3-c1). Use __int128 since the
            // cross-products can overflow 64-bit even when each value fits.
            __int128 lhs = (__int128)(c2 - c3) * (m1 - m3);
            __int128 rhs = (__int128)(m3 - m2) * (c3 - c1);
            if (lhs < rhs) break;
            lines.pop_back();
        }
        lines.push_back(nw);
    }

    // Return the best transition value at x (x non-decreasing across calls).
    long long query(long long x) {
        while (lines.size() >= 2) {
            long long m1 = lines[0].first, c1 = lines[0].second;
            long long m2 = lines[1].first, c2 = lines[1].second;
            if (m2 * x + c2 > m1 * x + c1) break;
            lines.pop_front();
        }
        return lines[0].first * x + lines[0].second;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n), b(n);
    for (int i = 0; i < n; ++i) cin >> a[i];
    for (int i = 0; i < n; ++i) cin >> b[i];

    vector<long long> dp(n, 0);
    dp[0] = 0;

    Transition tr;
    tr.register_line(b[0], dp[0]);

    for (int i = 1; i < n; ++i) {
        dp[i] = tr.query(a[i]);
        tr.register_line(b[i], dp[i]);
    }

    cout << dp[n - 1] << "\n";
    return 0;
}
```
