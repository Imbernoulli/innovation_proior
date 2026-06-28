We need to compute a one-dimensional dynamic program whose transition is

$$
dp[i] = \min_{0 \le j < i}\big(dp[j] + b[j]\cdot a[i]\big), \qquad dp[0]=0,
$$

where `a[i]` is the query point for state `i`, `b[j]` is the coefficient carried by a finished state `j`, and a state `j` may be used only after `dp[j]` is known. The direct way to evaluate this is the honest one: for each `i`, scan every earlier `j`, form $dp[j] + b[j]\cdot a[i]$, and keep the smallest. It is correct, but the total work is $1 + 2 + \cdots + (n-1)$, so the transition is $O(n^2)$. The whole difficulty is that this treats the inner minimum as a fresh, unrelated search at every step, ignoring that the same finished states are scanned again and again. What rescues us is the structure we have been handed but not yet used: `b[j]` is non-increasing as `j` grows, and `a[i]` is non-decreasing as `i` grows. Those two monotonicities are the only reason a linear method is possible, and the goal is to compute all `dp` values in $O(n)$ under exactly those assumptions.

I propose the Monotone Convex Hull Trick. The starting observation is that a finished state `j`, viewed as a function of a future query value $x$, is a straight line,

$$
L_j(x) = b[j]\,x + dp[j],
$$

with slope $b[j]$ and intercept $dp[j]$. Querying state `i` then asks for the minimum height among all currently available lines at $x = a[i]$. So the DP is nothing more than maintaining the lower envelope of a growing set of lines and reading that envelope at a sequence of query points. The slope ordering is what makes the envelope cheap to maintain. If I walk the lower envelope from left to right, the winning slope can only decrease: at very negative $x$ the steepest (largest) slope drops fastest and wins, and at very positive $x$ the smallest slope wins. Each line that survives owns one contiguous interval, and the surviving lines appear in decreasing slope order. The input hands me precisely that order for insertion, because `b[j]` is non-increasing, so every new line goes straight to the back of the structure. And the queries arrive in precisely the order I want, because `a[i]` is non-decreasing, so the best line can only move forward along the envelope. Both ends of a single deque therefore move monotonically, and that is the whole engine.

Before any three-line geometry is safe, I have to deal with the non-strict part of "non-increasing": equal slopes. Two lines with the same slope are parallel, so the one with the smaller intercept is at least as good at every $x$ and the other can be dropped on sight. Concretely, when the newcomer matches the slope of the current last line, I keep only the smaller intercept — if the newcomer is no better I discard it, otherwise I pop the old last line and continue with the newcomer. After this normalization, adjacent stored slopes are strictly decreasing, which is exactly the precondition the crossing test below needs.

Now the redundancy test. Take three consecutive stored lines in strictly decreasing slope order, $(m_1,c_1)$, $(m_2,c_2)$, $(m_3,c_3)$ with $m_1 > m_2 > m_3$, where the third is the line being inserted. The middle line earns its place only if it dips below the envelope formed by the outer two somewhere. The outer two cross at

$$
x^* = \frac{c_3 - c_1}{m_1 - m_3},
$$

and at that crossing they have equal height. If the middle line is at or above that height it never owns an interval and is redundant; if it is strictly below it carves out a genuine middle interval and must stay. So the middle line is removable exactly when $m_2 x^* + c_2 \ge m_3 x^* + c_3$. I do not want to actually compute $x^*$, both because of the division and because a fractional breakpoint invites precision trouble. Since $m_1 - m_3 > 0$, I may substitute $x^*$ and multiply through by $m_1 - m_3$ without flipping the inequality, which clears the denominator and leaves the integer cross-product test

$$
(c_2 - c_3)(m_1 - m_3) \ge (m_3 - m_2)(c_3 - c_1).
$$

This is the load-bearing comparison, and it only ever multiplies and subtracts integer differences — no division, no breakpoint. In Python the products are exact because integers widen automatically; in a fixed-width language I would deliberately evaluate this test in a wider type than the raw DP values, since the cross-products can overflow even when each individual line value fits in 64 bits. It is worth checking the direction once rather than trusting the algebra blindly. Let $L_1 = 2x$, $L_2 = 0$, $L_3 = -2x$; all three meet at the origin, so the middle line owns no positive-width interval and should go. The left side is $(0-0)(2-(-2)) = 0$ and the right side is $(-2-0)(0-0) = 0$, the inequality holds, and the middle line is removed. Lower the middle line to $L_2 = -5$: the left side becomes $(-5-0)\cdot 4 = -20$ while the right stays $0$, the inequality fails, and the middle line correctly survives. The sign is right.

With this, both operations are short. In `register`, the newcomer always belongs at the back because its slope is no larger than what is already stored; while the current last line is rendered useless by the line before it together with the newcomer, I pop from the back, then append. In `query`, the current $x = a[i]$ never decreases, so I look at the front: if the second line is no higher than the first at this $x$, the first line will never win again for this or any later query, so I pop it from the front; I repeat until the front line is strictly better than its successor (or stands alone), then return its value $m\,x + c$. These two one-way movements are exactly the amortization that makes the DP linear: every line is appended once, and afterward it is removed at most once — either from the back during a later insertion or from the front during a later query. Each failed while-loop comparison is charged to one such removal, so the total number of deque mutations is $O(n)$ and the whole DP runs in $O(n)$ time and $O(n)$ space. The linear form genuinely requires both monotonicities together: monotone slopes are what send every new line straight to the back, and monotone queries are what let the front pointer only advance. If slopes were arbitrary one would have to insert into the hull and binary-search it for $O(n\log n)$; if the query points were arbitrary or arrived online, a Li Chao tree would answer point queries against an arbitrary inserted set in $O(\log C)$ each and would be the structure to reach for instead.

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
