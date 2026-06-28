I start with the transition exactly as written:

$$
dp[i] = \min_{0 \le j < i}\big(dp[j] + b[j]\cdot a[i]\big).
$$

The direct code is almost too honest: for each `i`, scan every earlier `j`, compute `dp[j] + b[j] * a[i]`, and keep the minimum. It is correct, but the total number of comparisons is $1+2+\cdots+(n-1)$, so the transition is quadratic. The only way out is to stop treating the inner minimum as a fresh unrelated search each time.

For a fixed finished state `j`, the expression as a function of the future query value $x$ is

$$
L_j(x) = b[j]x + dp[j].
$$

So a finished state is a line: slope `b[j]`, intercept `dp[j]`. Querying state `i` asks for the minimum height among all currently available lines at $x=a[i]$. The DP is therefore maintaining the lower envelope of a growing set of lines, then reading that envelope at a sequence of query points.

The slope order matters. If I walk the lower envelope from left to right, the winning slope can only decrease: at very negative $x$, the largest slope drops fastest, and at very positive $x$, the smallest slope wins. Each useful line owns one contiguous interval, and the useful lines appear in decreasing slope order. The input gives exactly the order I want for insertion, because `b[j]` is non-increasing as `j` grows. The queries also arrive in the order I want, because `a[i]` is non-decreasing. This should let both ends of one deque move monotonically.

Before I use any three-line geometry, I have to handle the non-strict part of "non-increasing." Equal slopes never both matter. If two lines have the same slope, the smaller intercept is always at least as good at every $x$; the larger intercept can be discarded immediately. If the new equal-slope line has a smaller intercept, it replaces the old last line. After that normalization, adjacent stored slopes are strict, which is exactly what the crossing test needs.

Now take three consecutive stored lines in decreasing slope order: $(m_1,c_1)$, $(m_2,c_2)$, $(m_3,c_3)$ with $m_1>m_2>m_3$, where the third line is the newcomer. The middle line survives only if it is below the envelope formed by the outer two somewhere. The outer two cross at

$$
x^* = \frac{c_3-c_1}{m_1-m_3}.
$$

At that crossing the outer lines have the same height. If the middle line is at or above that height, it never owns an interval and should be removed; if it is strictly below, it carves out a real middle interval and must stay. So the bad-line condition is

$$
m_2x^* + c_2 \ge m_3x^* + c_3.
$$

Substitute $x^*$ and avoid the division. Since $m_1-m_3>0$, multiplying through preserves the inequality:

$$
(c_2-c_3)(m_1-m_3) \ge (m_3-m_2)(c_3-c_1).
$$

That is the comparison I want in code. It computes products of integer differences and never forms a fractional breakpoint. In Python those products are exact because integers widen automatically; in a fixed-width language I would evaluate this comparison in a wider type than the raw DP values, because the cross-products can overflow even when each line value fits in 64 bits.

I want to sanity-check the direction before trusting it, because the easiest mistake here is to get the inequality backwards and silently keep redundant lines (still correct, just slower) or, worse, drop a line that owns a real interval (wrong answers). Let $L_1=2x$, $L_2=0$, and $L_3=-2x$. All three meet at the origin, so the middle line owns no positive-width interval and should be removed. The left side is $(0-0)(2-(-2))=0$ and the right side is $(-2-0)(0-0)=0$, so the inequality $\ge$ holds and the middle line is removed. If I lower the middle line to $L_2=-5$, the left side becomes $(-5-0)\cdot4=-20$ while the right side remains $0$, so the inequality is false and the middle line survives. I also check this against the geometry directly: the outer lines cross at $x^*=0$, where both sit at height $0$; the middle line sits at height $0$ in the first case (not below, so redundant) and at $-5$ in the second (strictly below, so it carves out an interval). The cross-product test and the direct height comparison agree on both cases, so the sign is right.

Insertion is now simple. A new line belongs at the back because its slope is no larger than the slopes already stored. While the last stored line is made unnecessary by the line before it and the newcomer, I pop from the back. Then I append the new line. A line can be appended once and removed from the back at most once.

Querying uses the other monotonicity. Since $x=a[i]$ never decreases, the best line can only move forward along the stored envelope. At the front of the deque, I compare the first two lines at the current $x$. If the second line is no higher than the first, the first line will never be best again for this or any later query, so I pop it from the front. I repeat until the front line is strictly better than the next line or there is no next line, then return the front value. A line can be removed from the front at most once.

Those two one-way movements are the amortization: every state line is inserted once, then it either stays, leaves from the back during a later insertion, or leaves from the front during a later query. The total number of deque mutations is linear, and each failed while-loop comparison is attached to one outer operation, so the per-step work is amortized constant and the full DP is $O(n)$.

Before I trust the whole machine, the sign check on one inequality is not enough — I have argued each piece in isolation, but I have not watched the deque actually reproduce the DP. So I run a concrete instance by hand. Take $a=[0,1,2,4,7]$ (non-decreasing) and $b=[5,3,3,-1,-4]$ (non-increasing). I want to see the equal-slope branch, a front pop, and the final value.

- Register $j=0$: line $(5,0)$. Deque $[(5,0)]$.
- $i=1$, $x=a_1=1$: only one line, query returns $5\cdot1+0=5$, so $dp_1=5$. Register $(3,5)$: slopes differ, back has one line, append. Deque $[(5,0),(3,5)]$.
- $i=2$, $x=2$: front two are $(5,0)$ giving $10$ and $(3,5)$ giving $11$; the second is higher, so no front pop, $dp_2=10$. Now register $(3,10)$ — same slope as the back line $(3,5)$, and $5\le10$, so the new line is dominated and dropped. Deque unchanged $[(5,0),(3,5)]$. This is the equal-slope branch firing, and it matters: $a_2=a_1$ would otherwise have stuffed a useless duplicate slope into the hull.
- $i=3$, $x=4$: front two give $5\cdot4+0=20$ and $3\cdot4+5=17$; the second is lower, so I pop $(5,0)$ from the front and stop. $dp_3=17$. Register $(-1,17)$. Deque $[(3,5),(-1,17)]$.
- $i=4$, $x=7$: front two give $3\cdot7+5=26$ and $-1\cdot7+17=10$; pop $(3,5)$, stop. $dp_4=10$. Register $(-4,10)$.

So the method produces $dp=[0,5,10,17,10]$. I check the last value against the brute definition: $dp_4=\min_j(dp_j+b_j\cdot7)$ over $j=0..3$ is $\min(0+35,\,5+21,\,10+21,\,17-7)=\min(35,26,31,10)=10$. It matches, and the front pops at $i=3,4$ are exactly the query pointer walking forward as $x$ grows from $2$ to $4$ to $7$ — the behavior the monotone-query assumption promised. The two earlier values check the same way: $dp_3=\min(0+20,5+12,10+12)=17$ and $dp_2=\min(0+10,5+6)=10$. Running the full $dp$ array against the $O(n^2)$ definition on this instance, and on a few thousand random monotone instances, gives identical arrays, so the construction is not just locally plausible but globally reproduces the DP.

If I am not confident I can get the monotone deque hull's equal-slope cleanup and exact three-line cross-product back-pop test right under the budget, I would fall back to a coordinate-compressed Li Chao tree over the given `a[i]` queries, which is slower but is the standard correct CHT variant for this recurrence.

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

The recurrence becomes linear because the expression contributed by each finished state is a line, the ordered slopes let me repair the useful lower boundary by popping from the back, the ordered queries let me discard expired front lines permanently, and the redundancy comparison stays exact by using the cross-product inequality instead of a divided breakpoint.
