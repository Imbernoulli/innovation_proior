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

The easiest mistake here is to get the inequality backwards, which would silently keep redundant lines (still correct, just slower) or, worse, drop a line that owns a real interval (wrong answers). Let $L_1=2x$, $L_2=0$, and $L_3=-2x$. All three meet at the origin, so the middle line owns no positive-width interval and should be removed. The left side is $(0-0)(2-(-2))=0$ and the right side is $(-2-0)(0-0)=0$, so the inequality $\ge$ holds and the middle line is removed. If I lower the middle line to $L_2=-5$, the left side becomes $(-5-0)\cdot4=-20$ while the right side remains $0$, so the inequality is false and the middle line survives. I also check this against the geometry directly: the outer lines cross at $x^*=0$, where both sit at height $0$; the middle line sits at height $0$ in the first case (not below, so redundant) and at $-5$ in the second (strictly below, so it carves out an interval). The cross-product test and the direct height comparison agree on both cases, so the sign is right.

Insertion is now simple. A new line belongs at the back because its slope is no larger than the slopes already stored. While the last stored line is made unnecessary by the line before it and the newcomer, I pop from the back. Then I append the new line. A line can be appended once and removed from the back at most once.

Querying uses the other monotonicity. Since $x=a[i]$ never decreases, the best line can only move forward along the stored envelope. At the front of the deque, I compare the first two lines at the current $x$. If the second line is no higher than the first, the first line will never be best again for this or any later query, so I pop it from the front. I repeat until the front line is strictly better than the next line or there is no next line, then return the front value. A line can be removed from the front at most once.

Those two one-way movements are the amortization: every state line is inserted once, then it either stays, leaves from the back during a later insertion, or leaves from the front during a later query. The total number of deque mutations is linear, and each failed while-loop comparison is attached to one outer operation, so the per-step work is amortized constant and the full DP is $O(n)$.

The algebra has been argued piece by piece; watching the deque actually reproduce the DP is a different kind of check. Take $a=[0,1,2,4,7]$ (non-decreasing) and $b=[5,3,3,-1,-4]$ (non-increasing). Line $(5,0)$ registers for $j=0$. At $i=1$ the lone line gives $dp_1=5\cdot1+0=5$; line $(3,5)$ appends, since its slope is smaller than the one on file. At $i=2$: the front two give $5\cdot2+0=10$ and $3\cdot2+5=11$, so no front pop and $dp_2=10$. Registering $j=2$ hits the equal-slope case head-on, $b_2=b_1=3$, and since the incoming intercept $10$ is not smaller than the stored $5$, the new line is dropped rather than stuffed in as a useless duplicate slope. At $i=3$: $5\cdot4+0=20$ against $3\cdot4+5=17$, so $(5,0)$ pops from the front and $dp_3=17$; line $(-1,17)$ registers. At $i=4$: $3\cdot7+5=26$ against $-1\cdot7+17=10$, so $(3,5)$ pops and $dp_4=10$. Checking against the brute definition, $\min_j(dp_j+b_j\cdot7)=\min(35,26,31,10)=10$, matches, and the two front pops at $i=3,4$ are exactly the query pointer advancing as $x$ grows — the behavior the monotone-query assumption promised.

If I am not confident I can get the monotone deque hull's equal-slope cleanup and exact three-line cross-product back-pop test right under the budget, I would fall back to a coordinate-compressed Li Chao tree over the given `a[i]` queries, which is slower but is the standard correct CHT variant for this recurrence.

Turning this into code is direct: a deque of `(slope, intercept)` pairs, `register_line` collapsing an equal-slope duplicate and then running the back-pop while-loop with the `__int128` cross-product test, `query` running the front-pop while-loop and returning the winning line's value at $x$. Both are amortized $O(1)$, so looping over all $i$ once, querying then registering, is $O(n)$ end to end.
