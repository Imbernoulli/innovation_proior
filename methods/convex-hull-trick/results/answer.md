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

```python
import sys


def read_input(data):
    """Parse n, then the arrays a[0..n-1] and b[0..n-1] as integers."""
    it = iter(data)
    n = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    b = [int(next(it)) for _ in range(n)]
    return n, a, b


class Transition:
    """Stores finished states and answers min(value + slope * x)."""

    def __init__(self):
        from collections import deque
        self.lines = deque()

    def register(self, slope, value):
        """Make one finished state available to later transitions."""
        lines = self.lines

        if lines and lines[-1][0] == slope:
            if lines[-1][1] <= value:
                return
            lines.pop()

        new_line = (slope, value)
        while len(lines) >= 2:
            m1, c1 = lines[-2]
            m2, c2 = lines[-1]
            m3, c3 = new_line
            if (c2 - c3) * (m1 - m3) < (m3 - m2) * (c3 - c1):
                break
            lines.pop()

        lines.append(new_line)

    def query(self, x):
        """Return the best transition value at x."""
        lines = self.lines
        while len(lines) >= 2:
            m1, c1 = lines[0]
            m2, c2 = lines[1]
            if m2 * x + c2 > m1 * x + c1:
                break
            lines.popleft()

        slope, value = lines[0]
        return slope * x + value


def solve(n, a, b):
    dp = [0] * n
    dp[0] = 0

    tr = Transition()
    tr.register(b[0], dp[0])

    for i in range(1, n):
        dp[i] = tr.query(a[i])
        tr.register(b[i], dp[i])

    return dp


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, a, b = read_input(data)
    dp = solve(n, a, b)
    sys.stdout.write(str(dp[n - 1]) + "\n")


if __name__ == "__main__":
    main()
```
