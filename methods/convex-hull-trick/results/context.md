# Context

## Problem

A one-dimensional dynamic program has the transition

$$
dp[i] = \min_{0 \le j < i}\big(dp[j] + b[j]\cdot a[i]\big), \qquad i = 1,\dots,n-1,
$$

with base case $dp[0] = 0$. The array `a` supplies the query point for each
state, and `b[j]` is the coefficient attached to a finished state `j`, so state
`j` can only be used after `dp[j]` has already been computed.

Two monotonicity facts are guaranteed:

- `b[j]` is non-increasing as `j` grows;
- `a[i]` is non-decreasing as `i` grows.

The values can be negative, and products such as `b[j] * a[i]` can be large.
The direct transition scans every earlier `j` for every `i`, which is
$O(n^2)$. The target is to compute all `dp` values in linear time under the two
monotonicity assumptions.

## Code framework

The program reads `n`, then arrays `a` and `b`. The `Transition` object stores
finished states and answers the transition query for the next state. The bodies
inside `Transition` are the only missing pieces.

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
        # TODO
        pass

    def register(self, slope, value):
        """Make one finished state available to later transitions."""
        # TODO
        pass

    def query(self, x):
        """Return the best transition value at x."""
        # TODO
        pass


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
