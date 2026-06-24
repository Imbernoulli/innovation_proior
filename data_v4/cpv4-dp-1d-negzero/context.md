# One contiguous run of a machine with a fixed startup cost

## Research question

A small geothermal pump is monitored over `n` consecutive days. On day `i` the pump, if it is
running, produces a **net energy balance** `a[i]` (energy generated minus energy consumed); this
value may be positive, zero, or **negative** (on a bad day the pump costs more than it makes).

You may run the pump during **at most one** contiguous block of days `[l, r]` (with `0 <= l <= r <
n`), and running it costs a single fixed **startup fee** `c >= 0`, paid once for the whole block.
The net profit of running block `[l, r]` is `a[l] + a[l+1] + ... + a[r] - c`. You may also decide
**not to run the pump at all**, for a profit of exactly `0`.

Choose the option that **maximizes profit** and output that maximum. Because the "do nothing"
option is always available, the answer is never below `0`.

This is the "fixed-charge single window" variant of maximum subarray: the chosen run must be a
**non-empty** contiguous block (you cannot pay the fee for zero days), but the overall plan is
allowed to be empty. Getting the boundary between "best non-empty run minus the fee" and "do
nothing" exactly right — especially when every day is negative, when `c` swamps any gain, and when
`n = 0` — is the whole point.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `c`
  (`0 <= n <= 2*10^5`, `0 <= c <= 10^{14}`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable profit.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 7`, `c = 3`, `a = [4, -2, 5, -9, 3, 3, -1]` the answer is `4`: run the block
`[0, 2]` whose day-sum is `4 - 2 + 5 = 7`, pay the startup fee `3`, for net `7 - 3 = 4`. No other
block, after subtracting `3`, beats this, and it beats doing nothing (`0`).

## Background

The constraint is that the run is a single **contiguous** non-empty window, and the fee is paid
exactly once. Two families of approach are on the table before committing to one:

- **Enumerate all windows.** For every pair `(l, r)` compute the block sum and subtract `c`, taking
  the best (and comparing to `0`). This is `O(n^2)` and obviously correct, but far too slow at
  `n = 2*10^5`; it is useful only as a reference oracle on tiny inputs.
- **Linear dynamic programming (Kadane with a fee).** Scan left to right maintaining the best sum
  of a non-empty block **ending at the current day**; the running best over all days, minus the fee,
  competes against `0`. This is `O(n)`; the open questions are the exact recurrence, the base case
  that forbids an empty block, and exactly where the fee and the `0` enter.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, mixtures of negatives and zeros, the empty
array (`n = 0`), a single element (`n = 1`), **all-negative** arrays (answer should be `0`), a fee
`c` large enough to make every block unprofitable (answer `0`), `c = 0` (pure maximum subarray
against the empty option), and large `n = 2*10^5` with values near `10^9` (so a block sum can reach
`~2*10^14`, exceeding 32-bit range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long c;
    if (!(cin >> n >> c)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: best non-empty contiguous block sum minus the fee c, against the "do nothing" 0.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
