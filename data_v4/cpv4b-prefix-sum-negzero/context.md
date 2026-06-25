# Deepest drawdown of a reservoir gauge

## Research question

A mountain reservoir reports its level once per day against a fixed reference mark, so the level is a
**relative** number that may be positive (above the mark) or negative (below it). The season begins
with the gauge sitting exactly at the reference, i.e. level `0`, *before* the first day. Each day `i`
the level changes by a signed integer `d[i]` (snowmelt and rain add water, evaporation and release
remove it). The level after day `k` is therefore the prefix sum `L[k] = d[0] + d[1] + ... + d[k-1]`,
with `L[0] = 0` standing for the pre-season reading.

The operators care about the worst **drawdown**: the largest amount the level ever fell from an
earlier reading to a later one. Formally, the drawdown realised by a pair of days `i <= j` is
`L[i] - L[j]` (a positive number when the level dropped, negative when it rose). You must report

```
maxDrawdown = max over all 0 <= i <= j <= n of ( L[i] - L[j] ).
```

Because `i = j` is allowed, the drawdown is always at least `0`: if the gauge only ever rose across
the whole season, the worst drawdown is `0`, not a negative number. Getting the **sign of the drop**
and the **base reading `L[0] = 0`** exactly right — especially when every day is an outflow, when the
log is empty, or when the level only climbs — is the whole point of the exercise.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`), the number of days; then `n` integers
  `d[i]` (`-10^9 <= d[i] <= 10^9`), whitespace-separated, the daily signed level changes.
- Output (stdout): a single line with `maxDrawdown`, the deepest fall of the level from any earlier
  reading to any later one (including the pre-season reading `L[0] = 0`). The value is `>= 0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 8` and `d = [3, -2, -5, 4, -1, -3, 2, 1]` the answer is `7`. The levels are
`L = [0, 3, 1, -4, 0, -1, -4, -2, -1]`; the level peaks at `3` (after day 1) and later sits at `-4`,
a fall of `3 - (-4) = 7`, which is the deepest drop in the log.

## Background

Maximum drawdown is the standard "how far did it fall from a running high" statistic; the same shape
appears in finance (peak-to-trough loss of an equity curve), in monitoring (worst dip of a signal
below its prior peak), and in any prefix-sum log where you compare a later reading to the best earlier
one. Two routes are on the table before committing to one:

- **All-pairs over prefix sums.** Build the level array `L[0..n]` and check every pair `i <= j`. This
  is obviously correct and is the reference brute force, but it is `O(n^2)` and far too slow at
  `n = 2*10^5`. It is useful only as an oracle.
- **Single pass with a running peak.** Walk the days left to right maintaining the maximum level seen
  so far (the running peak, starting at `L[0] = 0`); at each day the best drawdown ending there is
  `peak - L[j]`. Track the maximum of those. This is `O(n)` and `O(1)` extra memory; the open
  questions are the exact base value of the peak, the order of "measure then update", and the sign of
  the subtraction.

## Evaluation settings

Judged on hidden tests covering: logs that only rise (answer `0`), logs that only fall (all-negative
deltas, answer is the whole accumulated decline), logs with zeros interspersed, the empty log
(`n = 0`, answer `0`), a single day (positive -> `0`, negative -> its magnitude), and large
`n = 2*10^5` with deltas near `10^9` so the level and the drawdown can exceed the 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // TODO: read the n signed daily changes, maintain the running level L (a prefix sum starting
    // at L[0] = 0) and the running peak, and report the maximum drawdown max_{i<=j} (L[i] - L[j]).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
