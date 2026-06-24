# Worst decline of a reservoir level

## Research question

A reservoir starts the season at level `0`. Over `n` days, day `i` changes the level by an integer
`d[i]` (which may be **positive, negative, or zero**). The level *after* day `i` is the prefix sum
`P[i] = d[0] + d[1] + ... + d[i]`, and the level *before* any day — the starting level — is
`P[-1] = 0`.

The **worst decline** is the largest drop from some earlier level to a later level:

```
worst_decline = max over -1 <= i <= j <= n-1 of ( P[i] - P[j] ).
```

Because `i = j` is allowed (a drop of `0`), the worst decline is always at least `0`. Output that
value. Concretely: how far, at most, does the reservoir fall from any prior high-water mark to any
later point in the season?

This is the "maximum drawdown" computation that appears in finance (peak-to-trough loss of an
equity curve), hydrology, and resource-buffer monitoring. The one-dimensional version is a textbook
prefix-sum sweep, but the negative/zero values and — critically — whether the *initial* level `0`
counts as a peak make the base case and the sign of the comparison easy to get wrong.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `d[i]`
  (`-10^9 <= d[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the worst decline (a non-negative integer).
- Time limit: 1 second. Memory: 256 MB.

Example: for `d = [3, -2, -3, 4, -5, 2, 1]` the levels are
`0 (start), 3, 1, -2, 2, -3, -1, 0`. The reservoir peaks at `3` (after day 0) and later bottoms out
at `-3` (after day 4), so the worst decline is `3 - (-3) = 6`.

## Background

The drop `P[i] - P[j]` for a fixed later index `j` is maximized by choosing the **highest** level
`P[i]` among all indices `i <= j`. So a single left-to-right sweep that carries the running maximum
of the levels seen so far suffices. Two families of approach are on the table before committing:

- **All-pairs scan.** For every pair `i <= j` compute `P[i] - P[j]` and keep the maximum. This is
  `O(n^2)`, obviously correct, and trivial to write — but quadratic blows the time limit at
  `n = 2*10^5`. Useful only as a reference oracle on tiny inputs.
- **Prefix sum with a running peak.** Sweep once, maintaining `peak = max level seen so far`
  (including the start level `0`) and the current level `prefix`. At each step the best decline
  ending here is `peak - prefix`. This is `O(n)`, `O(1)` memory; the open questions are the exact
  base case (does the start level `0` participate as a peak?) and the comparison's sign/order.

## Evaluation settings

Judged on hidden tests covering: all-increasing sequences (answer `0`), sequences mixing negatives
and zeros, the empty sequence (`n = 0`, answer `0`), a single day (`n = 1`, including a single
negative day, whose answer is *not* `0` because the start level `0` is the peak), all-negative
sequences (the level only ever falls, so the decline is the total fall from the start), and large
`n = 2*10^5` with `|d[i]|` near `10^9` so a level — and hence a decline — can reach about `2*10^14`,
far beyond 32-bit range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // TODO: read the n daily changes and compute the worst decline
    //       max over -1 <= i <= j <= n-1 of (P[i] - P[j]), where P[-1] = 0.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
