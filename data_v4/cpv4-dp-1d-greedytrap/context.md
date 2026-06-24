# Cheapest tightrope crossing with one-or-two-platform hops

## Research question

A tightrope walker must cross a canyon over a line of `n` rest platforms numbered `0..n-1`. She
starts already standing on platform `0` and must finish standing on platform `n-1`. From a platform
she can hop forward onto the **next** platform (`i -> i+1`) or **skip exactly one** and land two ahead
(`i -> i+2`); she can never hop more than two ahead. Every time she lands on (or starts on) a
platform `i` she pays its toll `t[i] >= 0`. She always pays `t[0]` (her start) and `t[n-1]` (her
finish). Output the **minimum total toll** of a legal crossing.

This is a one-dimensional shortest-route DP. The constraint that only `+1` and `+2` hops are allowed
makes the set of reachable next platforms tiny, which is exactly the situation where a *local* greedy
("always hop to the cheaper of the two platforms in front of me") looks plausible — and is wrong.
Getting the one-dimensional version exactly right, including the `n = 0`, `n = 1`, `n = 2` corners and
the 64-bit toll accumulation, is the point.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `t[i]`
  (`0 <= t[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the minimum total toll of a legal crossing from platform `0`
  to platform `n-1`. For `n = 0` (no platforms) the answer is `0`. For `n = 1` the answer is `t[0]`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `t = [3, 1, 1, 9, 1, 9, 1]` the answer is `6` (route `0 -> 2 -> 4 -> 6`, paying
`3 + 1 + 1 + 1`).

## Background

Because only `+1` and `+2` hops exist, from platform `i` the walker faces at most two choices, and the
toll she pays next is one of `t[i+1]`, `t[i+2]`. Two families of approach are on the table before
committing to one:

- **Local greedy.** Repeatedly hop to whichever of the two reachable platforms (`i+1`, `i+2`) carries
  the smaller toll, breaking ties however. It is `O(n)` and three lines. The open question is whether
  picking the cheaper *immediate* platform is globally optimal, given that a hop changes which
  platforms are reachable afterwards.
- **Linear dynamic programming.** Define `dp[i]` = the minimum total toll of a legal route that lands
  on platform `i`. A route reaching `i` arrives from `i-1` (a `+1` hop) or `i-2` (a `+2` hop), so
  `dp[i] = t[i] + min(dp[i-1], dp[i-2])`. This is `O(n)`, `O(1)` memory. The open question is the
  exact base cases (`dp[0]`, `dp[1]`) and the corner sizes.

## Evaluation settings

Judged on hidden tests covering: tiny arrays (`n = 0, 1, 2`), arrays engineered so the local greedy is
strictly suboptimal (alternating cheap/expensive tolls), arrays with many tolls equal to `0`,
all-equal tolls, and large `n = 2*10^5` with tolls near `10^9` (so the accumulated toll can exceed a
32-bit integer — it can reach about `2*10^14`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> t(n);
    for (auto &x : t) cin >> x;

    // TODO: compute the minimum total toll of a legal crossing from platform 0 to
    // platform n-1 using hops of +1 or +2 (every landed-on platform pays its toll).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
