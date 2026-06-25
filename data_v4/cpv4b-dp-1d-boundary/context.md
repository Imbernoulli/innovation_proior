# Cheapest crossing of a stepping-stone causeway

## Research question

A frog crosses a tidal causeway laid out as `n` stepping stones in a line, indexed `0..n-1`. It
starts on stone `0` and must finish standing on stone `n-1`. From a stone `i` the frog can leap
forward only, landing on some stone `j` with `i < j` and `j - i <= D`: the leg muscles give it a
reach of exactly `D` stones, and a gap of `D` is the longest leap it can still make (reach `D` is
**inclusive**). Each stone `j` charges a toll `c[j]` to stand on it; the toll is collected once,
when the frog lands there (the starting stone `0` also charges its toll). Some stones are cracked
and cannot bear the frog's weight at all — those are marked with a toll of `-1` and may never be
landed on.

Find the minimum total toll the frog pays to reach stone `n-1`, or report that the crossing is
impossible.

This is a shortest-path-on-a-line / partition-style one-dimensional DP. The whole difficulty lives
in one boundary: the predecessor window is the stones at distance `1..D` behind, and whether the
endpoint `D` is in or out of that window changes which crossings are feasible.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `D`
  (`1 <= n <= 2*10^5`, `1 <= D <= 2*10^5`). The second line (whitespace-separated, possibly wrapped)
  has `n` integers `c[0..n-1]` with `-1 <= c[i] <= 10^9`, where `c[i] = -1` marks a cracked stone
  and `c[i] >= 0` is the toll. It is guaranteed `c[0]` and `c[n-1]` may or may not be cracked (you
  must handle a cracked start or end yourself).
- Output (stdout): a single line with the minimum total toll to stand on stone `n-1`, or `-1` if
  it cannot be reached.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 8`, `D = 3`, `c = [0, 7, 2, -1, 9, 4, 1, 6]` the answer is `12`
(stand on stone 0 for toll 0, leap to stone 2 for toll 2, leap over the cracked stone 3 to stone 5
for toll 4, then leap to stone 7 for toll 6: `0 + 2 + 4 + 6 = 12`).

## Background

Two ways to organize the computation are on the table before committing:

- **Plain windowed DP.** Define `dp[j]` = least toll to stand on stone `j`. Then
  `dp[j] = c[j] + min(dp[i])` over the legal predecessors `i`, and the answer is `dp[n-1]`. The
  open question is the *exact* predecessor window — which `i` are legal — because the reach bound is
  where an off-by-one silently flips feasibility. A literal `O(n*D)` scan over the window is correct
  but too slow when `D` is large.
- **Windowed DP with a monotonic-deque minimum.** Keep the same recurrence but slide a deque that
  yields `min(dp[i])` over the moving window in amortized `O(1)`, for `O(n)` overall. This is the
  performant version; its correctness rests on the deque's eviction test using the *same* window
  boundary as the recurrence, so the boundary has to be nailed down once and reused consistently.

## Evaluation settings

Judged on hidden tests covering: `D = 1` (every stone must be stepped on, so the answer is the sum
of all tolls or `-1` if any interior stone is cracked); large `D` (a single leap can clear most of
the causeway); cracked start (`c[0] = -1`, immediately impossible); cracked end; runs of cracked
stones that are reachable only when the gap is exactly `D` (the inclusive-boundary case); `n = 1`
(start is already the end); and large `n = 2*10^5` with tolls near `10^9` so the running total
exceeds 32 bits.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> c(n);
    for (auto &x : c) cin >> x;

    // TODO: dp[j] = min toll to stand on stone j; legal predecessors are the stones
    // at distance 1..D behind a landable stone j. Output dp[n-1] or -1.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
