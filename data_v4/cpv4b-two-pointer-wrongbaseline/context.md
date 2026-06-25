# Shortest rapid-climb window in a drone altitude log

## Research question

A survey drone records a flight log: for each of `n` consecutive one-second ticks it stores the
**net altitude change** during that tick, an integer `a[i]` (positive when climbing, negative when
descending, zero when level). Mission control wants to certify a *rapid climb*: the **shortest**
contiguous run of ticks during which the drone's net altitude gain is **at least `S` meters**.

Formally, over all contiguous windows `[l, r]` with `0 <= l <= r <= n-1`, consider those whose
window sum `a[l] + a[l+1] + ... + a[r]` is `>= S`. Output the **minimum window length** `r - l + 1`
among them, or `-1` if no window reaches `S`. A window must be non-empty (length at least `1`).

Because the per-tick changes may be **negative**, this is the negative-aware version of "shortest
subarray with sum at least `S`". That negativity is the whole point: the textbook sliding-window
two-pointer that solves the all-positive version is *not* correct here, and certifying a climb on a
wrong (too long, or falsely "impossible") window would ground a perfectly good drone.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `S`
  (`0 <= n <= 2*10^5`, `-10^9 <= S <= 2*10^14`). The second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty/absent.
- Output (stdout): a single line with the minimum window length, or `-1` if no contiguous window
  has sum `>= S`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 8`, `S = 7`, `a = [4, 2, 2, 2, 2, -3, 3, 6]` the answer is `2` (the window
`a[6..7] = 3 + 6 = 9 >= 7` has length 2; no single tick reaches 7).

## Background

The window sum of `[l, r]` equals `prefix[r+1] - prefix[l]`, where `prefix[0] = 0` and
`prefix[k] = a[0] + ... + a[k-1]`. So the task is: over index pairs `i < j` of the prefix array,
minimize `j - i` subject to `prefix[j] - prefix[i] >= S`. Two families of approach are on the table
before committing to one:

- **Sliding-window two-pointer.** Keep a window `[l, r]`; advance `r`, and while the window sum is
  still `>= S` advance `l` to shrink it, recording lengths. This is the standard `O(n)` answer to the
  classic **all-positive** "smallest subarray with sum >= S". The open question is whether shrinking
  from the left is still valid once `a[i]` can be negative — i.e. whether window sum is still
  monotone in the window endpoints.
- **Monotonic deque over prefix sums.** Scan `j` from `0` to `n`; maintain a deque of candidate left
  indices `i` whose `prefix[i]` values are increasing, popping from the front whenever
  `prefix[j] - prefix[front] >= S`. This is `O(n)` and makes no positivity assumption. The open
  question is the exact pop rules (front vs back) and why each discarded index is provably useless.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays (where the naive two-pointer would *look*
right), arrays mixing negatives, zeros, and positives, `S <= 0` (a length-1 window may already
qualify), `S` larger than the total achievable gain (answer `-1`), the empty log (`n = 0`), a single
tick, and large `n = 2*10^5` with `|a[i]|` near `10^9` and `S` near `2*10^14` (so prefix sums and `S`
both exceed the 32-bit range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: output the minimum length of a contiguous window with sum >= S, or -1 if none.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
