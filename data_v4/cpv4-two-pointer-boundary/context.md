# Counting comfortable stretches on a trail

## Research question

A hiker records `n` daily altitude readings `a[0..n-1]` (meters; values may be negative, e.g. below sea
level). A **comfortable stretch** is a contiguous block of days `[l, r]` (with `0 <= l < r <= n-1`, so it
spans **at least two days**) in which the highest and lowest altitude differ by at most `D` — formally
`max(a[l..r]) - min(a[l..r]) <= D`. Count how many comfortable stretches there are.

The "at least two days" rule is deliberate: a single day trivially has spread `0`, so without it the answer
would be inflated by `n` degenerate blocks that carry no information about altitude variation. With the rule,
the count is a clean measure of how often the terrain stays within a `D`-meter band over a multi-day window.
This is the kind of bounded-spread window subproblem that appears inside signal smoothing, climate-band
detection, and sliding-window anomaly screening, so getting the **window boundaries** exactly right —
inclusive vs exclusive ends, and the length-`>= 2` cutoff — is the whole game.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `D` (`0 <= n <= 2*10^5`, `0 <= D <= 2*10^9`).
  The second line holds `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0`
  the second line is empty or absent.
- Output (stdout): a single line with the number of comfortable stretches.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `D = 3`, `a = [2, 4, 3, 7, 6, 9]` the answer is `6`.

## Background

The constraint is a **bounded-spread sliding window**: a block is valid iff its max minus its min is at most
`D`. Two structural facts shape the approach.

- **Monotone validity in the left end.** Fix the right end `r`. As the left end `l` moves rightward (the
  window shrinks), the max can only drop and the min can only rise, so the spread `max - min` is
  non-increasing. Hence for each `r` there is a threshold `L(r)`: windows `[l, r]` are valid exactly for
  `l >= L(r)`, and `L(r)` is itself non-decreasing in `r`. That monotonicity is what makes a single
  left pointer that never moves backward correct.
- **Per-`r` counting.** Once `L(r)` is known, every left end `l` in `[L(r), r]` gives a valid block ending
  at `r`. The number of those blocks that span at least two days is the quantity to add up — and pinning
  down that count is precisely where an off-by-one lurks.

Two families of approach are on the table before committing:

- **Brute force over all blocks.** Enumerate every `[l, r]` and track a running max/min as `r` grows from
  `l`. That is `O(n^2)` and obviously correct, but at `n = 2*10^5` it is `~2*10^10` operations — far too slow.
- **Two pointers with monotonic deques.** Advance `r`, maintain a max-deque and a min-deque of indices so
  the current window max and min are `O(1)`, and slide `l` forward only as needed. This is `O(n)`; the open
  questions are the exact deque pop conditions at the boundary and the exact per-`r` count formula.

## Evaluation settings

Judged on hidden tests covering: `D = 0` with repeated values (every length-`>= 2` block of equal readings
counts) and with distinct values (no two-day block counts); `n = 0` and `n = 1` (answer `0`, no two-day block
exists); arrays with negatives and below-sea-level readings; adversarial sequences where the valid window
boundary `L(r)` jumps; and large `n = 2*10^5` with `|a[i]|` near `10^9` and a wide `D`, so the count can reach
`~2*10^10` and overflow a 32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long D;
    if (!(cin >> n >> D)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count contiguous blocks [l, r] of length >= 2 with max - min <= D.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
