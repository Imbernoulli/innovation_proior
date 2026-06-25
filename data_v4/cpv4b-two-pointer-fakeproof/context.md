# Counting low-interference time windows (subarrays with bitwise-OR at most K)

## Research question

A radio receiver logs, for each of `n` consecutive time slots, a 31-bit *interference mask*
`a[0..n-1]`: bit `b` of `a[i]` is set when interference source `b` is active during slot `i`. A
*window* is a contiguous block of slots `[l, r]`. The combined interference over a window is the
**bitwise-OR** of its masks (a source counts as active over the window if it is active in *any* slot
of the window). The receiver tolerates combined interference up to a fixed budget `K`: a window is
**clean** if the OR of its masks is `<= K` when both are read as ordinary non-negative integers.

Count how many windows are clean — i.e. how many pairs `(l, r)` with `0 <= l <= r < n` satisfy
`a[l] | a[l+1] | ... | a[r] <= K`.

This is a sliding-window / two-pointer problem. The catch is that the window OR is **not** an
invertible quantity: when the window's left edge advances and a mask leaves, the new OR cannot be
recovered from the old OR and the departing mask by any single bit trick. Getting the OR-maintenance
identity right — and *not* asserting a plausible-but-false one — is the whole problem.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `K`
  (`0 <= n <= 2*10^5`, `0 <= K <= 2^31 - 1`). The second line (present iff `n > 0`) has `n`
  integers `a[i]` (`0 <= a[i] <= 2^31 - 1`), whitespace-separated.
- Output (stdout): a single line with the number of clean windows.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `K = 5`, `a = [4, 3, 1, 2]` the answer is `7`. The windows `[0,1]`, `[0,2]`,
`[0,3]` all have OR `= 7 > 5` and are dirty; the other 7 windows are clean.

## Background

Because OR-ing in more masks can only set bits (never clear them), the window OR is **monotone
non-decreasing** as the window grows. So for a fixed right edge `r`, the clean left edges form a
contiguous suffix `[l*(r), r]`, and `l*(r)` only moves rightward as `r` increases. That is exactly
the structure a two-pointer scan exploits: keep a left pointer, extend `r` by one, then push the left
pointer right until the window OR drops to `<= K`; every start in the surviving window is clean.

The one subtlety is maintaining the OR as the window changes. Adding a mask on the right is easy
(`cur |= a[r]`). Removing a mask on the left is the trap: OR has no inverse, so there is no
expression in `cur` and the departing mask `a[l]` alone that yields the new OR. Two standard fixes
exist, and the solver must choose one deliberately rather than assert a bit identity:

- **Per-bit counts.** Keep `cnt[b]` = how many masks currently in the window have bit `b` set. Adding
  a mask increments the counts of its set bits; removing one decrements them. Bit `b` is in the
  window OR iff `cnt[b] > 0`. This makes removal exact in `O(31)` per step.
- **Recompute from scratch** the OR of the current window whenever needed. Correct but, if done
  naively for every candidate left edge, risks an extra factor that must be checked against the time
  limit.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; `n = 1` (including a single mask above `K`, answer `0`);
`K = 0` (only windows of all-zero masks count); `K = 2^31 - 1` (every window clean, answer
`n(n+1)/2`); masks with heavily overlapping bits (so any naive xor/and-not removal identity is
exposed); and large `n = 2*10^5` with 31-bit masks. The answer can reach `n(n+1)/2 ~ 2*10^10`, so it
must be accumulated in a 64-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long K;
    if (!(cin >> n >> K)) return 0;
    vector<unsigned int> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count windows [l, r] whose bitwise-OR of masks is <= K, using a
    // two-pointer scan with correct OR maintenance when the left edge advances.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
