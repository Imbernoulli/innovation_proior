# Best contiguous shift on a circular conveyor belt

## Research question

A factory packs items onto a closed loop conveyor belt with `n` slots arranged in a circle: slot
`i` is followed by slot `i+1`, and slot `n-1` is followed back to slot `0`. Each slot `i` carries a
signed profit `a[i]` (a damaged item can be negative). A robot arm grabs one **contiguous run of
slots along the belt** and ships them together. Because the belt is a closed loop, a run is allowed
to wrap past the seam from the end back to the start (e.g. slots `n-2, n-1, 0, 1`). The run must be
**non-empty** — the arm always ships at least one slot — and it may use **each slot at most once**,
so its length is between `1` and `n`. Choose the run that **maximizes the total profit shipped** and
output that maximum.

Concretely: among all circular contiguous segments of length `1..n`, report the largest possible
sum. This is the *maximum circular subarray sum* with the standard "non-empty segment" rule, and the
wrap-around possibility is exactly what makes the usual textbook one-pass routine subtly wrong on one
family of inputs.

## Input / output contract

- Input (stdin): the first token is `n` (`1 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum achievable shipped profit.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [5, -3, 5]` the answer is `10` — the wrapping run `{slot 2, slot 0}` (i.e. `5 + 5`)
beats any non-wrapping run (the best non-wrapping run is `5`).

## Background

The non-circular version of this is the classic maximum-subarray problem, solved in `O(n)` by
Kadane's scan. The circular twist invites a well-known "standard" reduction:

- **Textbook wrap formula.** Run Kadane to get the best *non-wrapping* run `best`. A *wrapping* run is
  the whole belt minus some contiguous *non-wrapping* gap, so its best value is `total - worst`,
  where `total` is the sum of all slots and `worst` is the minimum non-empty subarray sum. The
  advertised answer is `max(best, total - worst)`. It is `O(n)`, two Kadane passes, and looks
  airtight — the open question is whether `total - worst` is always a *legal* run for this exact
  "non-empty segment" contract.
- **Brute search.** Try every start `s` and every length `1..n`, summing around the circle. This is
  `O(n^2)`, obviously correct, and only usable to check small cases.

## Evaluation settings

Judged on hidden tests covering: all-positive belts (answer is the whole loop), mixed belts where the
best run wraps the seam, **all-negative belts** (where the textbook wrap formula misfires), single
slot (`n = 1`), belts with zeros, and large `n = 2*10^5` with values near `10^9` (so `total` can
reach `2*10^14` and must not overflow a 32-bit integer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: maximum sum of a non-empty circular contiguous run (length 1..n).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
