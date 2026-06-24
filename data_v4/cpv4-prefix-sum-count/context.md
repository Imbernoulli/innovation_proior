# Counting tide-balanced windows (subarray sums divisible by a period)

## Research question

A coastal station logs `n` hourly net-flow readings `a[0..n-1]` (positive = water flowing in,
negative = flowing out; readings may be negative or zero). The tide repeats with period `m`, and the
engineers call a contiguous block of hours `[l, r]` (with `0 <= l <= r <= n-1`) a **balanced
window** when its net flow `a[l] + a[l+1] + ... + a[r]` is an exact multiple of `m` (that is,
divisible by `m`, where `0` counts as a multiple). They want to know **how many balanced windows the
log contains** — a single count over all `O(n^2)` contiguous blocks.

This is the counting face of prefix sums: the count is enormous to enumerate directly, but collapses
to a residue-bucket tally over prefix sums. The whole difficulty is doing that tally without
double-counting pairs and without mishandling the residue of negative running sums, which is exactly
where a naive implementation goes subtly wrong.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`0 <= n <= 2*10^5`, `1 <= m <= 10^6`); the second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line may be empty or
  absent.
- Output (stdout): a single line with the number of balanced windows.
- Time limit: 1 second. Memory: 256 MB.

Example: for `m = 5` and `a = [3, 1, 4, 1, 5, -4, 6]` the answer is `5`. The balanced windows are
`[0,5]` (sum 10), `[1,2]` (sum 5), `[2,3]` (sum 5), `[2,4]` (sum 10), and `[4,4]` (sum 5).

## Background

Naively there are `n*(n+1)/2` windows — up to about `2*10^10` for the largest `n` — so enumerating
them is both too slow and produces a count that overflows 32-bit arithmetic. Two ideas are on the
table before committing:

- **Direct enumeration with a running sum.** Fix the left end `l`, extend the right end `r`,
  maintain the running sum, and test divisibility at each step. This is `O(n^2)`, obviously correct,
  and the natural reference implementation — but `4*10^10` operations is far beyond a 1-second budget.
- **Prefix-residue bucketing.** Let `P[k] = (a[0] + ... + a[k-1]) mod m` be the prefix residue, with
  `P[0] = 0` for the empty prefix. A window `[l, r]` has sum divisible by `m` exactly when
  `P[l] = P[r+1]`, so the answer is the number of unordered index pairs sharing a prefix residue.
  Counting those pairs is `O(n)` — but it is precisely the place where an off-by-one in the
  empty-prefix seed, or `c^2` instead of `c*(c-1)/2`, silently double-counts.

## Evaluation settings

Judged on hidden tests covering: `m = 1` (every window balanced, answer `= n*(n+1)/2`); arrays with
negatives and zeros (so running sums and their residues go negative); `n = 0` and `n = 1`; arrays
engineered so one residue class is very dense (stressing the pair count and 64-bit overflow); and
large `n = 2*10^5` with `m` up to `10^6` and `|a[i]|` near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count contiguous windows whose sum is divisible by m,
    //       via prefix residues, without double-counting pairs.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
