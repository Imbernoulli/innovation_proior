# Counting equal-net stretches in a pirate's gold ledger

## Research question

A pirate keeps a daily ledger `a[0..n-1]` of how much gold changed hands each day: a positive value
is a haul, a negative value a loss, and a zero a quiet day (values may be negative or zero). The
quartermaster fixes a target `S` and calls a contiguous run of days `[l, r]` (with
`0 <= l <= r <= n-1`, so the run is **non-empty**) a **matching stretch** when its net change
`a[l] + a[l+1] + ... + a[r]` equals **exactly** `S`. He wants to know **how many matching stretches
the ledger contains** — a single count over all `O(n^2)` contiguous, non-empty runs.

This is the counting face of prefix sums. The count can be astronomically large to enumerate
directly, yet it collapses to a hash-table tally over prefix sums: a stretch `[l, r]` sums to `S`
exactly when two prefix sums differ by `S`. The entire difficulty is performing that tally **without
double-counting** — in particular without ever matching a length-0 run, and without seeding the
empty prefix twice — which is precisely where a naive implementation goes subtly wrong.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `S`
  (`0 <= n <= 2*10^5`, `-10^14 <= S <= 10^14`); the second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line may be empty or
  absent.
- Output (stdout): a single line with the number of matching stretches.
- Time limit: 1 second. Memory: 256 MB.

Example: for `S = 2` and `a = [3, -1, 1, 2, -2, 2, 1]` the answer is `6`. The matching stretches are
`[0,1]` (sum `3-1=2`), `[1,3]` (sum `-1+1+2=2`), `[1,5]` (sum `-1+1+2-2+2=2`), `[3,3]` (sum `2`),
`[3,5]` (sum `2-2+2=2`), and `[5,5]` (sum `2`).

## Background

Naively there are `n*(n+1)/2` non-empty runs — up to about `2*10^10` for the largest `n` — so
enumerating them is both too slow and produces a count that overflows 32-bit arithmetic. Two ideas
are on the table before committing:

- **Direct enumeration with a running sum.** Fix the left end `l`, extend the right end `r`,
  maintain the running sum, and test equality with `S` at each step. This is `O(n^2)`, obviously
  correct, and the natural reference implementation — but `~4*10^10` operations is far beyond a
  1-second budget.
- **Prefix-sum hashing.** Let `P[0] = 0` (the empty prefix) and `P[k] = a[0] + ... + a[k-1]`. A run
  `[l, r]` has sum `S` exactly when `P[r+1] - P[l] = S`, i.e. `P[l] = P[r+1] - S`. So as the right
  edge sweeps `r+1 = 1..n`, the answer grows by the number of earlier left endpoints `l <= r` whose
  prefix sum equals `P[r+1] - S`. Maintaining a frequency map of seen prefix sums makes this `O(n)`.
  The whole trap lives in the *order of query and insert* (so a length-0 run is never counted) and
  in seeding `P[0]` exactly once.

## Evaluation settings

Judged on hidden tests covering: `S = 0` (the densest case, where the empty-prefix self-match is the
classic double-count); all-zero arrays (every run matches, answer `= n*(n+1)/2`, which overflows
32-bit); arrays with negatives and zeros (so prefix sums and the needed `P[r+1] - S` go negative);
`n = 0` and `n = 1`; a target `S` that is unreachable (answer `0`); a single very dense prefix-sum
class (stressing the 64-bit pair count); and large `n = 2*10^5` with `|a[i]|` near `10^9` (so prefix
sums reach `~2*10^14`, well outside 32-bit but inside 64-bit).

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

    // TODO: count non-empty contiguous runs whose sum equals S, via a frequency
    //       map of prefix sums, without double-counting (no length-0 match,
    //       empty prefix seeded exactly once).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
