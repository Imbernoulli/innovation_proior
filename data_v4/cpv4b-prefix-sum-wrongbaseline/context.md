# Counting settlement windows in a cash ledger (sums divisible by m)

## Research question

A small co-op keeps a daily ledger of net cash flow. Day `i` has a signed integer `a[i]`: a positive
value is money taken in, a negative value is money paid out, and zero is a flat day. At the end of a
quarter the `m` partners want to find every contiguous stretch of days `[l, r]` whose **total net flow
is divisible by `m`**, because exactly those stretches can be split evenly among the `m` partners with
no remainder owed. A stretch is identified by its endpoints, so `[2, 5]` and `[2, 6]` are different
stretches even if they happen to total the same amount.

Count how many contiguous windows `[l, r]` (with `0 <= l <= r <= n-1`) have a sum divisible by `m`.

This is the "subarray sum divisible by `m`" counting problem, but the ledger is signed: a running
total can dip negative, sit at zero, and climb back, so any method that assumes non-negative running
sums has to be checked rather than trusted.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`0 <= n <= 2*10^5`, `1 <= m <= 2*10^5`). The second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty or absent.
- Output (stdout): a single line with the number of windows whose sum is divisible by `m`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `m = 4`, `a = [3, 1, -4, 2, -2, 4]` the answer is `10`.

## Background

The brute-force definition is an `O(n^2)` double loop: fix `l`, extend `r`, keep a running window sum,
and increment a counter whenever the sum is a multiple of `m`. That is obviously correct but far too
slow at `n = 2*10^5`.

The textbook speedup is a **prefix-sum** identity. Let `P[0] = 0` and `P[k] = a[0] + ... + a[k-1]`.
The sum of window `[l, r]` is `P[r+1] - P[l]`, and that difference is divisible by `m` exactly when
`P[r+1]` and `P[l]` leave the **same remainder modulo `m`**. So if we bucket the prefix sums by their
remainder and, for each new prefix, add the number of earlier prefixes sharing its remainder, we get
the count in `O(n)`. Two questions are open before committing:

- **Does the standard "bucket by `prefix % m`" code transcribe correctly here?** The identity is about
  the mathematical remainder in `{0, ..., m-1}`. The ledger is signed, so prefix sums go negative, and
  the language's `%` operator does not have to agree with the mathematical remainder on negatives.
- **What can the counts and sums reach?** With `n` windows nesting, the count can be quadratic in `n`,
  and the running total spans a wide signed range; both dictate the integer width.

## Evaluation settings

Judged on hidden tests covering: all-positive ledgers, ledgers mixing negatives, zeros, and positives
(so prefix sums go negative), `m = 1` (every window qualifies, count `= n(n+1)/2`), the empty ledger
(`n = 0`), a single day, ledgers where many prefixes collide in one remainder bucket, and large
`n = m = 2*10^5` with `|a[i]|` near `10^9` (so the running prefix and the final count each exceed a
32-bit integer).

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

    // TODO: count contiguous windows [l, r] whose sum is divisible by m,
    //       using prefix-sum remainders bucketed mod m.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
