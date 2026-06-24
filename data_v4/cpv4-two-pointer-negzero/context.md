# Counting thruster pairs that clear a net-impulse threshold

## Research question

A maneuvering module carries `n` thrusters. Thruster `i` produces a signed impulse `a[i]`: a positive
value is forward thrust, a **negative** value is reverse thrust, and `0` is an idle unit. To perform a
burn the controller fires exactly **two distinct thrusters at once**, and the burn is *admissible* if
their combined impulse `a[i] + a[j]` is at least a required net impulse `T` (which may itself be
negative, zero, or positive). Order does not matter — firing thrusters `{i, j}` is the same burn as
`{j, i}`.

Count the number of admissible unordered pairs `{i, j}` with `i != j`, i.e. the number of pairs whose
sum satisfies `a[i] + a[j] >= T`. Because values may be negative and zero, and because the fleet may
be tiny (`n = 0` or `n = 1`, where **no pair exists at all**) or entirely reverse-thrust (all-negative),
the count can legitimately be `0`, and the answer can be as large as `n*(n-1)/2`.

This is a two-pointer counting problem. After sorting, two converging pointers count all qualifying
pairs in one linear sweep — but the negative/zero values, an out-of-range threshold, and the
fewer-than-two-thrusters corner are exactly where a naive implementation goes wrong.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); the second token is `T`
  (`-2*10^9 <= T <= 2*10^9`); then `n` integers `a[i]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the number of unordered pairs `{i, j}`, `i != j`, with
  `a[i] + a[j] >= T`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `T = 3`, `a = [4, -1, 0, 5, -3, 2]` the answer is `7`. After sorting to
`[-3, -1, 0, 2, 4, 5]`, the qualifying pairs are `{-1,4}, {-1,5}, {0,4}, {0,5}, {2,4}, {2,5}, {4,5}`.

## Background

The constraint `a[i] + a[j] >= T` couples two indices, so a brute force is `O(n^2)`: enumerate every
pair and test it. That is fine for tiny inputs but dies at `n = 2*10^5` (about `2*10^10` pairs). Two
families of faster approach are on the table before committing to one:

- **Sort + binary search.** Sort ascending; for each index `i`, the partners `j > i` that qualify are
  those with `a[j] >= T - a[i]`, a contiguous suffix locatable by `lower_bound`. Summing the suffix
  lengths gives the count in `O(n log n)`. The open question is the off-by-one in where the suffix
  starts and whether the `j > i` (versus `j != i`) bookkeeping double-counts.
- **Sort + two pointers.** Sort ascending and run two converging pointers `lo` and `hi`. Whenever
  `a[lo] + a[hi] >= T`, every index in `[lo, hi-1]` paired with `hi` also qualifies (the array is
  sorted), so they contribute `hi - lo` pairs at once; otherwise `a[lo]` is too small for the current
  `hi` and `lo` advances. This is `O(n log n)` for the sort plus `O(n)` for the sweep. The open
  questions are the exact count increment (`hi - lo` versus `hi - lo - 1`) and the base case when
  `n < 2`.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays; arrays mixing negatives, zeros, and positives;
the empty array (`n = 0`) and single thruster (`n = 1`), both of which have **no pair** and must print
`0`; all-negative arrays with a threshold no pair can reach (answer `0`) and with a very negative
threshold every pair clears (answer `n*(n-1)/2`); all-zero arrays at `T = 0` (every pair qualifies);
thresholds outside the reachable sum range on both ends; and large `n = 2*10^5` with `|a[i]|` near
`10^9` and `|T|` near `2*10^9`, so individual pair sums and the running count both exceed the 32-bit
range.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    long long T;
    cin >> T;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count unordered pairs {i, j}, i != j, with a[i] + a[j] >= T.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
