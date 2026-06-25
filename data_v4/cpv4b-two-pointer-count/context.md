# Counting compatible tuning-fork pairs by frequency gap

## Research question

A workshop has `n` tuning forks laid out on a bench, fork `i` having an integer frequency `f[i]`
(hertz). Two distinct forks `i` and `j` are **compatible** when the absolute difference of their
frequencies lands inside a fixed tolerance band: `L <= |f[i] - f[j]| <= R`. The luthier wants to know
how many **unordered** pairs of forks `{i, j}` (with `i != j`, and each pair counted once) are
compatible.

The catch is scale. There can be up to a million forks, so the number of compatible pairs can be on
the order of `n^2 / 2`, i.e. hundreds of billions — too many to enumerate, and large enough that the
count itself must be reported **modulo `1 000 000 007`**. Output that count modulo the prime.

This is the counting/constructive face of the two-pointer technique: after sorting, every fork has a
contiguous band of admissible partners, and a pair of sliding indices sweeps those bands in one pass.
The whole difficulty is arithmetic hygiene — counting each unordered pair exactly once, getting the
half-open window endpoints right, and applying the modulus only at the very end.

## Input / output contract

- Input (stdin):
  - line 1: integer `n` (`0 <= n <= 10^6`);
  - line 2: two integers `L` and `R` (`0 <= L <= R <= 2*10^9`) — the inclusive tolerance band;
  - line 3: `n` integers `f[i]` (`-10^9 <= f[i] <= 10^9`), whitespace-separated (may be empty when
    `n = 0`).
- Output (stdout): a single line with the number of compatible unordered pairs, taken modulo
  `1 000 000 007`.
- Time limit: 2 seconds. Memory: 256 MB.

Note `L` may be `0`, in which case two forks of *equal* frequency are compatible (gap `0`). Note also
that `R` can exceed any achievable gap, in which case the band's upper bound never binds.

Example: for `n = 6`, `L = 2`, `R = 5`, and `f = [10, 1, 4, 8, 13, 5]`, the answer is `8`.

## Background

Two routes are on the table before committing to one.

- **Quadratic enumeration.** Check all `C(n, 2)` pairs directly. This is obviously correct and is the
  reference oracle, but at `n = 10^6` it is `~5*10^11` operations — hopelessly over the time limit.
- **Sort, then two pointers.** Sort the frequencies. For a fixed *larger* element the set of
  admissible partners (the elements whose value sits in `[f[j] - R, f[j] - L]`) is a contiguous block
  of the sorted array, and as the larger element advances rightward both ends of that block only move
  rightward. Two indices chasing those two ends give an `O(n log n)` algorithm dominated by the sort.
  The open questions are the exact window endpoints, whether to use `<` or `<=`, and — the dangerous
  one — how to count each unordered pair once rather than twice.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1` (answer `0`); `L = 0` so equal-frequency forks
pair up; heavy duplicate values (so the count is large and ties stress the window edges); bands where
`R` never binds and bands where `L` never binds; negative frequencies; and `n = 10^6` with the count
exceeding a 64-bit-friendly range so the modulus genuinely matters.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    long long L, R;
    cin >> L >> R;
    vector<long long> f(n);
    for (auto &x : f) cin >> x;

    const long long MOD = 1000000007LL;

    // TODO: count unordered pairs {i, j} with L <= |f[i]-f[j]| <= R, output the count mod MOD.
    long long answer = 0;

    cout << answer % MOD << "\n";
    return 0;
}
```
