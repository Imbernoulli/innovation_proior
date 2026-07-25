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
