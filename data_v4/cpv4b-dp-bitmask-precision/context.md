# Heaviest gear train (maximum-product Hamiltonian ordering by bitmask DP)

## Research question

A watchmaker is assembling a single gear train out of `n` distinct gears laid in a line. Every gear is
used exactly once, so an assembly is an **ordering** (a permutation) `v0, v1, ..., v_{n-1}` of the `n`
gears. The first gear in the line spins at a base rate `b[v0]` (an integer "torque seed"). Each time
gear `j` is placed immediately to the right of gear `i`, the train's accumulated figure of merit is
**multiplied** by an integer meshing factor `m[i][j]` — the two specific teeth profiles mesh that much
more strongly in that direction. The figure of merit of the whole assembly is therefore the product

```
M = b[v0] * m[v0][v1] * m[v1][v2] * ... * m[v_{n-2}][v_{n-1}].
```

The watchmaker wants the ordering that **maximizes `M`**, and wants the exact value of that maximum —
not an approximation. Output `M` for the best ordering.

This is a maximum-product Hamiltonian-path problem. With `n` small it is the textbook setting for a
**bitmask dynamic program** over the set of already-placed gears, but the multiplicative objective is
the catch: `M` grows so fast that it overruns 64-bit arithmetic, and "exact value" forbids any
floating-point or log-domain shortcut. Getting the integer width and the comparison right is the whole
problem.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 13`). Then `n` integers `b[i]`
  (`1 <= b[i] <= 60`), the base rates. Then `n` lines each with `n` integers; the `j`-th integer on
  line `i` is `m[i][j]` (`1 <= m[i][j] <= 60`). The diagonal entries `m[i][i]` are present in the
  input but never used (a gear is never placed next to itself). All tokens are whitespace-separated.
- Output (stdout): a single line with the maximum figure of merit `M` of any ordering, as an exact
  decimal integer (it can have up to ~24 digits, so it does not fit in a 64-bit type).
- For `n = 0` the only assembly is empty; output `0`. For `n = 1` the assembly is the single gear and
  `M = b[0]`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for

```
4
3 4 2 5
1 6 4 4
5 1 3 7
6 8 1 5
4 9 7 1
```

the answer is `1400`. The best ordering is gears `4, 3, 2, 1` (1-indexed): start at gear 4 with base
`b[3] = 5`, then `m[3][2] = 7` (gear 3 after gear 4), `m[2][1] = 8` (gear 2 after gear 3), and
`m[1][0] = 5` (gear 1 after gear 2), giving `5 * 7 * 8 * 5 = 1400`. No other ordering does better.

## Background

With `n <= 13`, enumerating all `n!` orderings is `13! ≈ 6.2*10^9` — too slow. The standard tool is a
DP indexed by a **bitmask of which gears are already placed**, together with the last gear placed:
`dp[mask][last]` is the best (largest) figure of merit of a sub-train that uses exactly the gears in
`mask` and ends at `last`. A single gear `v` seeds `dp[1<<v][v] = b[v]`; extending a sub-train by
appending an unused gear `nxt` multiplies by `m[last][nxt]`. The answer is the best `dp[full][last]`
over all `last`, where `full = (1<<n) - 1`. This is `O(2^n * n^2)` time and `O(2^n * n)` states.

Two representation questions decide correctness before any code is written:

- **How wide must the accumulator be?** The product of `n - 1` factors of up to `60`, times a base of
  up to `60`, reaches `60^13 ≈ 1.3*10^23`. A signed 64-bit integer caps near `9.2*10^18`, so the
  product overflows by four orders of magnitude on the largest inputs. The open question is which
  integer type survives.
- **How should two candidate products be compared?** The DP merges sub-trains by keeping the larger
  product at each `(mask, last)`. One could store `log` of the product in a `double` and compare logs,
  which would sidestep the width problem — but the output demands the *exact* integer, and at this
  magnitude a `double` cannot even represent the answer. The open question is whether a floating-point
  representation can ever be trusted here.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1` corners; tiny cases (`n <= 4`) checkable by hand;
all-equal large factors (so the product lands exactly on `60^n`-scale values that overflow 64 bits);
mixed cases where the best ordering ends at a non-first gear; and full-size `n = 13` with factors near
the cap, so any 64-bit accumulator silently wraps and any floating-point comparison loses the low
digits. Outputs are compared as exact decimal strings.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Print a non-negative __int128 in decimal.
static string i128_to_string(__int128 x) {
    if (x == 0) return "0";
    bool neg = false;
    if (x < 0) { neg = true; x = -x; }
    string s;
    while (x > 0) { int d = (int)(x % 10); s.push_back(char('0' + d)); x /= 10; }
    if (neg) s.push_back('-');
    reverse(s.begin(), s.end());
    return s;
}

int main() {
    int n;
    if (!(cin >> n)) return 0;                 // empty input -> treat as n = 0
    vector<long long> b(n);
    for (auto &x : b) cin >> x;
    vector<vector<long long>> m(n, vector<long long>(n, 0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> m[i][j];

    // TODO: bitmask DP over placed gears; maximize the exact product; print it.
    __int128 best = 0;

    cout << i128_to_string(best) << "\n";
    return 0;
}
```
