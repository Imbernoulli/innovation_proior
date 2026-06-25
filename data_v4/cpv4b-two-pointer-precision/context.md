# Counting balanced amplifier pairs by gain ratio

## Research question

An audio lab has `n` amplifier modules on a rack; module `i` has an integer **gain** `g[i]`
(a positive whole number of microvolts-per-microvolt). Two distinct modules `i` and `j` can be
wired in series only if their gains are *close in proportion*: writing `g_lo = min(g[i], g[j])` and
`g_hi = max(g[i], g[j])`, the pair is **balanced** when the larger gain is at most a fixed tolerance
factor of the smaller one,

```
g_hi / g_lo  <=  p / q ,
```

where the tolerance `p / q` is given as two positive integers with `p >= q` (so the bound is at
least `1`, and two modules of equal gain are always balanced — ratio `1`). The lab wants to know how
many **unordered** pairs of modules `{i, j}` (`i != j`, each pair counted once) are balanced.

Output that count. Equal-gain modules count (ratio exactly `1`). A pair whose larger gain is more
than `p/q` times the smaller one does not count.

This is the counting face of the two-pointer technique married to an **exact-ratio** test: after
sorting the gains, every module has a contiguous band of admissible partners, and a pair of sliding
indices sweeps those bands in a single linear pass. The whole difficulty is *arithmetic hygiene* —
the ratio test must be turned into a cross-multiplication, and the cross-products are large enough to
overflow ordinary 64-bit integers, so the comparison cannot be done with `double` division nor with
`long long` multiplication.

## Input / output contract

- Input (stdin):
  - line 1: three integers `n` (`0 <= n <= 2*10^5`), `p`, and `q`
    (`1 <= q <= p <= 4*10^9`) — `p/q` is the inclusive tolerance factor;
  - line 2: `n` integers `g[i]` (`1 <= g[i] <= 4*10^9`), whitespace-separated (the line may be empty
    or absent when `n = 0`).
- Output (stdout): a single line with the number of balanced unordered pairs.
- Time limit: 1 second. Memory: 256 MB.

Note `p` and `q` and the gains can exceed `2^31`, so they do not fit in a 32-bit `int`; they fit in
`long long`, but a *product* of two of them does **not** fit in a signed 64-bit integer.

Example: for `n = 6`, `p = 5`, `q = 2`, and `g = [10, 1, 4, 8, 13, 5]`, the answer is `8`. The eight
balanced pairs (by value) are `{4,10}` (ratio `10/4 = 5/2`, exactly on the bound), `{8,10}` (`5/4`),
`{10,13}` (`13/10`), `{5,10}` (`2`), `{4,8}` (`2`), `{4,5}` (`5/4`), `{8,13}` (`13/8`), and `{5,8}`
(`8/5`). Pairs such as `{1,10}` (ratio `10`) or `{1,4}` (ratio `4`) exceed `5/2` and do not count.

## Background

Sort the gains ascending. Once sorted, fix the **larger** member of a pair at index `j`; any partner
`i < j` has `g[i] <= g[j]`, so the pair is balanced exactly when

```
g[j] / g[i]  <=  p / q     <=>     g[j] * q  <=  g[i] * p .
```

For a fixed `j`, as `i` increases `g[i]` increases, so `g[i] * p` increases, so the balanced
condition gets *easier* to satisfy — the admissible partners form a suffix `[lo_j, j-1]` of the
sorted prefix. And as `j` advances, `g[j]` grows, so `g[j] * q` grows, so the left boundary `lo_j`
can only move rightward. That monotonicity is exactly what a two-pointer sweep needs.

Two routes are on the table before committing to one.

- **Quadratic enumeration.** Test all `C(n, 2)` pairs directly with the cross-multiplication. This is
  obviously correct and is the reference oracle, but at `n = 2*10^5` it is `~2*10^10` operations —
  hopelessly over the time limit.
- **Sort, then two pointers.** Sort the gains; sweep `j` from left to right while a trailing pointer
  `lo` chases the left boundary of the admissible band. `O(n log n)` dominated by the sort.

The arithmetic, not the sweep, is where this problem bites. The ratio test must never be done as
floating-point division (`g[j] / (double)g[i] <= p / (double)q`) — a `double` carries only 53 bits of
mantissa, and on pairs whose ratio sits on or near the bound the rounding can flip the verdict. Nor
can the cross-products be formed in `long long`: with values and `p, q` up to `4*10^9`, a product like
`g[j] * q` reaches `~1.6*10^19`, well past the signed 64-bit ceiling of `~9.22*10^18`, so it wraps to
a negative number and the comparison lies. The products must be formed in a 128-bit integer (or with
an equivalent overflow-safe routine), and the count itself reaches `~2*10^10`, so the accumulator must
be 64-bit too.

## Evaluation settings

Judged on hidden tests covering: the empty rack (`n = 0`) and a single module (`n = 1`), both giving
`0`; all-equal gains with `p >= q` (every one of the `C(n, 2)` pairs is balanced, stressing the
64-bit count); `p = q` so only equal-gain modules pair; tolerances so large that every pair is
balanced and so small that none are; adversarial pairs whose ratio equals the bound exactly (the
on-threshold case a `double` test misrounds); and large `n = 2*10^5` with gains and `p, q` near
`4*10^9` so the cross-products overflow signed 64-bit and the count exceeds `2^31`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long p, q;
    if (!(cin >> n >> p >> q)) return 0;
    vector<long long> g(n);
    for (auto &x : g) cin >> x;

    // TODO: sort, then two-pointer sweep counting pairs with g_hi * q <= g_lo * p,
    //       comparing the cross-products in exact integer arithmetic (no division, no double).
    long long count = 0;

    cout << count << "\n";
    return 0;
}
```
