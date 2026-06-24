# Distributing candies with a per-child cap, counted modulo a prime

## Research question

A teacher has `S` identical candies and `k` distinguishable children. Each child may receive between
`0` and `c` candies **inclusive**. Two distributions are different if some child receives a different
number of candies. Count the number of valid distributions — that is, the number of integer solutions
to

```
x_1 + x_2 + ... + x_k = S,   with 0 <= x_i <= c  for every i,
```

and report that count modulo a given prime `M`.

This is the classic "bounded composition" / "stars-and-bars with an upper cap" count. The unbounded
version (`c = infinity`) is the textbook `C(S + k - 1, k - 1)`; the cap turns it into an
inclusion-exclusion over how many children break the ceiling. The whole difficulty lives at the
*boundary of that inclusion-exclusion sum*: how far the alternating sum runs, and whether the term
that drives the remaining candy count to exactly `0` is included or excluded.

## Input / output contract

- Input (stdin): four whitespace-separated integers on one line:
  `k c S M`, where
  - `0 <= k <= 10^6`   (number of children),
  - `0 <= c <= 10^9`   (per-child cap),
  - `0 <= S <= 10^6`   (total candies),
  - `M` is a prime with `M > S + k`  (so the largest binomial argument the natural solution forms,
    `S + k - 1`, is strictly below `M`; `M <= 10^9 + 7`).
- Output (stdout): a single line with the number of valid distributions modulo `M`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `k = 4`, `c = 5`, `S = 12`, `M = 1000000007` the answer is `125` (the number of
4-tuples from `{0,...,5}` that sum to 12).

## Background

Two facts frame the approach:

- **Stars and bars (no cap).** The number of nonnegative integer solutions to
  `y_1 + ... + y_k = T` is `C(T + k - 1, k - 1)`. This is exact and modular-friendly once factorials
  and their inverses are precomputed.
- **Inclusion-exclusion for the cap.** "Child `i` overflows" means `x_i >= c + 1`. Subtracting
  `c + 1` from an overflowing child turns the cap constraint into an unbounded count on a smaller
  total. By inclusion-exclusion over the set of overflowing children,

  ```
  answer = sum_{j} (-1)^j * C(k, j) * C(S - j*(c+1) + k - 1, k - 1),
  ```

  where the sum runs over every `j` for which the remaining total `S - j*(c+1)` is still **>= 0**.
  Each such `j` chooses which `j` children are *forced* to overflow (the `C(k, j)` factor) and then
  counts the unbounded distributions of what is left.

The danger is entirely at the summation boundary. The remainder `S - j*(c+1)` must be allowed to reach
exactly `0` (that is a genuine configuration — every overflow removed lands the leftover at zero), so
the stopping test is `>= 0`, not `> 0`; and `j` may never exceed `k`. An off-by-one here — stopping one
term early, or one term late, or using a strict inequality — corrupts the alternating sum.

## Evaluation settings

Judged on hidden tests covering: `S = 0` (exactly one distribution, all zeros); `S = k*c` (exactly one,
everyone maxed); `S > k*c` (impossible, answer `0`); `c = 0` (only `S = 0` is feasible); `k = 0`
(feasible iff `S = 0`); totals landing exactly on a multiple of `c+1` (the boundary term of the
inclusion-exclusion); large `c >= S` (cap never binds, only the `j = 0` term survives); and large
`k, S ~ 10^6` so the factorial table and the alternating sum are both stressed. Both small primes
(that still exceed `S + k`) and `10^9 + 7` are used to exercise the modular reduction.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    long long k, c, S, M;
    if (!(cin >> k >> c >> S >> M)) return 0;

    // TODO: count integer solutions of x_1 + ... + x_k = S with 0 <= x_i <= c, modulo prime M,
    // via inclusion-exclusion over the number of children that exceed the cap.
    long long answer = 0;

    cout << answer % M << "\n";
    return 0;
}
```
