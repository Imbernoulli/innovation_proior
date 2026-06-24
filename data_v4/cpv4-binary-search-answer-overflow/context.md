# Stamping-press line: earliest time to reach a quota

## Research question

A factory line has `m` stamping presses running in parallel. Press `i` needs a fixed **warm-up
delay** `w[i]` milliseconds before it can emit its *first* part; after that it emits one part every
`c[i]` milliseconds. So by elapsed time `T` (milliseconds, measured from when the line is switched
on) press `i` has produced

- `0` parts if `T < w[i]`, and
- `floor((T - w[i]) / c[i]) + 1` parts if `T >= w[i]` (the `+1` counts the part stamped exactly at
  the moment `w[i]`).

All presses run simultaneously and independently. Given a quota `N`, find the **smallest** time `T`
such that the presses together have produced at least `N` parts. Output that `T`.

This is a binary-search-on-the-answer problem: the total output is monotonically non-decreasing in
`T`, so "is `N` reached by time `T`?" is a monotone predicate and we search for its threshold. The
value of `T` itself — not just intermediate sums — can be astronomically large, which is the corner
the problem is built to expose.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `N`
  (`1 <= m <= 10^5`, `0 <= N <= 10^9`). Then `m` lines follow; line `i` has two integers `w[i]` and
  `c[i]` (`0 <= w[i] <= 10^9`, `1 <= c[i] <= 10^9`).
- Output (stdout): a single line with the smallest `T` (milliseconds) at which total production is at
  least `N`. If `N = 0` the answer is `0` (the quota is already met before anything is stamped).
- Time limit: 1 second. Memory: 256 MB.

Example: for `m = 3`, `N = 10` and presses `(w, c)` = `(0, 3), (2, 5), (1, 2)`, the answer is `9`.
At `T = 8` the presses have produced `3 + 2 + 4 = 9` parts (`< 10`); at `T = 9` they have produced
`4 + 2 + 5 = 11` parts (`>= 10`), and no earlier time reaches `10`.

## Background

The quantity `produced(T) = sum_i [T >= w[i]] * (floor((T - w[i]) / c[i]) + 1)` is non-decreasing in
`T`: raising `T` can only turn presses on and add more cycles, never remove parts. That monotonicity
is exactly what a binary search on the answer needs. Two design questions remain before committing:

- **What is a safe upper bound `hi`?** We need a `T` that is *certainly* enough so the search has a
  right end. A single press alone reaches `N` parts by time `w[i] + (N - 1) * c[i]`; the maximum of
  that over all presses is a valid (loose) upper bound, since the real line is at least as fast as
  its fastest single press.
- **How big do the numbers get?** With `N` up to `10^9` and `c[i]` up to `10^9`, the bound
  `w[i] + (N - 1) * c[i]` is on the order of `10^18`. The answer `T`, the search endpoints, and the
  product `(N - 1) * c[i]` all live far outside 32-bit range, while `produced(T)` summed across
  presses can also exceed `10^9`. Every one of those must be 64-bit, and the running total inside
  `produced` should be guarded against overflow even in 64-bit.

## Evaluation settings

Judged on hidden tests covering: small hand-checkable lines; `N = 0` (answer `0`); `N = 1` (answer is
the minimum warm-up `min_i w[i]`); a single press (answer `w[0] + (N - 1) * c[0]`, ~`10^18`); many
presses with tiny cycle times (total output overflows 32-bit at the threshold); and the full-scale
`m = 10^5`, `N = 10^9`, `w[i], c[i]` near `10^9` (so the answer is ~`10^18` and an `int` endpoint or
accumulator silently overflows).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int m;
    long long N;
    if (!(cin >> m >> N)) return 0;
    vector<long long> w(m), c(m);
    for (int i = 0; i < m; i++) cin >> w[i] >> c[i];

    // TODO: binary search the smallest time T with produced(T) >= N, where
    // produced(T) = sum over presses of (T >= w[i] ? (T - w[i]) / c[i] + 1 : 0).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
