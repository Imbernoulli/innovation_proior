# Counting domino tilings of a 3 x N board, modulo m

## Research question

A `3 x N` board is to be completely covered by `1 x 2` dominoes (each domino covers two cells that
share an edge; dominoes may be placed horizontally or vertically; no overlaps, no cells left
uncovered, nothing sticks off the board). Let `f(N)` be the number of distinct ways to do this.
Because `N` can be astronomically large, you do not report `f(N)` itself but its remainder modulo a
given integer `m`. Output `f(N) mod m`.

When `N` is odd the board has `3*N` cells, which is odd, while every domino covers exactly two
cells; a complete cover is then impossible and `f(N) = 0`.

## Input / output contract

- Input (stdin): a single line with two integers `N` and `m`.
  - `0 <= N <= 10^18`
  - `1 <= m <= 10^9`
- Output (stdout): a single line with `f(N) mod m`.
- Time limit: 1 second. Memory: 256 MB.

Note that `m` need not be prime, and `m` may equal `1` (in which case every remainder is `0`).

Example:

```
Input
4 1000000007
Output
11
```

(The `3 x 4` board has 11 tilings: `1, 3, 11, 41, 153, ...` is the sequence of tiling counts for
`N = 0, 2, 4, 6, 8, ...`.)

## Background

The small cases are easy to obtain by hand or by a tiny exhaustive search, and they look extremely
regular:

| N    | 0 | 1 | 2 | 3 | 4  | 5 | 6  | 7 | 8   | 9 | 10  |
|------|---|---|---|---|----|---|----|---|-----|---|-----|
| f(N) | 1 | 0 | 3 | 0 | 11 | 0 | 41 | 0 | 153 | 0 | 571 |

So the non-zero values `1, 3, 11, 41, 153, 571, ...` march along in a tidy pattern. The challenge is
that the constraints push `N` all the way to `10^18`, far beyond anything you can fill in a table or
iterate column by column. Two ingredients are on the table before committing to an implementation:

- **A linear recurrence over even indices.** The even-index counts appear to satisfy a fixed-order
  recurrence with constant coefficients; if so, the value at index `N` is a linear function of a
  bounded window of earlier values, which is the gateway to logarithmic-time evaluation.
- **The column-profile transfer matrix.** Tilings of a `3 x N` strip can be built one column at a
  time, where the only state carried between columns is which rows have a horizontal domino
  protruding into the next column. There are `2^3 = 8` such profiles, and the number of tilings is an
  entry of the `N`-th power of an `8 x 8` non-negative integer matrix. The open question is exactly
  which matrix, and how to keep the whole computation correct modulo a possibly-composite `m`.

## Evaluation settings

Judged on hidden tests covering: the tiny cases `N = 0, 1, 2`; odd `N` (answer `0`); moderate even
`N` where a direct count is still feasible for a checker; `m = 1` (answer `0`); prime moduli such as
`10^9 + 7` and `998244353`; composite moduli; and large `N` up to `10^18` paired with various moduli
(so any method that materializes or iterates over `N` columns is far too slow, and any precomputed
table of small answers is simply absent from the relevant range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long N, m;
    if (!(cin >> N >> m)) return 0;

    // TODO: compute the number of domino tilings of a 3 x N board, modulo m.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
