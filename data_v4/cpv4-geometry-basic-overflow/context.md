# Largest triangle from a set of points (twice the area)

## Research question

You are given `n` points in the plane with integer coordinates. Among all unordered triples of
distinct points, find the triangle of **maximum area**, and report **twice that area** as an exact
integer. Twice the area of a triangle on integer coordinates is always a non-negative integer (it is
the absolute value of a cross product), so the answer is exact and no floating point is needed. If no
triple exists (`n < 3`), or every triple is degenerate (collinear / coincident), the answer is `0`.

This is the "largest triangle" subproblem of computational geometry. Brute force over all triples is
the textbook approach for moderate `n`, but the arithmetic is the trap: with coordinates as large as
`10^9`, the cross product that computes the area overflows a 32-bit integer by many orders of
magnitude. Getting the exact-integer area right — and choosing the data type that holds it — is the
whole point.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`). Then `n` lines (or whitespace-separated
  pairs) follow, each `x[i] y[i]` with `-10^9 <= x[i], y[i] <= 10^9`. Points may coincide.
- Output (stdout): a single line with **twice** the maximum triangle area, as an integer.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the four corners of a `10^9 x 10^9` axis-aligned square together with its centre,

```
5
0 0
1000000000 0
1000000000 1000000000
0 1000000000
500000000 500000000
```

the answer is `1000000000000000000`. Three corners of the square form a right triangle with legs
`10^9` and `10^9`; its area is `10^18 / 2`, so twice the area is `10^18`. (No triple beats that.)

## Background

For three points `A`, `B`, `C`, twice the signed area of triangle `ABC` is the 2-D cross product

```
cross = (B - A) x (C - A) = (Bx - Ax) * (Cy - Ay) - (Cx - Ax) * (By - Ay).
```

Twice the *area* is `|cross|`. Two facts make this convenient: it is an integer for integer inputs
(no division), and its sign tells orientation (positive = counter-clockwise), though here only the
magnitude matters. So the maximum-area triangle is the triple maximizing `|cross|`.

Two routes are on the table before committing to one:

- **Brute force over all triples.** Try every `i < j < k`, compute `|cross|`, keep the maximum.
  `O(n^3)`; with `n <= 500` that is at most `~2 * 10^7` triples, comfortably inside the time limit.
  Simple and obviously correct — the only open question is the arithmetic.
- **Convex-hull + rotating ideas.** The largest-area triangle has all three vertices on the convex
  hull, so one could hull first and search the hull. Faster asymptotically, but more code and more
  places to be subtly wrong; for `n <= 500` it buys nothing.

## Evaluation settings

Judged on hidden tests covering: `n < 3` (answer `0`), all-collinear point sets (answer `0`),
coincident/duplicate points, small random clusters, and large `n = 500` with coordinates near
`+/- 10^9` so that the cross product reaches roughly `8 * 10^18` — far outside 32-bit range and into
the regime where the data type is the whole game.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    // TODO: over all triples i<j<k, compute twice the triangle area via the
    // cross product and keep the maximum (0 if no non-degenerate triple).
    long long best = 0;

    cout << best << "\n";
    return 0;
}
```
