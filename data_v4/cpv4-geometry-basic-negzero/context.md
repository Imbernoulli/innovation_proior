# Largest signed sweep between two beacons

## Research question

A surveyor stands at the origin `O = (0, 0)` and records `n` beacons as lattice points
`P[0..n-1]`, where each `P[i] = (x_i, y_i)` and coordinates may be negative or zero. For an
**ordered** pair of beacons `(i, j)` with `i < j` (the two were logged in that order), the
*signed sweep* of the pair is twice the signed area of triangle `O, P[i], P[j]`:

```
sweep(i, j) = x_i * y_j - x_j * y_i
```

This is the 2D cross product `P[i] x P[j]`. It is **positive** when going `O -> P[i] -> P[j]`
turns counter-clockwise, **zero** when `O`, `P[i]`, `P[j]` are collinear, and **negative** when
the turn is clockwise. Among all ordered pairs with `i < j`, report the **maximum** signed sweep.

The catch is that the maximum sweep may be negative (every pair could turn clockwise), or there may
be no pair at all (fewer than two beacons). Getting the one-dimensional reduction exactly right —
including the negative-sweep and no-pair corners — is the whole point.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2000`); then `n` pairs of integers
  `x_i y_i` (`-10^6 <= x_i, y_i <= 10^6`), whitespace-separated.
- Output (stdout): if `n < 2` there is no ordered pair, so print the single word `NONE`.
  Otherwise print one line with the maximum value of `sweep(i, j) = x_i*y_j - x_j*y_i` over all
  `i < j`. This value may be negative or zero.
- Time limit: 1 second. Memory: 256 MB.

Example: for the four beacons `(2,0), (0,2), (-2,0), (0,-2)` the answer is `4` (e.g. the pair
`(2,0), (0,2)` gives `2*2 - 0*0 = 4`).

## Background

The signed sweep `x_i*y_j - x_j*y_i` is the standard orientation primitive of computational
geometry: its sign tells you whether a triple turns left, straight, or right, and its magnitude is
twice the triangle area. Two angles of attack are on the table before committing:

- **Sort by polar angle, then pair extremes.** Intuitively the largest counter-clockwise sweep
  should pair the "most clockwise" direction with the "most counter-clockwise" direction. Sorting by
  `atan2(y, x)` is `O(n log n)`; the open question is whether the angular extremes actually maximize
  `x_i*y_j - x_j*y_i` subject to `i < j`, and whether floating-point angle comparisons stay exact.
- **Direct enumeration.** With `n <= 2000` there are at most about `2*10^6` ordered pairs, so simply
  evaluating `sweep(i, j)` for every `i < j` and taking the maximum is `O(n^2)` and uses only
  integer arithmetic. The open question is purely transcription: the initial value of the running
  maximum, the loop bounds, and the data type.

## Evaluation settings

Judged on hidden tests covering: pairs whose maximum sweep is strictly positive; configurations
where every ordered pair is clockwise so the maximum sweep is negative; collinear / zero-coordinate
beacons producing zero sweeps; `n = 0` and `n = 1` (must print `NONE`); duplicated points; and
near-maximum coordinates (`|x|, |y| ~ 10^6`) so a single product `x_i*y_j` reaches `10^12` and a
32-bit accumulator silently overflows.

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

    // TODO: if n < 2 print "NONE"; else print the maximum of
    //       sweep(i,j) = x[i]*y[j] - x[j]*y[i] over all i < j (may be negative).

    return 0;
}
```
