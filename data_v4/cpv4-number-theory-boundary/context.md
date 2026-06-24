# Resonance-free teeth on a gear track

## Research question

A long linear gear track has teeth at the integer positions `1, 2, 3, ...`. A drive wheel with `m`
teeth meshes with the track. A track position `x` is **resonance-free** for this wheel exactly when
`gcd(x, m) = 1` — when the position shares no factor with the wheel size, the contact never lands on
a repeating beat. You are handed `q` independent inspection windows `[L, R]` (both endpoints
**inclusive**), and for each window you must report how many resonance-free positions it contains,
i.e.

```
count of x with L <= x <= R and gcd(x, m) = 1.
```

This is a counting-in-a-range problem on the coprime-to-`m` predicate. It is the kind of subproblem
that shows up inside Euler-totient summations, Mobius inclusion-exclusion, and "count numbers with a
divisibility property in a range" tasks — so getting the *boundary* exactly right (inclusive `R`,
inclusive `L`, and the `L = 1` corner) is the whole game.

## Input / output contract

- Input (stdin): the first line has two integers `m` and `q`
  (`1 <= m <= 10^12`, `1 <= q <= 2*10^5`). Then `q` lines follow, each with two integers `L` and `R`
  (`1 <= L <= R <= 10^18`).
- Output (stdout): `q` lines; line `i` is the number of resonance-free positions in the `i`-th window,
  i.e. the count of `x` in `[L, R]` with `gcd(x, m) = 1`.
- Time limit: 3 seconds. Memory: 256 MB.

Example: for `m = 12` and windows `[1, 12]`, `[5, 5]`, `[7, 24]` the answers are `4`, `1`, `6`.
The numbers coprime to `12` in `[1, 12]` are `1, 5, 7, 11` (four of them); `5` alone is coprime to
`12` (one); and in `[7, 24]` they are `7, 11, 13, 17, 19, 23` (six).

## Background

The predicate `gcd(x, m) = 1` is not monotone in `x`, so there is no shortcut that scans only the
endpoints. Two families of approach are on the table before committing to one:

- **Direct scan per window.** For each window loop `x` from `L` to `R` and test `gcd(x, m) = 1`. It is
  trivial to write and obviously correct, but it is `O((R - L + 1))` per query. With `R - L` up to
  `10^18` this is hopeless for the real bounds; it survives only as a brute-force oracle on tiny
  inputs.
- **Prefix-count via inclusion-exclusion (Mobius).** Define `C(N)` = the count of integers in
  `[1, N]` that are coprime to `m`. Then the answer for a window is a *difference of prefix counts*,
  and `C(N)` itself is an inclusion-exclusion over the **distinct prime factors** of `m`:
  `C(N) = sum over squarefree divisors d of m of mu(d) * floor(N / d)`. Factor `m` once
  (`m <= 10^12`, so trial division to `10^6` suffices); each prime set has at most ~11 distinct
  primes, so there are at most `2^11` signed terms. The open questions are (1) the *exact* difference
  of prefix counts that turns the half-open prefix into the closed window `[L, R]`, and (2) the
  `N = 0` corner that the difference produces when `L = 1`.

## Evaluation settings

Judged on hidden tests covering: `m = 1` (everything is coprime, every window answer is `R - L + 1`);
prime `m` and prime-power `m`; highly composite `m` (product of many small primes, the maximum number
of inclusion-exclusion terms); single-point windows `L = R`; windows anchored at the left edge `L = 1`
(which forces the `L - 1 = 0` prefix); the full magnitude `R = 10^18` (so prefix counts and the
running accumulator must be 64-bit); and the maximum `q = 2*10^5` queries against a worst-case
multi-prime `m`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long m;
    int q;
    if (!(cin >> m >> q)) return 0;

    // TODO: factor m into its distinct primes; for each query [L, R] count the
    // integers in the *inclusive* window that are coprime to m, using a
    // difference of prefix counts computed by Mobius inclusion-exclusion.

    while (q--) {
        long long L, R;
        cin >> L >> R;
        long long answer = 0;
        cout << answer << "\n";
    }
    return 0;
}
```
