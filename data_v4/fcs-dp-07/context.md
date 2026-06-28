# Counting integers whose digit sum is divisible by their digit count

## Research question

For an integer `x` written in decimal with no leading zeros, let `len(x)` be its number of digits and
`S(x)` be the sum of its digits. Call `x` **balanced** when `S(x)` is divisible by `len(x)`, i.e.
`S(x) mod len(x) == 0`. Given a range `[L, R]`, count how many integers in that range are balanced.

The catch that makes this more than a textbook exercise is that the divisor is **not fixed**: it is the
number of digits of the candidate number itself. So `19` (two digits, digit sum `10`) is balanced
because `2 | 10`, but `1900000000000000000` (nineteen digits, digit sum `10`) is *not* balanced because
`19` does not divide `10`. The property being counted is **non-local**: whether a number qualifies
depends on a global feature (its length) interacting with an aggregate of all its digits (their sum).
This is the kind of "the rule changes with the size of the object" constraint that breaks naive
per-digit counting and forces a more careful decomposition.

## Input / output contract

- Input (stdin): a single line with two integers `L` and `R` (`1 <= L <= R <= 10^18`), space-separated.
- Output (stdout): a single line with one integer — the count of balanced integers in `[L, R]`.
- Time limit: 1 second. Memory: 256 MB.

Worked example: for `L = 1`, `R = 20` the answer is `15`.
- The single-digit numbers `1..9` all have `len = 1`, and every integer is divisible by `1`, so all `9`
  of them are balanced.
- Among `10..20` (all `len = 2`, divisor `2`, so "even digit sum"): `11, 13, 15, 17, 19, 20` have even
  digit sums (`2, 4, 6, 8, 10, 2`). That is `6` more.
- Total `9 + 6 = 15`.

## Background

The answer for a range `[L, R]` is the difference of two prefix counts: `f(R) - f(L-1)`, where `f(N)` is
the number of balanced integers in `[1, N]`. So the whole problem reduces to computing `f(N)` for a
single bound `N` up to `10^18`.

`N` can be as large as `10^18`, so any approach that visits each integer one by one is hopeless: at
`10^9` operations per second a scan to `10^18` would take on the order of `10^9` seconds. The standard
tool for "count numbers `<= N` with some digit property" is **digit dynamic programming**, which walks
the decimal representation position by position and carries a small amount of state (current position, a
running residue, and a flag for whether the prefix is still equal to the bound's prefix). The total work
is then proportional to (number of digits) × (size of the carried state), independent of `N`'s
magnitude.

The twist here is that the modulus the digit sum must be divisible by depends on the number's length,
which the digit DP does not normally know until it has committed to a length. The background question
this trace must resolve is how to make digit DP carry the *right* modulus when that modulus is itself a
function of the unknown length.

## Evaluation settings

Judged on hidden tests covering: tiny ranges; ranges that straddle a power-of-ten boundary (where the
length, and therefore the divisor, changes mid-range); single-point ranges `L = R`; the extreme upper
bound `R = 10^18`; and ranges chosen so the answer exceeds the 32-bit integer range (the count over
`[1, 10^18]` is about `5.6 * 10^16`, so the output and all internal accumulators must be 64-bit).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Count balanced integers in [1, N]: those whose decimal digit sum is divisible
// by their number of digits. N >= 0; returns 0 for N <= 0.
ll countUpTo(ll N) {
    // TODO: count integers x in [1, N] with digitSum(x) % numDigits(x) == 0.
    return 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    ll L, R;
    if (!(cin >> L >> R)) return 0;
    ll answer = countUpTo(R) - countUpTo(L - 1);
    cout << answer << "\n";
    return 0;
}
```
