# Counting tiled prefixes of a string

## Research question

You are given a lowercase string `s` of length `n`. For a length `L` (`1 <= L <= n`) consider the
prefix `s[0..L-1]`. Call this prefix **tiled** if it is exactly some shorter block repeated **two or
more** times: there is a tile length `d` with `1 <= d < L`, `d` divides `L`, and the prefix equals
`s[0..d-1]` written `L/d` times (equivalently `s[i] == s[i-d]` for all `d <= i < L`). For a tiled
prefix its **minimal tile length** is the smallest such `d`.

Report two numbers:

1. how many of the `n` prefixes are tiled, and
2. the sum, over all tiled prefixes, of their minimal tile length.

A prefix that is *not* the repetition of any strictly shorter block (for example a single character,
or `abc`, or `aab`) is not tiled and contributes nothing. This is the periodicity question that sits
under string-compression, run detection, and pattern-matching code, so the boundary between "a border
that happens to exist" and "a period that actually tiles the whole length" has to be drawn exactly.

## Input / output contract

- Input (stdin): a single line containing the string `s` (`1 <= n <= 10^6`), consisting only of
  lowercase English letters `a`..`z`.
- Output (stdout): one line with two space-separated integers: the count of tiled prefixes and the
  sum of their minimal tile lengths.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = "abcabcabc"` the answer is `2 6`. The only tiled prefixes are `abcabc` (length 6,
minimal tile `abc`, `d = 3`) and `abcabcabc` (length 9, minimal tile `abc`, `d = 3`); the count is
`2` and the tile-length sum is `3 + 3 = 6`. The prefixes of length 1, 2, 3, 4, 5, 7, 8 are each
aperiodic in the "two-or-more whole copies" sense.

## Background

The naive test — for every length `L`, try every divisor `d` and compare `L/d` copies — is
`O(n^2)` or worse and cannot survive `n = 10^6`. The structural fact that makes it linear is the
**Knuth–Morris–Pratt failure function** `pi`. Define `pi[L]` as the length of the longest *proper*
prefix of `s[0..L-1]` that is also a suffix of `s[0..L-1]` (a *border*). The classic periodicity
lemma states that the **shortest period** of the length-`L` prefix is exactly `d = L - pi[L]`, and a
period `d` tiles the prefix into whole copies precisely when `d` divides `L`.

Two candidate framings are on the table before committing:

- **Per-length divisor scan.** For each `L`, factor `L` and test each divisor as a candidate period.
  Correct but `O(n * sigma(L))` aggregate; too slow at `10^6` and easy to get the comparison wrong.
- **Single failure-function pass.** Build `pi` once in `O(n)`, then for each `L` test the single
  candidate `d = L - pi[L]` against `d < L` and `L % d == 0`. `O(n)` overall; the open questions are
  the exact indexing of `pi` (by prefix *length*, 1-based, vs. by 0-based position) and the exact
  inclusive/exclusive boundary in the tiling test.

## Evaluation settings

Judged on hidden tests covering: a single character; aperiodic strings like `abc`; the unit-period
string `aaaa...`; clean repetitions like `(abc)^k` and `(ab)^k`; near-misses where one trailing
character breaks divisibility (`aaab`, `abcabca`); strings whose borders exist but whose period does
not divide the length (`abcab`, period 3, length 5); and a full `n = 10^6` worst case
(`a^10^6` and `(ab)^500000`) where the count can reach `~10^6` and the tile-length sum exceeds the
32-bit range, so the second accumulator must be 64-bit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int n = (int)s.size();

    // TODO: build the KMP failure function indexed by prefix length, then count
    // tiled prefixes and sum their minimal tile lengths.
    long long count = 0, sumTile = 0;

    cout << count << " " << sumTile << "\n";
    return 0;
}
```
