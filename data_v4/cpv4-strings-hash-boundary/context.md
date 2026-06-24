# Smallest period of a queried substring (hash the inclusive window)

## Research question

You are given a lowercase string `s` and `q` queries. Each query is a pair `(l, r)` with
`1 <= l <= r <= |s|` (1-indexed, **both ends inclusive**). For the substring `t = s[l..r]` of
length `len = r - l + 1`, report the **smallest period** of `t`.

A period of `t` is an integer `p` with `1 <= p <= len` such that `t[i] == t[i+p]` for every index
`i` with `0 <= i < len - p`. Equivalently, the length-`(len - p)` prefix of `t` equals its
length-`(len - p)` suffix:

```
t[0 .. len-p-1]  ==  t[p .. len-1].
```

`p = len` is always a period (the overlap then has length `0`, vacuously equal), so a smallest
period always exists. For example the smallest period of `abcabcab` is `3`, of `aaaa` is `1`, and of
`aab` is `3` (no shorter `p` works).

This is the kind of substring-comparison query that string-search, periodicity, and compression code
is built on, and it lives or dies on getting the **inclusive/exclusive boundaries** of the hash
exactly right.

## Input / output contract

- Input (stdin):
  - line 1: the string `s` (`1 <= |s| <= 5000`), lowercase letters `a`-`z`;
  - line 2: an integer `q` (`1 <= q <= 5000`);
  - next `q` lines: two integers `l r` (`1 <= l <= r <= |s|`), 1-indexed inclusive.
- Output (stdout): `q` lines; line `k` is the smallest period of the `k`-th queried substring.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = abcabcab` and the four queries `(1,8), (1,6), (2,5), (4,4)` the answers are
`3, 3, 3, 1`.

## Background

The substrings are given as **inclusive** 1-indexed ranges, but a polynomial prefix-hash table is
naturally **exclusive** on its upper index (`H[i]` covers the first `i` characters). Bridging those
two conventions is where off-by-one errors breed: converting `(l, r)` to internal 0-indexed bounds,
extracting the hash of an inclusive window, and — the sharp part — forming the two equal-length
windows whose equality decides whether `p` is a period.

Two approaches are on the table before committing:

- **Direct character comparison.** For each query, test `p = 1, 2, ...` and for each `p` compare the
  prefix and suffix character by character, stopping at the first valid `p`. Correct and obvious, but
  a single period test is `O(len)`, so a query is `O(len^2)` and the whole thing is far too slow at
  the stated limits — useful only as a reference oracle.
- **Polynomial hashing with prefix tables.** Precompute prefix hashes so the hash of any inclusive
  window `s[a..b]` is `O(1)`, then each period test `p` becomes one `O(1)` hash equality, and a query
  scans `p = 1, 2, ...` until the first hit. The open questions are the exact extraction formula and,
  above all, the **boundaries** of the two windows being compared.

## Evaluation settings

Judged on hidden tests covering: single-character strings and `len = 1` queries (answer `1`); highly
periodic strings like `(ab)^k` and `a^k` (small periods, deep prefix/suffix overlaps); aperiodic
strings where the answer is `len`; queries whose left end is not `1` (so the index conversion
matters); and full-size `|s| = q = 5000` with full-length queries (so an `O(len^2)` per query
solution times out and a wrong boundary on the window flips an answer).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int q;
    if (!(cin >> q)) return 0;

    // TODO: build prefix hashes; for each query (l, r) find the smallest period p of s[l..r]
    //       by testing, for p = 1, 2, ..., whether the length-(len-p) prefix and suffix are equal.

    return 0;
}
```
