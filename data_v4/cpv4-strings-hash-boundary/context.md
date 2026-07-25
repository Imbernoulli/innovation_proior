# Smallest period of a queried substring

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
is built on.

## Input / output contract

- Input (stdin):
  - line 1: the string `s` (`1 <= |s| <= 5000`), lowercase letters `a`-`z`;
  - line 2: an integer `q` (`1 <= q <= 5000`);
  - next `q` lines: two integers `l r` (`1 <= l <= r <= |s|`), 1-indexed inclusive.
- Output (stdout): `q` lines; line `k` is the smallest period of the `k`-th queried substring.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = abcabcab` and the four queries `(1,8), (1,6), (2,5), (4,4)` the answers are
`3, 3, 3, 1`.

## Evaluation settings

Judged on hidden tests covering: single-character strings and `len = 1` queries; highly
periodic strings like `(ab)^k` and `a^k`; aperiodic strings; queries whose left end is not `1`;
and full-size `|s| = q = 5000` with full-length queries.

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

    // TODO: for each query (l, r), find the smallest period p (1 <= p <= len) of s[l..r].

    return 0;
}
```
