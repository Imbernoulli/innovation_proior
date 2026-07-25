# Lexicographically least rotation of a string

## Research question

You are given a single string `s` of length `n`. Consider all `n` cyclic
rotations of `s`: rotation `k` (for `0 <= k < n`) is the string
`s[k] s[k+1] ... s[n-1] s[0] s[1] ... s[k-1]`, i.e. `s[k:] + s[:k]`. Among these
`n` rotations, exactly one (up to ties) is **lexicographically smallest**. Output
the starting index `k` of that smallest rotation, and when several distinct
indices produce the same smallest rotation, output the **smallest** such index.

This "minimal rotation" / "least circular shift" is the canonical normal form for
a cyclic string: two strings are rotations of one another iff their minimal
rotations are equal. It is the workhorse behind canonicalizing necklaces,
de-duplicating cyclic sequences (circular DNA, polygon vertex loops, bracelet
enumeration), and is the first step of several string-periodicity algorithms.
Getting the **index** right — not just the rotated string, and with the correct
tie rule — is what makes it reusable as a building block.

## Input / output contract

- Input (stdin): a single token `s`, the string. Its length satisfies
  `1 <= n <= 10^6`. The string consists of printable non-whitespace characters
  (comparison is by raw byte / `char` value, i.e. ordinary lexicographic order on
  the underlying code units). If the input is empty (no token at all), treat the
  string as empty.
- Output (stdout): a single line containing one integer — the smallest 0-based
  index `k` such that `s[k:] + s[:k]` is the lexicographically least rotation. For
  the empty string, output `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = "abracadabra"` (length 11) the answer is `10`. The rotation
starting at index 10 is `"aabracadabr"`, which is lexicographically smaller than
every other rotation (no other rotation begins with `"aa"`).

## Evaluation settings

Judged on hidden tests covering: single-character strings; all-equal strings
(`"aaaa..."`, every rotation identical); period-2 and other highly periodic
strings (`"abab..."`, repeated blocks); strings with long internal runs
(`"aaaa...ab"`, `"b aaaa..."`); near-Lyndon and near-palindromic inputs; random
strings over small and large alphabets; and large `n = 10^6` stress inputs
(all-equal, period-2, single-mismatch character).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {
        // Empty input -> empty string; least-rotation index is 0.
        cout << 0 << "\n";
        return 0;
    }

    // TODO: compute the smallest index k such that s[k:] + s[:k] is the
    // lexicographically least rotation of s (smallest index on ties).
    int k = 0;

    cout << k << "\n";
    return 0;
}
```
