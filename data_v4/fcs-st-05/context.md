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

## Background

The constraint is purely lexicographic but defined over a *cyclic* object, so the
naive plan is to materialize candidates and compare. Two families of approach are
on the table before committing to one:

- **Compare all rotations.** Generate each of the `n` rotations and keep the
  lexicographically smallest, breaking ties by index. Each rotation is length `n`
  and a comparison can scan all `n` characters, so this is `O(n^2)` time (and
  `O(n)` or more for the candidate strings). It is trivially, obviously correct;
  the open question is whether the quadratic cost is affordable at the stated `n`.
- **Linear single-pass scan.** Walk a failure-function / KMP-style automaton over
  the conceptual doubled string `s + s`, maintaining a single candidate start
  index and, on a mismatch, sliding that candidate forward past an entire matched
  block at once instead of one position at a time. This is `O(n)` time and `O(n)`
  space; the open questions are the exact update rule for the candidate start and
  for the failure array, and the tie-breaking behaviour.

## Evaluation settings

Judged on hidden tests covering: single-character strings; all-equal strings
(`"aaaa..."`, every rotation identical — the answer must be the tie-break index
`0`); period-2 and other highly periodic strings (`"abab..."`, repeated blocks);
strings with long internal runs that force large block-skips (`"aaaa...ab"`,
`"b aaaa..."`); near-Lyndon and near-palindromic inputs; random strings over
small and large alphabets; and large `n = 10^6` adversarial inputs (all-equal,
period-2, single-mismatch) where an `O(n^2)` method times out but an `O(n)` one
finishes in well under the limit.

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
