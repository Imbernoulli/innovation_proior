# Maximum square-tiled coverage of a string

## Research question

A "square" is a string of the form `uu`: some non-empty string `u` written twice in a row (so `abab`
is a square with `u = ab`, and `aa` is a square with `u = a`). Squares are the simplest possible
form of local repetition, and detecting how much of a stream is built out of them is a basic
redundancy measurement.

You are given a string `s` of length `n`. You may select a set of **non-overlapping** substrings of
`s`, where **every selected substring must itself be a square** (even length `>= 2`, first half equal
to second half). The substrings may not share any position. Your goal is to **maximize the total
number of characters covered** by the selected squares. Output that maximum coverage.

Concretely: choose disjoint intervals `[l_1, r_1), [l_2, r_2), ...` of `s`, each of which is a
square, so that the sum of their lengths is as large as possible. A position covered by no chosen
square contributes nothing; the empty selection (cover `0`) is always allowed, so the answer is at
least `0`.

This is a coverage/packing problem whose feasible pieces are defined by a string property (being a
square). The pieces overlap each other in complicated ways, so a square chosen at one position can
preclude other, possibly better, choices nearby.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 5000`). If `n > 0`, the next token is the string
  `s`, consisting of `n` lowercase-or-arbitrary printable non-whitespace characters (the reference
  tests use lowercase letters). If `n = 0` there is no string token.
- Output (stdout): a single line with the maximum total number of characters that can be covered by
  non-overlapping squares.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `s = "aabab"` the answer is `4`. The squares available are `aa` (positions `[0,2)`) and
`abab` (positions `[1,4)`); they overlap at position `1`, so at most one can be chosen, and choosing
`abab` covers `4` characters, which beats `aa`'s `2`.

## Evaluation settings

Judged on hidden tests covering: strings with no square at all (answer `0`), single characters,
the empty string (`n = 0`), all-equal strings like `aaaa...` (every even window is a square, so the
whole string is coverable), highly self-similar strings over a 2-letter alphabet where many squares
interlock, and full-size `n = 5000` cases.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;             // n = 0 (or empty input) -> answer 0
    string s;
    if (n > 0) cin >> s;

    // TODO: maximize the total length covered by non-overlapping squares (substrings of form uu).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
