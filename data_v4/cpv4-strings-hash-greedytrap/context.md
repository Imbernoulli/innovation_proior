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
square). The pieces overlap each other in complicated ways, which is exactly what makes the "obvious"
left-to-right greedy tempting — and, as it turns out, wrong.

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

## Background

The set of squares inside `s` can be large and the squares interlock, so two families of approach
are on the table before committing to one:

- **Left-to-right greedy.** Scan from the left; at the first position where some square begins, grab
  one of them (the shortest, or the longest) and jump past it, then continue. This is near-linear and
  a few lines to write; the open question is whether a local "take a square now" decision can ever be
  forced to give up more coverage than it gains, because squares starting at one position can overlap
  longer squares starting one position later.
- **Prefix dynamic programming.** Define `dp[i]` = the best coverage achievable using
  non-overlapping squares lying entirely inside the first `i` characters `s[0..i)`. To extend to `i`
  we either leave position `i-1` uncovered (`dp[i-1]`) or close a square that ends exactly at `i`.
  Testing "does a square of length `2L` end at `i`" reduces to comparing two equal-length halves,
  which a polynomial rolling hash answers in `O(1)`. The open questions are the exact recurrence and
  getting the substring-equality test right.

## Evaluation settings

Judged on hidden tests covering: strings with no square at all (answer `0`), single characters,
the empty string (`n = 0`), all-equal strings like `aaaa...` (every even window is a square, so the
whole string is coverable), highly self-similar strings over a 2-letter alphabet where many squares
interlock and the greedy trap bites, and full-size `n = 5000` cases (so an `O(n^2)` method with an
`O(1)` square test is the intended complexity, and a naive `O(n^3)` char-by-char comparison would be
too slow).

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
