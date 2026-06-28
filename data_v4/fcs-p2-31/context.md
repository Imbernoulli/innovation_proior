# Wildcard pattern matching with `?` and `*`

## Research question

You are given a **pattern** `p` and a **string** `s`. The pattern may contain ordinary lowercase
letters together with two wildcard metacharacters:

- `?` matches **exactly one** arbitrary character of `s`;
- `*` matches **any sequence** of characters of `s`, including the empty sequence.

Every other character in `p` is a literal that must match the corresponding character of `s`
exactly. The match is **anchored**: `p` must match the **entire** string `s`, not a substring of it.

Decide whether `p` matches `s`. Output `YES` if it does and `NO` otherwise.

This is the kernel that sits behind shell globbing, simple log filters, and `LIKE`-style predicates.
The one rule that makes it interesting is the semantics of `*`: a single `*` can stand for zero,
one, or many characters, and a pattern may contain many `*`s, so the number of ways `*`s could carve
up `s` is combinatorial.

## Input / output contract

- Input (stdin): two whitespace-separated tokens. The first token is the pattern `p`, the second is
  the string `s`. Both consist of lowercase letters `a`–`z` plus, for `p` only, the metacharacters
  `?` and `*`.
- Because an empty token cannot be represented on a whitespace-separated line, the literal token `-`
  (a single hyphen) denotes the **empty** string. So a pattern of `-` means the empty pattern, and a
  string of `-` means the empty string. The hyphen is never a real pattern/string character.
- Output (stdout): a single line, `YES` or `NO`.
- Constraints: `0 <= |p| <= 2000` and `0 <= |s| <= 2000` (lengths after decoding `-`).
- Time limit: 1 second. Memory: 256 MB.

Examples:

- `p = "a*b"`, `s = "axxxb"` -> `YES` (`*` absorbs `xxx`).
- `p = "a*b"`, `s = "axxxc"` -> `NO` (the trailing literal `b` has nothing to match).
- `p = "*a*a*a*"`, `s = "aaa"` -> `YES`.
- `p = "*a*a*a*"`, `s = "aa"` -> `NO` (three forced `a`s cannot all land in a 2-character string).
- `p = "-"`, `s = "-"` -> `YES` (empty pattern matches empty string).
- `p = "*"`, `s = "-"` -> `YES` (`*` can be the empty sequence).

## Background

Two ways to attack this are on the table before committing to one:

- **Two-pointer / recursive backtracking on `*`.** Walk `p` and `s` together; on a `*` remember a
  backtrack point, and if a later literal mismatch occurs, rewind to the last `*` and let it swallow
  one more character. It is short and "feels" like how a human matches globs by eye. The open
  question is its worst-case running time and whether the rewind bookkeeping is correct in every
  corner (trailing `*`, consecutive `*`s, `*` at the very start, empty `s`).
- **Two-dimensional dynamic programming.** Define a boolean table over prefixes of `p` and prefixes
  of `s` and fill it with a fixed recurrence. This is `O(|p| * |s|)` time. The open question is the
  exact transition for `*` and the boundary row/column for empty prefixes.

## Evaluation settings

Judged on hidden tests covering: empty pattern and/or empty string; patterns that are only `*`s;
patterns that mix literals, `?`, and `*`; the classic adversarial shapes such as `*a*a*...*` against
long runs of `a`, and `*a*...*b` against a string that contains no `b` (forces a deep rewind to a
final `NO`); and full-size `|p| = |s| = 2000`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string p, s;
    if (!(cin >> p)) return 0;   // no pattern token -> nothing to do
    if (!(cin >> s)) s = "";     // pattern present but string token missing

    if (p == "-") p = "";        // "-" decodes to the empty pattern
    if (s == "-") s = "";        // "-" decodes to the empty string

    // TODO: decide whether pattern p matches the whole string s under the
    // wildcard rules ('?' = one char, '*' = any sequence including empty),
    // then print "YES" or "NO".

    cout << "NO" << "\n";
    return 0;
}
```
