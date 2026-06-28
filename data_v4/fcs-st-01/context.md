# Counting distinct substrings of a string

## Research question

You are given a single string `s` of lowercase English letters. Count how many
**distinct non-empty substrings** `s` has. A substring is any contiguous block
`s[i..j]`; two substrings are "the same" iff they are equal as strings, so the
count is the size of the set `{ s[i..j] : 0 <= i <= j < |s| }`.

The naive identity for an all-distinct string is `|s|*(|s|+1)/2`, but real
strings repeat, so the true count is smaller and must be computed. The catch is
scale: with `|s|` up to `10^6` the number of *positions* alone is `~5*10^11`,
so any method that materializes substrings is hopeless. The interesting question
is whether the distinct count can be read off a linear-size structure in linear
time.

## Input / output contract

- Input (stdin): one line containing the string `s` (lowercase `a`-`z`),
  `0 <= |s| <= 10^6`. The empty string is allowed (a blank/absent line).
- Output (stdout): a single line with one integer — the number of distinct
  non-empty substrings of `s`. For the empty string the answer is `0`.
- The answer can reach `~5*10^11`, which overflows 32-bit; it fits in a signed
  64-bit integer.
- Time limit: 2 seconds. Memory: 512 MB.

Example: for `s = "banana"` the answer is `15`. The distinct substrings are
`a, b, n, an, ba, na, ana, ban, nan, anan, bana, nana, anana, banan, banana`.

## Background

Counting substrings is a deduplication problem over an enormous implicit set.
Two families of approach are on the table before committing to one:

- **Enumerate-and-hash.** Insert every substring (or a rolling hash of it) into
  a hash set and report the set size. Conceptually it is the definition itself,
  so it is obviously correct, but it touches `Theta(|s|^2)` substrings and
  `Theta(|s|^3)` characters in the worst case; even the hashed variant is
  `O(|s|^2 log)` and `O(|s|^2)` memory. That is fine for `|s| <= 2000` and dies
  far below `10^6`.

- **A linear-size automaton over all substrings.** Build a structure whose
  states partition the set of all distinct substrings of `s`, with only `O(|s|)`
  states and edges, then count by summing a per-state contribution. The open
  questions are *which* structure has this partition property and *what* the
  per-state contribution is.

## Evaluation settings

Judged on hidden tests covering: the empty string and length-1 string; strings
with one repeated character (answer `= |s|`); strings of all-distinct
characters (answer `= |s|(|s|+1)/2`); classic repetitive strings (`banana`,
`mississippi`, `abab...`); random strings over small alphabets (heavy structural
overlap, which exercises the automaton's hardest update path); and full-scale
`|s| = 10^6` inputs over alphabets of size 1, 2, and 26, where the answer
exceeds the 32-bit range.

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
        cout << 0 << "\n";   // empty input -> empty string -> 0 substrings
        return 0;
    }

    // TODO: count the number of distinct non-empty substrings of s.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
