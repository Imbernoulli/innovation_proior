# Minimum character insertions to make a string a palindrome

## Research question

You are given a single string `s` consisting of lowercase English letters. In one operation you may
insert **one** character (any lowercase letter) at **any** position of the string — the front, the
back, or between any two existing characters. You want the string to become a **palindrome** (it
reads the same forwards and backwards). Output the **minimum number of insertions** needed.

You only ever insert; you never delete or replace. A string of length `0` or `1` is already a
palindrome and needs `0` insertions.

This is a classic string-DP subproblem: it shows up inside text reconstruction, sequence alignment,
and "make-it-symmetric" style tasks, so getting the optimal count exactly right — including the
short-string and already-a-palindrome corners — matters.

## Input / output contract

- Input (stdin): a single token `s`, the string (`1 <= |s| <= 2000`), lowercase letters `a`–`z`.
  (If the input is empty, treat the string as empty and output `0`.)
- Output (stdout): a single line with the minimum number of insertions.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `s = "race"` the answer is `3` (for instance insert to form `"ecarace"`); for
`s = "abcba"` the answer is `0` (already a palindrome).

## Background

The "make it a palindrome with the fewest edits, only insertions allowed" phrasing invites a quick
local heuristic. Two families of approach are on the table before committing to one:

- **Greedy two-pointer matching.** Put a pointer at each end. If the two characters match, step both
  inward. If they differ, "fix" one side — e.g. insert a copy of one end's character at the other
  end (count one insertion) and move the corresponding pointer. Repeat until the pointers cross. It
  is `O(n)` and a handful of lines; the open question is whether the local "which side do I fix?"
  decision is ever forced to guess wrong.
- **Interval dynamic programming.** Define the minimum insertions to palindromize each substring and
  build up from short substrings to the whole string. With `|s| <= 2000` this is `O(n^2)` time and
  fits comfortably; the open question is the exact recurrence and the base cases.

## Evaluation settings

Judged on hidden tests covering: already-palindromes (answer `0`), single characters, length-2
strings, two-letter-alphabet strings (which stress the matching decisions), full-alphabet random
strings, repeated-character strings, near-palindromes (one character off), and the maximum
`|s| = 2000`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {            // empty input -> empty string is already a palindrome
        cout << 0 << "\n";
        return 0;
    }
    int n = (int)s.size();

    // TODO: compute the minimum number of single-character insertions that make s a palindrome.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
