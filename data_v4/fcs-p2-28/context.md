# Longest palindromic subsequence length

## Research question

You are given a single lowercase string `s`. A **subsequence** is obtained by deleting zero or more
characters without changing the order of the remaining ones (the deleted positions need not be
contiguous). A string is a **palindrome** if it reads the same forwards and backwards. Among all
subsequences of `s` that are palindromes, output the **length of the longest** one.

The empty string is a palindrome, so the answer is always at least `0`; any non-empty `s` has the
answer at least `1` (every single character is a one-letter palindrome).

This is the subsequence version of "longest palindrome", and the word *subsequence* (not *substring*)
is the crux: the characters of the palindrome may be scattered through `s`. For example, in
`character` the longest palindromic subsequence is `carac` of length `5`, even though no length-5
palindromic *substring* exists.

## Input / output contract

- Input (stdin): a single token `s`, a non-empty-or-empty string of lowercase English letters
  (`a`-`z`). Its length satisfies `0 <= |s| <= 2000`. (An empty input — no token — denotes the empty
  string.)
- Output (stdout): a single line containing one integer, the length of the longest palindromic
  subsequence of `s`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = bbbab` the answer is `4` (the subsequence `bbbb`).

## Background

The constraint that the palindrome is a *subsequence* rather than a *substring* changes the problem
completely, and two families of approach are on the table before committing to one:

- **Expand around centers.** The classic linear-ish technique for longest palindromic *substring*:
  for each of the `2n-1` possible centers, expand outward while the two sides match, and keep the
  longest run. It is short and fast (`O(n^2)`), and the open question is whether it can be made to
  find the longest *subsequence* rather than only contiguous substrings.
- **Interval dynamic programming.** Define a value for each interval `s[i..j]` — the length of its
  best palindromic subsequence — and combine intervals by their endpoints. This is `O(n^2)` time;
  the open question is the exact recurrence at the endpoints and how to keep memory within bounds.

## Evaluation settings

Judged on hidden tests covering: the empty string (`|s| = 0`), a single character, two characters
(equal and unequal), strings that are already palindromes (even and odd length), strings with no
repeated characters (answer `1`), strings over a tiny alphabet where palindromic subsequences are
dense, a single character repeated up to the limit (answer equals the length), and worst-case
`|s| = 2000` random strings.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) {           // empty input -> empty string
        cout << 0 << "\n";
        return 0;
    }
    int n = (int)s.size();

    // TODO: compute the length of the longest palindromic subsequence of s.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
