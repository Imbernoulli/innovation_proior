# Lexicographically smallest string after deleting k characters

## Research question

You are given a string `s` of lowercase letters and digits (more precisely: printable
non-whitespace characters, but every test uses characters from `0`–`9` and `a`–`z`) together with an
integer `k`. You must **delete exactly `k` characters** from `s` — when `k < |s|` — keeping the
remaining characters **in their original relative order**, so that the resulting string of length
`|s| - k` is **lexicographically smallest** among all such strings. If `k >= |s|` you have deleted
everything and the result is the empty string. Output that smallest string.

"Lexicographically smallest" is the dictionary order on strings of the **same length** `|s| - k`
(all candidates have exactly that length, so the comparison is unambiguous): compare character by
character, and the first position where two candidates differ decides — the one with the smaller
character there is smaller. No characters are stripped or normalized; in particular a leading `0` is
a perfectly good (and small) first character and is kept, not removed.

This is the one-dimensional core of a whole family of "make it smallest / make it largest by
removing things" problems (smallest subsequence, "remove k digits", competition shrink-to-fit
tasks). Getting the linear-time version exactly right — including the ties, the run-out-of-budget
corner, and the delete-everything corner — is the point.

## Input / output contract

- Input (stdin): two whitespace-separated tokens.
  - The first token is the string `s` with `1 <= |s| <= 10^6`, consisting of non-whitespace
    characters (tests use `a`–`z` and `0`–`9`).
  - The second token is the integer `k` with `0 <= k <= 10^6`. `k` may be `>= |s|`.
- Output (stdout): a single line containing the lexicographically smallest string obtainable by
  deleting exactly `min(k, |s|)` characters while preserving order. When `min(k, |s|) = |s|` the line
  is empty (just a newline).
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = "1432219"` and `k = 3`, the answer is `"1219"`. Deleting the `4`, the first `3`,
and one `2` leaves the length-4 subsequence `1219`, and no other length-4 subsequence of `1432219`
is smaller.

## Background

The result must be a length-`(|s|-k)` subsequence of `s`, chosen to be lexicographically smallest.
Two ways to think about it are on the table before committing to one:

- **Greedy by deleted character.** Phrase the task as "perform `k` deletions"; at each step delete
  the character that seems most harmful — e.g. the largest character, or the leftmost character that
  is larger than its right neighbour. The hope is that a simple local rule, repeated `k` times,
  lands on the global optimum. The open question is whether any such per-deletion rule, done
  independently, is actually optimal — and whether it can be made fast enough for `|s| = 10^6`.
- **Position-by-position construction with a stack.** Build the answer left to right, maintaining
  the characters chosen so far on a stack, and decide at each incoming character whether earlier
  chosen characters should be undone (deleted) because the new, smaller character can take an earlier
  slot. The open questions are the exact pop condition, how the deletion budget gates it, and what to
  do with any leftover budget at the end.

## Evaluation settings

Judged on hidden tests covering: `k = 0` (no deletion, output `= s`); `k >= |s|` (delete everything,
empty output); strictly increasing strings (deletions must come from the tail); strictly decreasing
strings (deletions must come from the front); strings with long equal runs and ties; digit strings
with interior and leading zeros; single-character strings; and large `|s| = 10^6` with mixed
characters and a worst-case "many forced pops" layout, to confirm the solution is genuinely linear
and fits the 1-second limit.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    long long k;
    if (!(cin >> s >> k)) return 0;

    long long n = (long long)s.size();

    // TODO: delete exactly min(k, n) characters, preserving order, so the
    // remaining length-(n - min(k,n)) string is lexicographically smallest.
    string answer;

    cout << answer << "\n";
    return 0;
}
```
