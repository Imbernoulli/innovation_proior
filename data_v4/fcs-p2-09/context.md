# Longest common subsequence length of two strings

## Research question

You are given two strings `s` and `t` of lowercase English letters. A *common subsequence* is a
string that can be obtained from both `s` and `t` by deleting zero or more characters without
reordering the rest. Compute the length of the **longest** common subsequence (LCS) of `s` and `t`.

This is the core measure behind diff tools, DNA/protein alignment, and plagiarism/edit-distance
scoring. The deletions-only, order-preserving rule is what separates LCS from "largest shared
multiset of characters," and it is exactly where the cheap-looking heuristics break.

## Input / output contract

- Input (stdin): the first line is the string `s`; the second line is the string `t`. Both consist
  only of lowercase English letters `a`–`z`.
- Constraints: `1 <= |s| <= 3000` and `1 <= |t| <= 3000`.
- Output (stdout): a single line with one integer — the length of the longest common subsequence.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = abcbdab` and `t = bdcaba` the answer is `4` (for instance the subsequence `bdab`,
or `bcba`).

## Background

Two families of approach are on the table before committing to one:

- **Greedy "match-as-you-scan".** Walk a pointer along `t`; for each character of `s`, if it equals
  the character under the `t`-pointer (or the next equal character ahead in `t`), count a match and
  advance. It is `O(n + m)` and a handful of lines. The open question is whether committing to the
  first available match is ever globally suboptimal under the order-preserving rule.
- **Two-dimensional dynamic programming.** Define `dp[i][j]` as the LCS length of the first `i`
  characters of `s` and the first `j` characters of `t`, and fill it with a short recurrence. This is
  `O(n*m)`; the open questions are the exact recurrence and whether `3000 * 3000` work fits the time
  and memory budget.

## Evaluation settings

Judged on hidden tests covering: identical strings (answer `= |s|`), disjoint alphabets (answer `0`),
strings that are reorderings of each other (where greedy is most tempting and most wrong), strings
that share a long hidden subsequence buried in noise, length-1 strings, and worst-case sizes
`|s| = |t| = 3000` over both large and tiny alphabets (the tiny alphabet maximizes the number of
crossing candidate matches).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s, t;
    cin >> s;
    cin >> t;

    int n = (int)s.size(), m = (int)t.size();

    // TODO: compute the length of the longest common subsequence of s and t.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
