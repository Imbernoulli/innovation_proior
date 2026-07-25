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

## Evaluation settings

Judged on hidden tests covering: identical strings, disjoint alphabets, strings that are reorderings
of each other, strings that share a long hidden subsequence buried in noise, length-1 strings, and
worst-case sizes `|s| = |t| = 3000` over both large and tiny alphabets.

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
