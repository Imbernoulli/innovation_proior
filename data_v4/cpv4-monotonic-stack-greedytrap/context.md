# Demolish K towers to leave the lowest skyline

## Research question

A city block is described by a row of `n` towers, left to right. Tower `i` has an integer height
`h[i]` written as a single decimal digit (`0`–`9`), so the whole skyline is a string `s` of `n`
digits. A demolition crew will remove **exactly `k`** towers (`0 <= k <= n`). The towers that survive
keep their original left-to-right order and are pushed together into a new string of `n - k` digits.

Among all the ways to choose which `k` towers to demolish, output the surviving skyline that is
**lexicographically smallest** (compared as a string of equal length `n - k`; if `k = n` the result
is the empty string). Lexicographic order is the natural one on digit characters: a smaller digit in
an earlier surviving position dominates everything to its right.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k` (`1 <= n <= 2*10^5`, `0 <= k <= n`).
  The second line is the skyline string `s` of exactly `n` characters, each a digit `'0'`–`'9'`.
- Output (stdout): a single line with the lexicographically smallest surviving string of length
  `n - k`. When `k = n` this line is empty (just a newline). No leading zeros are stripped — every
  surviving digit, including leading zeros, is printed.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 7`, `k = 3`, `s = "1432219"` the answer is `"1219"`.

## Background

The constraint "keep the surviving digits in their original order" makes this a subsequence-selection
problem under lexicographic comparison.

## Evaluation settings

Judged on hidden tests covering: already non-decreasing skylines, strictly decreasing skylines,
heavy ties (many equal digits, e.g. `"00000"`), `k = 0` (output `s` unchanged), `k = n` (output the
empty line), small alphabets that stress tie handling, and large `n = 2*10^5` to rule out
inefficient solutions.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    string s;
    cin >> s;

    // TODO: remove exactly k characters so the remaining length-(n-k) string,
    // kept in original order, is lexicographically smallest. Print it.
    string answer;

    cout << answer << "\n";
    return 0;
}
```
