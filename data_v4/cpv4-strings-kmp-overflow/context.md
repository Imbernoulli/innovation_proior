# Self-similarity score of a string (total prefix occurrences)

## Research question

You are given a single string `s` of length `n` over lowercase Latin letters. For every prefix length
`k` from `1` to `n`, let `occ(k)` be the number of positions where the prefix `s[0..k-1]` occurs as a
(contiguous) substring of `s`. Two occurrences are different if they start at different positions; an
occurrence may overlap another. Define the **self-similarity score** of `s` as

```
score(s) = occ(1) + occ(2) + ... + occ(n)
```

i.e. the total number of (prefix, occurrence) incidences across all prefixes. Output that single number.

This is the "count the occurrences of every prefix" problem in disguise, summed into one scalar. It is
the kind of self-overlap measurement that shows up in periodicity analysis and string compression
heuristics, so getting the border-chain bookkeeping exactly right — and not losing the answer to a
silent integer overflow — is the whole game.

## Input / output contract

- Input (stdin): one line containing the string `s` (`1 <= n <= 2*10^5`), consisting only of lowercase
  letters `a`-`z`. (An empty input is treated as `n = 0` with score `0`.)
- Output (stdout): a single line with `score(s)`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = "ababa"` the answer is `9`. The prefix `"a"` occurs at positions `0, 2, 4` (3 times);
`"ab"` at `0, 2` (2 times); `"aba"` at `0, 2` (2 times); `"abab"` at `0` (1 time); `"ababa"` at `0`
(1 time). Total `3 + 2 + 2 + 1 + 1 = 9`.

## Background

Counting, for one fixed pattern, how often it occurs in a text is the classic Knuth-Morris-Pratt
problem. Here the "patterns" are *all `n` prefixes of `s` itself*, which is what makes a single linear
pass possible: the prefix function already encodes, for every position, the longest prefix that ends
there, and the shorter prefixes that end there are exactly the entries of its border chain.

Two families of approach are on the table before committing to one:

- **Per-prefix search.** For each of the `n` prefixes, run a substring count over `s`. Conceptually
  simple and obviously correct, but it is `O(n^2)` (or worse) and cannot survive `n = 2*10^5`.
- **One prefix-function pass.** Compute the KMP prefix function `pi` once, then aggregate occurrence
  counts down the border chains in `O(n)`. The open questions are the exact aggregation order and the
  off-by-one in how `pi` values map to prefix lengths.

## Evaluation settings

Judged on hidden tests covering: short strings checked against a brute force; highly periodic strings
(`"aaaa...a"`, `"abababab..."`) where prefixes recur the most; strings with a unique alphabet where
every prefix occurs exactly once; and large `n = 2*10^5` strings (notably `"aaaa...a"`, whose score is
`n(n+1)/2 = 2.0000100*10^10`) so that a 32-bit accumulator silently overflows.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    if (!(cin >> s)) { cout << 0 << "\n"; return 0; } // empty input -> score 0
    int n = (int)s.size();

    // KMP prefix function: pi[i] = length of the longest proper border of s[0..i].
    vector<int> pi(n, 0);
    for (int i = 1; i < n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[i] != s[j]) j = pi[j - 1];
        if (s[i] == s[j]) j++;
        pi[i] = j;
    }

    // TODO: aggregate occurrence counts of every prefix via the border chains and
    //       sum them into the self-similarity score (mind the integer width).
    long long score = 0;

    cout << score << "\n";
    return 0;
}
```
