# Minimum palindromic factorization

## Research question

You are given a string `s` of lowercase English letters. A **palindromic factorization** of
`s` is a way to cut `s` into consecutive non-empty blocks `s = p_1 p_2 ... p_k` such that every
block `p_t` reads the same forwards and backwards. Among all such factorizations, output the
**minimum possible number of blocks** `k`.

Every string has at least one palindromic factorization (each single character is a palindrome),
so the answer is always between `1` and `|s|` for a non-empty string, and `0` for the empty
string. The interesting cases are strings with rich internal palindrome structure, where a few
long palindromes can replace many short ones.

This "fewest palindromic pieces" question is the core subproblem behind palindrome-aware text
segmentation and several string-compression schemes, so the one-dimensional version has to be
exactly right — including the strings whose palindromic suffixes form long arithmetic chains,
which are exactly the inputs that make a naive method slow.

## Input / output contract

- Input (stdin): a single token `s`, the string, consisting of lowercase English letters
  `a`–`z`. The string may be empty (in which case stdin contains no non-whitespace token).
- Output (stdout): a single line with one integer — the minimum number of palindromic blocks
  in a factorization of `s` (`0` for the empty string).
- Constraints: `0 <= |s| <= 10^6`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `s = bobseesanna` the answer is `3` (`bob | sees | anna`). For `s = abacaba`
the answer is `1` (the whole string is already a palindrome).

## Background

Let `dp[i]` be the minimum number of palindromic blocks needed to factor the prefix of length
`i`, with `dp[0] = 0`. The only way to finish the prefix of length `i` is to let the last block
be a palindromic **suffix** of that prefix: if `s[i-L .. i-1]` is a palindrome then
`dp[i] = min(dp[i], dp[i-L] + 1)`. So the whole problem reduces to, for every prefix, iterating
over its palindromic suffixes and relaxing `dp`.

Two families of approach are on the table before committing to one:

- **Per-position palindrome scan.** For each cut point, test candidate suffixes for being
  palindromes (via a precomputed Manacher / interval-palindrome table) and relax `dp`. This is
  correct and easy to reason about, but a single prefix can have `Theta(n)` palindromic suffixes
  (think `aaaa...a`), so the total work is `Theta(n^2)` — fine at `n = 2000`, hopeless at
  `n = 10^6`.
- **Eertree (palindromic tree) with series links.** Maintain, incrementally as characters are
  appended, the set of all distinct palindromic suffixes of the current prefix. The open question
  is how to relax `dp` over *all* of a prefix's palindromic suffixes without paying for each one
  individually, since there can be linearly many of them.

## Evaluation settings

Judged on hidden tests covering: the empty string and single characters; all-equal strings
(`aaaa...`), which maximize the number of palindromic suffixes per prefix; alternating and
small-alphabet strings; whole-string palindromes (answer `1`); random large-alphabet strings
(answer near `|s|`, mostly singletons); and adversarial **Fibonacci-word** prefixes of length up
to `10^6`, which are the classic worst case for the palindromic-suffix structure (long chains of
palindromic suffixes whose lengths form arithmetic progressions).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    cin >> s;                  // empty string -> s stays ""
    int n = (int)s.size();

    // TODO: compute the minimum number of palindromic blocks in a
    // factorization of s (0 for the empty string).
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
