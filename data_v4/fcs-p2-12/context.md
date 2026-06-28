# Palindrome partitioning: minimum cuts

## Research question

You are given a single non-empty string `s` of lowercase English letters. A *palindrome partition*
of `s` is a way of splitting `s` into consecutive, non-overlapping substrings (covering all of `s`)
such that **every** substring is a palindrome. A *cut* is a position between two adjacent substrings
of the partition, so a partition into `k` pieces uses exactly `k - 1` cuts.

Output the **minimum possible number of cuts** over all palindrome partitions of `s`.

Every single character is itself a palindrome, so a partition always exists (in the worst case, cut
`s` into its individual characters). If `s` is already a palindrome, the answer is `0`.

This is the classic "palindrome partitioning II" shape. It looks like it should yield to a quick
greedy rule, which makes it a good test of whether a tempting local heuristic survives contact with a
counterexample, or whether a plain dynamic program is the thing to ship.

## Input / output contract

- Input (stdin): one line containing the string `s` (`1 <= |s| <= 2000`), lowercase letters `a`–`z`
  only, no spaces.
- Output (stdout): a single line with one integer — the minimum number of cuts.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
input:
aab

output:
1
```

Explanation: `aab` can be partitioned as `aa | b` (two palindromes), using one cut. No partition uses
zero cuts because `aab` is not itself a palindrome.

## Background

The decision at each position — where to place the next cut — interacts with every later decision,
so two families of approach are on the table before committing to one:

- **Greedy by longest palindrome prefix.** Walk left to right; at the current position, take the
  *longest* palindromic substring starting there as the next piece, place a cut after it, and repeat.
  It is near-linear and a handful of lines. The open question is whether grabbing the longest
  palindrome locally can ever force more cuts globally than necessary.
- **Dynamic programming over prefixes.** Precompute, for every pair `(i, j)`, whether `s[i..j]` is a
  palindrome, then let `cut[j]` be the minimum cuts needed to partition the prefix `s[0..j]` and fill
  it left to right. This is `O(n^2)` time and `O(n^2)` (or `O(n)` with care) memory. The open
  questions are the exact recurrence and how to build the palindrome table so each lookup is `O(1)`.

With `|s| <= 2000`, an `O(n^2)` method does about `4 * 10^6` table entries and the same order of DP
work — comfortably inside two seconds — so quadratic is acceptable if it is provably correct.

## Evaluation settings

Judged on hidden tests covering: strings that are already palindromes (answer `0`), single
characters (`|s| = 1`), strings with no repeated structure so the answer is `|s| - 1`, strings built
by concatenating palindromes, near-palindromes with a few perturbations, small alphabets (so
palindromic substrings are dense and overlap heavily), and the maximum length `|s| = 2000`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    string s;
    if (!(cin >> s)) {           // no token on input
        cout << 0 << "\n";
        return 0;
    }
    int n = (int)s.size();

    // TODO: compute the minimum number of cuts so that every piece of s is a palindrome.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
