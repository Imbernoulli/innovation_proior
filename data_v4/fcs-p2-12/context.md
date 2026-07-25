# Palindrome partitioning: minimum cuts

## Research question

You are given a single non-empty string `s` of lowercase English letters. A *palindrome partition*
of `s` is a way of splitting `s` into consecutive, non-overlapping substrings (covering all of `s`)
such that **every** substring is a palindrome. A *cut* is a position between two adjacent substrings
of the partition, so a partition into `k` pieces uses exactly `k - 1` cuts.

Output the **minimum possible number of cuts** over all palindrome partitions of `s`.

Every single character is itself a palindrome, so a partition always exists (in the worst case, cut
`s` into its individual characters). If `s` is already a palindrome, the answer is `0`.

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

## Evaluation settings

Judged on hidden tests covering: strings that are already palindromes, single characters
(`|s| = 1`), strings with little or no repeated palindromic structure, strings built by
concatenating palindromes, near-palindromes with a few perturbations, small alphabets (so
palindromic substrings are dense and overlap heavily), and the maximum length (`|s| = 2000`).

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
