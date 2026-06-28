# Smallest wildcard-period of a string

## Research question

You are given a string `s` of length `n` over the lowercase letters `a..z` together with the
wildcard symbol `?`. A wildcard may later be replaced by **any** single lowercase letter.

Call an integer `p` with `1 <= p <= n` a **period** of `s` if there exists a replacement of every
`?` by a concrete lowercase letter so that the resulting (wildcard-free) string `t` satisfies
`t[i] == t[i+p]` for all `0 <= i < n-p`. Equivalently, `t` is built by repeating a block of length
`p`. Find and output the **smallest** period of `s`.

A period always exists, because `p = n` imposes no constraint at all (the condition ranges over an
empty set of indices), so the answer is a well-defined integer in `[1, n]`.

This is the periodicity question for strings with don't-care characters. The one-dimensional version
shows up inside pattern matching, compression and tandem-repeat detection, and getting it exactly
right — especially the way wildcards interact across a whole residue class rather than between
neighbours — is the crux.

## Input / output contract

- Input (stdin): a single line containing the string `s` (`1 <= n <= 2*10^5`), made of characters
  from `a..z` and `?`. There is no separate length field; the length is the length of the token.
- Output (stdout): a single line with one integer — the smallest period `p` of `s`.
- Time limit: 2 seconds. Memory: 256 MB.

Example 1: for `s = b?a` the answer is `3`. The only positions are `0,1,2`. For `p = 1` all three
positions share one residue class and would have to receive the same letter, but positions `0` and
`2` are the distinct concrete letters `b` and `a`, so no replacement of the middle `?` works.
`p = 2` fails for the same reason (positions `0` and `2` are still forced equal). `p = 3` works
vacuously. So the smallest period is `3`.

Example 2: for `s = abab` the answer is `2` (repeat the block `ab`).

Example 3: for `s = aabaab` the answer is `3` (repeat the block `aab`).

## Background

The condition "`p` is a period" partitions the indices into residue classes modulo `p`: class
`r` is `{r, r+p, r+2p, ...}`. Within one class every position must end up with the same letter in
`t`, so a replacement exists **iff every residue class contains at most one distinct concrete
letter** (wildcards are free and take whatever letter the class settles on).

Two routes are on the table before committing to one:

- **Pairwise compatibility scan.** Treat the wildcard `?` as matching anything, and for each
  candidate `p` check that `s[i]` and `s[i+p]` are "compatible" (equal, or at least one is `?`) for
  every `i`. This is the textbook border/period test transported to wildcards; the open question is
  whether neighbour-compatibility along a class actually implies the class is monochromatic.
- **Per-class consistency scan.** For each candidate `p`, walk every residue class and check that
  all of its concrete letters coincide. This is obviously correct; the open question is its cost,
  since done naively it is `O(n)` per `p` and `O(n^2)` over all `p`, far too slow at `n = 2*10^5`.

## Evaluation settings

Judged on hidden tests covering: all-wildcard strings (`?...?`, answer `1`), wildcard-free strings,
strings whose only valid period is `n`, single characters (`n = 1`), tiny alphabets where many
near-periods compete, and adversarial wildcard placements that make pairwise-compatibility disagree
with the true (transitive) period. Large `n = 2*10^5` instances test the asymptotics.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int n = (int)s.size();

    // TODO: compute the smallest p in [1, n] such that some replacement of the
    // '?' characters makes s periodic with period p (s[i] == s[i+p] for all i).
    int answer = n;

    cout << answer << "\n";
    return 0;
}
```
