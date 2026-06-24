# Prefix occurrence counts of a string

## Research question

You are given a single string `S` of length `n` over lowercase Latin letters. For every length
`L` from `1` to `n`, consider the length-`L` **prefix** of `S`, that is `S[0..L-1]`. Let `c[L]` be the
number of times this prefix occurs as a (contiguous) substring of `S` itself, where occurrences are
allowed to **overlap**. Output `c[1], c[2], ..., c[n]`.

Concretely, `c[L]` counts the start positions `i` (with `0 <= i <= n-L`) such that
`S[i..i+L-1] == S[0..L-1]`. The whole prefix always matches itself at position `0`, so `c[L] >= 1`
for every `L`; and `c[n] = 1` always, because the only place the full string `S` fits is position `0`.

This is the classic "count occurrences of every prefix" task. It is the workhorse behind period
detection, run-length compression of repetitive strings, and the inner loop of several string-matching
and suffix-structure constructions, so computing all `n` counts in linear time — without double-counting
through the border chain and without an off-by-one in the prefix length — is the thing that matters.

## Input / output contract

- Input (stdin): a single line containing the string `S` (`1 <= n = |S| <= 2*10^5`), made of
  lowercase Latin letters `a`–`z`. There is no separate length token; the string length is `|S|`.
- Output (stdout): one line with `n` integers separated by single spaces: `c[1] c[2] ... c[n]`,
  terminated by a newline.
- Time limit: 1 second. Memory: 256 MB.

Example: for `S = "abacaba"` the answer is `4 2 2 1 1 1 1`.
- `c[1]`: prefix `"a"` occurs at positions `0,2,4,6` -> `4`.
- `c[2]`: prefix `"ab"` occurs at positions `0,4` -> `2`.
- `c[3]`: prefix `"aba"` occurs at positions `0,4` -> `2`.
- `c[4]`: prefix `"abac"` occurs only at position `0` -> `1`.
- `c[5..7]`: each longer prefix occurs only at position `0` -> `1` each.

## Background

The brute force — for each of the `n` prefixes, scan all start positions and compare — is
`O(n^2)` character comparisons in the worst case (e.g. `S = "aaaa...a"`), which is far too slow at
`n = 2*10^5`. Two structural facts cut it to linear:

- **Failure function (KMP).** Define `pi[i]` as the length of the longest proper prefix of `S[0..i]`
  that is also a suffix of `S[0..i]`. It is computable in `O(n)`. The key observation: a prefix of
  length `L` ends at position `i` exactly when `L` is one of the *border lengths* obtained by
  following the chain `pi[i], pi[pi[i]-1], ...` down to `0`. So the multiset of prefix-occurrences is
  fully described by the failure function.
- **Counting by propagation.** Naively walking the border chain from every end position is again
  `O(n^2)`. Instead one seeds, for each end position `i`, a single count at length `pi[i]`, and then
  pushes counts *down the border chain* once, in decreasing order of length. Each length is touched a
  constant number of times, giving `O(n)`.

Both ideas are standard; the danger is entirely in the bookkeeping — which length gets `+1`, in what
order the propagation runs, and whether the "prefix occurs as itself" term is added exactly once.

## Evaluation settings

Judged on hidden tests covering: a single character (`n = 1`); strings with no repeats
(`"abcde"` -> all `1`s); maximal-overlap strings (`"aaaa...a"`, where `c[L] = n - L + 1`); classic
KMP stress strings (`"abacaba"`, `"aabaaab"`); random strings over a 2- and 3-letter alphabet so the
border chains are long; and a worst case `n = 2*10^5` (so an `O(n^2)` brute force or an unbounded
border walk both time out).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string s;
    if (!(cin >> s)) return 0;
    int n = (int)s.size();

    // TODO: compute c[L] = number of (overlapping) occurrences of the length-L
    // prefix of s inside s, for L = 1..n, and print them space-separated.

    return 0;
}
```
