# Lexicographic substring comparison queries

## Research question

You are given a lowercase string `s` of length `n` and `q` queries. Each query names two
substrings of `s` by a starting position and a length, `(l1, len1)` and `(l2, len2)`, and asks
how they compare lexicographically. For each query you must output:

- `-1` if `s[l1 .. l1+len1)` is strictly lexicographically smaller,
- `0` if the two substrings are equal,
- `1` if the first is strictly larger.

The comparison is the ordinary dictionary order: compare character by character; at the first
position where they differ, the smaller character wins; if one runs out first and everything
matched so far, the shorter substring is smaller. With `q` up to `2*10^5` queries on a string of
up to `2*10^5` characters, the question is whether we can answer each comparison in (amortized)
constant time after a single pass over the string, rather than re-scanning characters per query.

This is the indexing core that shows up under "is `s[a..)` and `s[b..)` equal for the first `len`
characters", k-th-distinct-substring walks, and suffix-based pattern queries, so getting the
suffix-order primitive exactly right — including the equal-prefix and length tie-break corners —
matters.

## Input / output contract

- Input (stdin):
  - line 1: the string `s` (`1 <= n = |s| <= 2*10^5`), lowercase letters `a`–`z`;
  - line 2: an integer `q` (`1 <= q <= 2*10^5`);
  - the next `q` lines: four integers `l1 len1 l2 len2` per line, `1`-indexed starts, with
    `1 <= l1, l2 <= n`, `1 <= len1 <= n-l1+1`, `1 <= len2 <= n-l2+1` (each substring stays inside
    `s`).
- Output (stdout): `q` lines, each `-1`, `0`, or `1`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `s = banana` and the query `2 3 4 3` the two substrings are `s[2..5) = "ana"` and
`s[4..7) = "ana"`, which are equal, so the answer is `0`.

## Background

The naive comparator compares the two substrings character by character, which is `O(min(len1,
len2))` per query and `O(q * n)` in the worst case (e.g. a string of all `a`s with full-length
queries) — far too slow at `2*10^5 * 2*10^5 = 4*10^10` character comparisons.

The key observation is that comparing two substrings reduces to comparing the two **suffixes** they
start at: the lexicographic order of `s[l1..)` versus `s[l2..)` is decided entirely by their
**longest common prefix (LCP)**. If that LCP already reaches the compared length, the substrings
agree on every compared character and only the lengths break the tie. So if we can get the LCP of
*any two suffixes* in `O(1)`, every query is `O(1)`.

Two families of approach are on the table before committing to one:

- **Hashing.** Precompute prefix hashes; binary-search the LCP of two suffixes by hashing equal-length
  blocks. This is `O(log n)` per query and randomized (collision risk), and the constant factor of
  rolling-hash comparisons is non-trivial.
- **Suffix array + LCP + RMQ.** Sort all suffixes (suffix array), compute the LCP between adjacent
  sorted suffixes (Kasai), and answer the LCP of two *arbitrary* suffixes as a range-minimum over the
  LCP array between their ranks (sparse table). Deterministic, `O(n log n)` preprocessing, `O(1)`
  per query.

## Evaluation settings

Judged on hidden tests covering: random strings over small and full alphabets; highly repetitive
strings (e.g. all one letter, periodic patterns like `abcabc...`) where naive comparison is
worst-case; equal substrings at different offsets; prefix-of relationships where the LCP reaches the
compared length and the length tie-break decides; single-character strings (`n = 1`); and the full
scale `n = q = 2*10^5`.

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

    int q;
    cin >> q;

    // TODO: preprocess s so that the longest common prefix of any two suffixes
    // can be queried quickly, then answer each (l1, len1, l2, len2) comparison.

    for (int t = 0; t < q; t++) {
        int l1, len1, l2, len2;
        cin >> l1 >> len1 >> l2 >> len2;
        // TODO: output -1 / 0 / 1 for this query.
    }
    return 0;
}
```
