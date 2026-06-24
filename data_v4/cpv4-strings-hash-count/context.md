# Counting length-k substrings that repeat

## Research question

You are given a lowercase string `s` of length `n` and an integer `k`. Consider every contiguous
substring of `s` that has length exactly `k`. Count how many **distinct** such substrings occur at
**two or more different starting positions** in `s`. Output that count.

A length-`k` substring is identified by its *content*, not by where it sits: if the same string of
`k` characters can be read starting at three different indices, it still counts as **one** distinct
repeated substring, not three. If `k` is `0`, larger than `n`, or otherwise leaves no length-`k`
window, the answer is `0`.

This is the counting core of de-duplication and plagiarism/near-duplicate detection: chop a text into
fixed-length shingles and ask how many distinct shingles are shared. Getting the *count* exactly right
— distinct strings, not occurrences, and the right number of windows — is where this kind of problem
quietly goes wrong.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k`
  (`0 <= n <= 2*10^5`, `0 <= k <= 2*10^5`). The second line has the string `s`, exactly `n`
  lowercase letters `a`..`z`. When `n = 0` the string line is empty.
- Output (stdout): a single line with the number of distinct length-`k` substrings that appear at
  at least two distinct starting positions.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 8`, `k = 3`, `s = "ababbaba"` the answer is `2`. The length-3 windows are
`aba, bab, abb, bba, bab, aba` (at indices 0..5); `aba` repeats (indices 0 and 5) and `bab` repeats
(indices 1 and 4), so two distinct substrings repeat.

## Background

The natural plan is to reduce each length-`k` window to a comparable key and then group equal keys.
Two families of approach are on the table before committing to one:

- **Sort the raw substrings.** Materialize all `n - k + 1` windows as strings, sort them, and walk
  the sorted list counting groups of size `>= 2`. Correct and easy to reason about, but comparing
  length-`k` strings makes it `O(n * k * log n)` in the worst case (e.g. `k ~ n/2`), which is far too
  slow at `n = 2*10^5`.
- **Polynomial rolling hash.** Give each window an integer fingerprint computed in `O(1)` per step
  by rolling, so all `n - k + 1` fingerprints cost `O(n)`. Sort the fingerprints and count groups.
  This is `O(n log n)`. The open questions are the exact roll (which power of the base leaves with
  the departing character), how many windows there even are, and — the part that actually decides the
  output — how a group of equal fingerprints maps to the *count* the problem asks for.

## Evaluation settings

Judged on hidden tests covering: strings with many repeats over a tiny alphabet (so most windows
collide), strings with no repeats at all, `k = 1` (single-character windows), `k = n` (exactly one
window, answer always `0`), `k = 0` and `k > n` (no windows, answer `0`), `n = 0`, and large
`n = 2*10^5` with `k` ranging from small to `n/2` (so an `O(n*k)` comparison would time out and the
fingerprint count must be exactly distinct-substrings, not occurrences).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k;
    if (!(cin >> n >> k)) return 0;
    string s;
    if (n > 0) cin >> s;            // when n == 0 there is no string token

    // TODO: count DISTINCT length-k substrings that occur at >= 2 starting positions.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
