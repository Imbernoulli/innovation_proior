# Longest bitonic (increase-then-decrease) subsequence

## Research question

You are given an array `a[0..n-1]` of integers. A **bitonic** subsequence is one whose values
**strictly increase** up to a single peak and then **strictly decrease** afterwards. Formally, a
subsequence `b[0..k-1]` (elements taken in their original left-to-right order) is bitonic if there is
a peak position `p` with

```
b[0] < b[1] < ... < b[p]   and   b[p] > b[p+1] > ... > b[k-1].
```

We require at least one real increase before the peak (`p >= 1`), so the array must actually go up
before it comes down. The strictly-decreasing tail may be empty: a purely strictly-increasing
subsequence of length `>= 2` is bitonic (its last element is the peak, with an empty descent). Both
the ascending and descending comparisons are **strict**, so equal values may not sit next to each
other in the chosen subsequence.

Output the **length** of the longest bitonic subsequence, or `0` if no valid bitonic subsequence
exists (e.g. a strictly non-increasing array, or `n <= 1`).

This is a one-dimensional "mountain" extraction that shows up inside terrain/altitude problems,
unimodal-fit problems, and as a subroutine in longer DP pipelines, so pinning the strict-comparison
and minimum-length corners exactly is what matters.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 5000`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the length of the longest bitonic subsequence (an integer
  `>= 0`).
- Time limit: 2 seconds. Memory: 256 MB.

Example 1: for `a = [1, 11, 2, 10, 4, 5, 2, 1]` the answer is `6` (one witness is
`1 < 2 < 10 > 4 > 2 > 1`).

Example 2: for `a = [5, 4, 3, 2, 1]` the answer is `0` (strictly decreasing — never increases).

## Background

The strict-up-then-strict-down shape suggests two families of approach, and the choice of which to
commit to is the crux:

- **Single-pass / greedy peak walk.** Walk the array once, climbing while the next element is larger
  and then descending while it is smaller, in one sweep. It is `O(n)` and a dozen lines; the open
  question is whether one local pass can ever recover the best *subsequence* (which may skip
  arbitrarily many elements on either slope), or whether it only ever sees one contiguous mountain.
- **Length DP from both ends.** For every index `i`, compute the longest strictly-increasing
  subsequence **ending** at `i` and the longest strictly-decreasing subsequence **starting** at `i`,
  then combine them at each candidate peak. This is `O(n^2)` at these constraints; the open questions
  are the exact combine formula (the peak is shared and must not be double-counted) and the
  minimum-length / strictness corners.

## Evaluation settings

Judged on hidden tests covering: clean mountains; strictly increasing arrays (full ascent, empty
descent); strictly decreasing arrays (answer `0`); arrays with many duplicate / equal values (where
strictness bites); `n = 0` and `n = 1` (answer `0`); plateaus of equal values (answer `0`); and large
`n = 5000` arrays with values near `+-10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: compute the length of the longest strictly-increase-then-strictly-decrease
    // subsequence (at least one increase required; empty descent allowed). Print 0 if none.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
