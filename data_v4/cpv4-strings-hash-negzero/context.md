# Longest repeated block in an integer sequence

## Research question

You are given a sequence of `n` integers `a[0..n-1]` whose values may be **negative, zero, or
positive**. A *block* is a contiguous subarray, identified by its sequence of values (two blocks are
equal exactly when they have the same length and the same values in the same order). A block is
**repeated** if it occurs at two or more *distinct* starting positions; the two occurrences are
allowed to overlap. Output the length of the longest repeated block, or `0` if no block of length
`>= 1` repeats.

This is the integer-alphabet analogue of "longest repeated substring." It shows up whenever you must
detect duplicated runs in numeric data — repeated motifs in a signal, the longest period that recurs
in a log of deltas, or the longest tandem-friendly segment in a difference array.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated (newlines or spaces, any layout).
- Output (stdout): a single line with the length of the longest repeated block, or `0` if none.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `a = [4, -1, 0, 0, 4, -1, 0, 7]` the answer is `3`: the block `[4, -1, 0]` starts at
index `0` and again at index `4`. No block of length `4` repeats, so `3` is the longest.

## Evaluation settings

Judged on hidden tests covering: the empty array (`n = 0`) and `n = 1`; all-negative arrays both
with and without a repeat; all-zero arrays; tiny alphabets `{-1, 0, 1}` that force long repeats;
large-magnitude values near `+-10^9`; and large `n = 2*10^5` random and highly repetitive inputs.
Solutions must complete within the time and memory limits on every case above.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) { cout << 0 << "\n"; return 0; }
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: return the length of the longest block (contiguous subarray) that occurs at two or more
    // distinct starting positions (overlaps allowed); 0 if no block of length >= 1 repeats.
    // Watch the alphabet: a[i] may be negative or zero, and n may be 0.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
