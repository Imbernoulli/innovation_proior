# Quality gate on a packaging line (fixed-length batches in a weight band)

## Research question

A packaging line carries `n` parcels in a single row, left to right. Parcel `i` has an integer weight
`a[i]` (weights can be negative — a parcel may be a returned item logged as a credit). The line's
quality gate inspects **batches**: a batch is any block of **exactly `w` consecutive parcels**. A batch
*passes* the gate when its total weight lies inside the **closed** band `[L, R]` (both endpoints
allowed). You must report how many distinct batches pass.

Concretely: how many start positions `s` are there such that the block of parcels
`s, s+1, ..., s+w-1` exists on the belt and has a weight sum `T` with `L <= T <= R`?

This is a fixed-window prefix-sum count. The whole problem lives or dies on two boundary decisions —
how many window start positions actually exist, and which two prefix entries you subtract to get a
window's sum — and on the band being **inclusive** on both ends. Each is a classic off-by-one, and a
single mis-set boundary changes the count on inputs as small as the worked sample.

## Input / output contract

- Input (stdin): one line with four integers `n w L R`
  (`1 <= n <= 2*10^5`, `1 <= w <= 10^9`, `-10^15 <= L <= R <= 10^15`).
  Then `n` integers `a[0], ..., a[n-1]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the number of passing batches.
- Note: if `w > n` no batch fits, so the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 6`, `w = 3`, `L = 10`, `R = 15`, weights `a = [4, 2, 5, 1, 9, 3]`, the answer is `3`.
The length-3 batches are `[4,2,5]=11` (pass), `[2,5,1]=8` (fail), `[5,1,9]=15` (pass, hits the upper
edge), `[1,9,3]=13` (pass). Three of the four pass.

## Background

The brute-force reading is direct: enumerate every length-`w` block, sum its `w` weights, test the
band. That is `O(n*w)` and is far too slow when both `n` and `w` are large, but it is unambiguous about
the intended semantics — exactly `w` elements per block, inclusive band — and is the reference against
which any faster method must agree.

The standard acceleration is **prefix sums**. Define an exclusive prefix `P[0] = 0` and
`P[k] = a[0] + ... + a[k-1]`. Then the weight of any block is a single subtraction of two prefix
entries, turning each window query into `O(1)` and the whole count into `O(n)`. The entire difficulty
is in the indexing: with this `P`, the block starting at 0-indexed position `i` and covering
`i, ..., i+w-1` has sum `P[i+w] - P[i]`, and the valid starts are `i = 0, ..., n-w` — that is `n-w+1`
of them, **inclusive** of both ends. Shift to a 1-indexed prefix and the same window becomes
`P[s+w-1] - P[s-1]` for `s = 1, ..., n-w+1`. Mixing the two conventions, or looping to `n-w` instead of
`n-w+1`, drops or duplicates a window; comparing with `<`/`>` instead of `<=`/`>=` silently excludes
batches that sit exactly on a band edge. Any of these is a wrong answer that a small trace exposes.

## Evaluation settings

Judged on hidden tests covering: `w = 1` (every parcel is its own batch); `w = n` (a single batch
spanning the whole belt); `w > n` (no batch, answer `0`); degenerate bands `L = R` (exact-equality
counting); negative and zero weights so the band test is genuinely two-sided; window sums near
`2*10^14` so 32-bit accumulators overflow; and large `n = 2*10^5` with large `w` so an `O(n*w)`
method times out but `O(n)` does not.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, w, L, R;
    if (!(cin >> n >> w >> L >> R)) return 0;
    vector<long long> a(n + 1);
    for (long long i = 1; i <= n; i++) cin >> a[i];

    // TODO: build an exclusive prefix-sum array and count the length-w windows
    //       whose sum lies in the closed band [L, R], getting the window-start
    //       range and the prefix subtraction boundaries exactly right.
    long long count = 0;

    cout << count << "\n";
    return 0;
}
```
