# Sum of subarray minimums

## Research question

You are given an array `a[0..n-1]` of integers. Consider **every** contiguous subarray
`a[l..r]` (with `0 <= l <= r <= n-1`). For each such subarray let `min(l, r)` be the smallest
element it contains. Compute

```
S = sum over all (l, r) of min(l, r)
```

and output `S mod 1000000007`. There are `n*(n+1)/2` subarrays, so for `n = 2*10^5` an explicit
enumeration is hopeless; the interesting question is whether each element's *contribution* to `S`
can be counted directly.

This is the classic "sum of subarray minimums" question. It is the kind of contribution-counting
subproblem that hides inside histogram, stack-of-spans, and range-aggregate problems, so getting the
boundary bookkeeping exactly right — which side of a tie an element owns, and whether a span count
is inclusive or exclusive — is the whole game.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with `S mod 1000000007`. Because the modulus is taken, the printed
  value is always in `[0, 1000000006]`, even when individual `a[i]` are negative.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 2, 4]` the answer is `17`. The ten subarray minima are
`3, 1, 1, 1, 1, 1, 1, 2, 2, 4`, summing to `17`.

## Background

Enumerating subarrays is `O(n^2)`. The efficient route reframes the sum by **element ownership**:
instead of asking "what is the minimum of each subarray", ask "for each index `i`, how many
subarrays have `a[i]` as their minimum?" If that count is `c[i]`, then `S = sum_i c[i] * a[i]`.

To count `c[i]` we measure how far `a[i]` can "reach" as the minimum:

- a **left reach**: how many consecutive positions ending at `i` keep `a[i]` the minimum, i.e. how
  far left we can extend before hitting an element that is smaller than `a[i]`;
- a **right reach**: symmetrically to the right.

The product of the two reaches is the number of subarrays whose minimum is `a[i]`. A nearest-smaller
scan with a **monotonic stack** computes both reaches in `O(n)`. The subtle part is *ties*: when two
equal values both qualify as a subarray's minimum, the subarray must be credited to exactly one of
them, or it is double-counted (or dropped). The standard remedy is to break the tie by using a
**strict** comparison on one side and a **non-strict** comparison on the other.

## Evaluation settings

Judged on hidden tests covering: strictly increasing and strictly decreasing arrays, arrays with
many equal values (where the tie-breaking matters), single element (`n = 1`), empty array (`n = 0`),
arrays with negative values (so the modular fold must not print a negative), and large
`n = 2*10^5` with values near `10^9` (so reach products and the running total overflow 64-bit unless
reduced modulo).

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

    const long long MOD = 1000000007LL;

    // TODO: for each i, count subarrays whose minimum is a[i] using nearest-smaller
    //       spans (monotonic stack), break ties so each subarray is credited once,
    //       and accumulate sum_i count[i] * a[i] modulo MOD.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
