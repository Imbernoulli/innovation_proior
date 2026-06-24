# Position-weighted count of leftmost-minimum ownership

## Research question

You are given an array `a[0..n-1]` of positive integers. For every subarray `a[l..r]`
(`0 <= l <= r <= n-1`) we look at its **minimum value**, and we declare the subarray to be
**owned** by exactly one index: the index of its minimum element, with ties broken by taking the
**leftmost** position that attains the minimum. So each of the `n(n+1)/2` subarrays has a single,
well-defined owner.

Let `c[i]` be the number of subarrays owned by index `i`. Output

```
S = ( sum over i of  i * c[i] )  mod (10^9 + 7),
```

a **position-weighted** total of the ownership counts (the index `i` is 0-based). Because every
subarray has exactly one owner, `sum_i c[i] = n(n+1)/2` always holds — a useful invariant to check
against.

This is the counting flavour of the classic "contribution of each element as a minimum" technique:
the difficulty is not the `O(n)` monotonic-stack skeleton but getting the **tie-break between equal
elements exactly right** so that no subarray is double-counted and none is dropped.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`1 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with `S`, the position-weighted ownership total, modulo
  `10^9 + 7`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [2, 1, 2, 1, 3]` the ownership counts are `c = [1, 8, 1, 4, 1]` (they sum to
`15 = 5*6/2`), so `S = 0*1 + 1*8 + 2*1 + 3*4 + 4*1 = 26`.

## Background

For a single fixed index `i`, the set of subarrays for which `a[i]` is *a* minimum is a rectangle:
it is determined by how far left and how far right you can extend before hitting a strictly smaller
value. The standard `O(n)` tool is a **monotonic stack** that finds, for each `i`, the nearest
smaller element on each side. The count of such subarrays is `(i - L) * (R - i)` where `L` is the
left barrier index and `R` the right barrier index.

The subtlety lives entirely in **equal elements**. When several positions share the minimum value,
the rectangles of those positions overlap, and a naive "use `<` on both sides" or "use `<=` on both
sides" either double-counts the shared subarrays or drops some. With the **leftmost-minimum** tie
break, the correct asymmetric convention is the question to resolve before writing code.

- **Brute force.** For every left endpoint `l`, sweep `r` to the right, track the running minimum
  and its (leftmost) owner index, and increment that owner. `O(n^2)`; obviously correct, but far too
  slow at `n = 2*10^5`.
- **Monotonic stack.** Compute the two barriers per index in two linear passes and accumulate
  `(i - L) * (R - i)` weighted by `i`. `O(n)`; the open question is the precise `<` vs `<=`
  asymmetry that matches the leftmost tie-break.

## Evaluation settings

Judged on hidden tests covering: `n = 0` and `n = 1`; arrays of all-equal values (maximum tie
density, where the tie-break convention is fully exercised); strictly increasing and strictly
decreasing arrays; many small distinct values with heavy repetition; and large `n = 2*10^5` with
values up to `10^9` (so the intermediate `i * c[i]` must be reduced under the modulus and the count
`(i - L)*(R - i)` must be formed in 64-bit before taking the modulus).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long MOD = 1000000007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<long long> a(n);
    for (auto &x : a) cin >> x;

    // TODO: for each i, count the subarrays owned by i under the leftmost-minimum
    // tie-break, then output sum_i (i * c[i]) mod 1e9+7.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
