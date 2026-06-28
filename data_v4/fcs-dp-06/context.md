# Counting disjoint-mask pairs via sum-over-subsets

## Research question

You are given `n` non-negative integers `a[0..n-1]`, each strictly less than `2^20`. Interpret each
value as a 20-bit mask. Count the number of **unordered index pairs** `(i, j)` with `i < j` such that
the two masks are **disjoint** — that is,

```
a[i] AND a[j] == 0   (bitwise AND, no shared set bit)
```

Output that count. The empty value `0` is disjoint from every value (including another `0`), so pairs
of zeros count. The challenge is the scale: `n` can be as large as `10^6`, so the naive "compare every
pair" approach does `~5 * 10^11` operations and cannot finish in time. The interesting question is how
to count disjoint pairs without ever forming a pair explicitly.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 10^6`); then `n` integers `a[i]`
  (`0 <= a[i] < 2^20 = 1048576`), whitespace-separated.
- Output (stdout): a single line with the number of unordered pairs `i < j` with `a[i] AND a[j] == 0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `a = [1, 2, 3, 0]` the answer is `4`. The disjoint pairs are
`(a0,a1)=1&2=0`, `(a0,a3)=1&0=0`, `(a1,a3)=2&0=0`, and `(a2,a3)=3&0=0`; the pairs
`(a0,a2)=1&3=1` and `(a1,a2)=2&3=2` share a bit and do not count.

## Background

The constraint `a[i] AND a[j] == 0` says: the set bits of `a[j]` must avoid every set bit of `a[i]`.
Equivalently, `a[j]` must be a **submask** of the complement `comp_i = (2^20 - 1) XOR a[i]` (the mask
with exactly the bits that `a[i]` does not use). So for a fixed `i`, the number of valid partners is
"how many array values are submasks of `comp_i`".

Two families of approach are on the table before committing:

- **Pairwise enumeration.** For every pair `(i, j)` test `a[i] AND a[j] == 0`. This is `O(n^2)` and
  trivially correct, but at `n = 10^6` it is hopeless; even at `n = 2000` it is the right tool only as a
  reference oracle.
- **Counting over the mask lattice.** Build a histogram of the values over the `2^20` possible masks,
  then for each mask `m` precompute `f[m]` = "number of array values that are submasks of `m`". With
  `f` in hand, the partner count for index `i` is just `f[comp_i]` — an `O(1)` lookup. The open
  question is how to compute `f` over all `2^20` masks fast enough; doing it mask-by-mask by iterating
  submasks is `3^20` in the worst case and is itself too slow.

## Evaluation settings

Judged on hidden tests covering: tiny arrays (`n = 0, 1`), arrays that are all zeros (the answer is
`C(n,2)` and exceeds a 32-bit integer), arrays using the full 20-bit width, sparse masks (few bits set,
so many pairs are disjoint), zero-heavy arrays (stressing the self-pair correction), duplicate values,
and large `n = 10^6` (so the algorithm must be near-linear in `n` plus the lattice transform).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> a(n);
    for (auto &x : a) cin >> x;

    // TODO: count unordered pairs (i, j), i < j, with a[i] AND a[j] == 0,
    //       without enumerating all O(n^2) pairs.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
