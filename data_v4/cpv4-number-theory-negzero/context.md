# Largest common tuning step over a sub-ring of bells

## Research question

A foundry is retuning a long ring of `n` bells. Bell `i` is currently mistuned by `a[i]` cents — an
integer that may be **negative** (the bell is flat), **zero** (the bell is already in tune), or
positive (the bell is sharp). A retuning rig applies one fixed *master step* of `d` cents (a positive
integer) and may apply it to any bell **any whole number of times in either direction**. So bell `i`
can be driven exactly back to true iff `a[i]` is an integer multiple of `d`, i.e. iff `d` divides
`a[i]`.

For a contiguous block of bells `l..r` you are asked for the **largest** master step `d` that
simultaneously corrects every bell in that block. A bell with `a[i] = 0` is already in tune and any
step works for it, so it imposes **no** constraint. If *every* bell in the block is already in tune
(all zeros), then every positive `d` works and there is no largest one; by convention you report `0`
to mean "unbounded — no constraint". Because `d | a[i]` is equivalent to `d | |a[i]|`, the sign of a
mistuning is irrelevant to which steps divide it.

This is range-GCD with signed values, and the whole difficulty is in the corners the story forces on
you: negatives must be folded to magnitudes, zeros must act as identity, and an all-zero block must
collapse to `0` rather than to a garbage value or a crash. Getting the **base/identity element of the
gcd fold exactly right** is the crux.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `q` (`1 <= n <= 2*10^5`, `1 <= q <= 2*10^5`): the number
    of bells and the number of queries.
  - The second line has `n` integers `a[0..n-1]` (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
  - Each of the next `q` lines has two integers `l` and `r` (`1 <= l <= r <= n`), a 1-indexed
    inclusive block.
- Output (stdout): for each query, one line — the largest positive integer dividing every `|a[i]|`
  for `l <= i <= r`, or `0` if the whole block is zero.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
7 4
-12 18 0 -30 0 24 9
1 7
1 2
3 5
6 7
```

produces

```
3
6
30
3
```

Query `1..7` is `gcd(12,18,0,30,0,24,9)=3`; query `1..2` is `gcd(12,18)=6`; query `3..5` is
`gcd(0,30,0)=30` (the zeros drop out, leaving the lone `30`); query `6..7` is `gcd(24,9)=3`.

## Background

The set of master steps that correct bell `i` is exactly the set of positive divisors of `|a[i]|`
(with the convention that *every* positive integer divides `0`). A step corrects a whole block iff it
divides every member, so the steps that work form the common-divisor set of the block, and the
**largest** common divisor is the greatest common divisor of the magnitudes. The empty (all-zero) case
has every positive integer as a common divisor and so no greatest one — reported as `0`, which is also
the value the standard gcd recursion yields for `gcd(0,0,...,0)`, keeping the convention consistent.

Two implementation routes are on the table before committing:

- **Per-query gcd scan.** For each query, fold `gcd` across `|a[l]|..|a[r]|`. Trivial to write and
  obviously correct, but `O(q * n)` in the worst case — up to `4*10^10` operations here, far too slow.
- **Static range-GCD structure.** Precompute a structure over `|a[i]|` that answers each block gcd
  quickly. Because gcd is associative *and idempotent* (`gcd(x,x)=x`), overlapping ranges may be
  combined freely, which a sparse table exploits for `O(1)` queries after `O(n log n)` build. The open
  questions are the identity element to seed empty/zero folds and the exact index arithmetic.

## Evaluation settings

Judged on hidden tests covering: all-positive blocks, blocks mixing negatives and zeros, single-bell
queries (`l = r`), blocks that are entirely zero (answer `0`), all-negative blocks (sign must be
stripped), values at the magnitude extremes `±10^9`, and large `n, q = 2*10^5` so an `O(q*n)` scan
times out and the precompute is mandatory.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    // TODO: build a range-GCD structure over |a[i]| (gcd is idempotent), then for each query [l, r]
    // output gcd(|a[l]|,...,|a[r]|); an all-zero block must yield 0.

    return 0;
}
```
