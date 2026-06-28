# Maximum dot product of two equal-length subsequences

## Research question

You are given two integer arrays, `A` of length `n` and `B` of length `m` (values may be
negative). Pick a **non-empty** subsequence of `A` and a **non-empty** subsequence of `B` of the
**same length** `k` (`k >= 1`), keeping the original left-to-right order within each array, and
align them position by position. The score of such a choice is the dot product

```
A[i_1]*B[j_1] + A[i_2]*B[j_2] + ... + A[i_k]*B[j_k]
```

where `i_1 < i_2 < ... < i_k` are the chosen indices in `A` and `j_1 < j_2 < ... < j_k` the chosen
indices in `B`. Output the **maximum** achievable score over all valid choices of `k` and of the two
subsequences. Because at least one pair must be chosen, the answer can be negative.

This is the core "align two sequences for maximum reward" subproblem that appears inside sequence
comparison, bilingual alignment, and trading/path DP problems. Getting it exactly right hinges on
the negative-value corners and on the fact that you must pair the chosen elements **in order**.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m` (`1 <= n, m <= 500`). The second line
  has the `n` integers of `A`; the third line has the `m` integers of `B`. Each value satisfies
  `-1000 <= A[i], B[j] <= 1000`. Tokens are whitespace-separated; line breaks are not significant.
- Output (stdout): a single line with the maximum achievable dot product.
- Time limit: 1 second. Memory: 256 MB.

Example: for `A = [2, 1, -2]`, `B = [3, 0, -1]` the answer is `8`: pair `A[0]*B[0] = 6` with
`A[2]*B[2] = (-2)*(-1) = 2`, total `8`. This beats the single best product `A[0]*B[0] = 6`.

## Background

Two families of approach are on the table before committing to one:

- **Greedy by best product.** Since a length-1 choice is always legal, one tempting shortcut is to
  return the single largest entrywise product `max_{i,j} A[i]*B[j]`, or to sort both arrays and pair
  the largest magnitudes together. These are `O(nm)` or `O(n log n + m log m)` and trivial to write;
  the open question is whether a single dominant product (or a sign-aware sorted pairing) is actually
  optimal once longer alignments and the in-order constraint are taken into account.
- **Alignment dynamic programming.** Scan both prefixes and maintain, for each pair of prefix lengths
  `(i, j)`, the best dot product of a non-empty aligned pairing using `A[0..i-1]` and `B[0..j-1]`.
  This is `O(nm)`; the open question is the exact recurrence, how the "must be non-empty" requirement
  is enforced, and how negatives are handled without a phantom empty pairing leaking in.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, all-negative arrays (where two negatives
multiply to a large positive), mixed signs with zeros, the single-element corners (`n = 1` and/or
`m = 1`), cases where a length-`k` alignment with `k >= 2` strictly beats any single product, cases
where order-preservation forbids the magnitude-optimal pairing, and the largest sizes `n = m = 500`
with `|values| = 1000` (so the dot product can reach `5*10^8`).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> A(n), B(m);
    for (auto &x : A) cin >> x;
    for (auto &x : B) cin >> x;

    // TODO: compute the maximum dot product over all non-empty equal-length
    // subsequences (chosen in increasing index order within each array).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
