# Counting positive-product subsequences, modulo m

## Research question

You are given a sequence of `n` integers `a[0..n-1]`. The values may be **positive, negative, or
zero**. Consider every **non-empty** subsequence (i.e. every non-empty subset of positions) and look
at the **product** of its chosen values. Count how many of these subsequences have a **strictly
positive** product, and report that count **modulo a given integer `m`**.

A product is strictly positive exactly when the subset contains no zero and an *even* number of
negative factors (zero negatives counts as even). So the answer is a parity-counting problem on the
signs of the elements — the actual magnitudes are irrelevant, only the sign of each value matters.
This is the kind of "count configurations under a sign/parity constraint, reduce mod m" task that
appears throughout combinatorics-mod problems, and getting the corners right — all-negative arrays,
arrays full of zeros, the empty array, and `m = 1` — is the whole game.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`0 <= n <= 2*10^5`, `1 <= m <= 10^9`). The second line has `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated. When `n = 0` the second line is empty or absent.
- Output (stdout): a single line with the number of non-empty strictly-positive-product
  subsequences, taken modulo `m`. The printed value must be in the range `[0, m-1]`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, -2, -5, 0, 4]` with `m = 1000000007` the answer is `7`. The positive-product
subsequences are: `{3}`, `{4}`, `{3,4}`, `{-2,-5}`, `{-2,-5,3}`, `{-2,-5,4}`, `{-2,-5,3,4}` — seven
of them; the `0` can never appear, and any subset with a single negative is excluded.

## Background

Because only signs matter, classify the elements into `P = #positives`, `N = #negatives`, and the
zeros (which are simply unusable). The count splits into two independent choices:

- **Choosing positives.** Any subset of the `P` positives is allowed, contributing a factor of `2^P`.
- **Choosing negatives.** Only an *even-sized* subset of the `N` negatives keeps the product
  positive. The number of even-sized subsets of an `N`-element set is the open quantity here, and its
  closed form has a base case that is easy to get wrong.

Two routes are on the table before committing:

- **Direct subset enumeration.** Iterate over all `2^n` subsets, track the running sign, and count.
  Obviously correct, `O(2^n)`, fine only as a brute-force oracle — it cannot survive `n = 2*10^5`.
- **Closed-form parity count.** Combine `2^P` with the count of even-sized negative subsets, subtract
  the empty subset, and reduce mod `m`. This is `O(log` of the exponents`)` via fast exponentiation;
  the open questions are the exact even-subset formula (and its `N = 0` base case) and keeping the
  final subtraction non-negative under the modulus.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, all-negative
arrays, arrays that are entirely zeros, the empty array (`n = 0`), single elements, `m = 1` (where
every answer collapses to `0`), and large `n = 2*10^5` with values near the magnitude bounds (so a
naive `2^P` overflows 64-bit before the modulus and must be reduced with modular exponentiation).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, m;
    if (!(cin >> n >> m)) return 0;

    long long P = 0, N = 0; // positives, negatives; zeros excluded
    for (long long i = 0; i < n; i++) {
        long long x;
        cin >> x;
        // TODO: classify x into P / N (and drop zeros), then count even-negative
        // subsets times 2^P, subtract the empty subset, all modulo m.
    }

    long long answer = 0;
    cout << answer << "\n";
    return 0;
}
```
