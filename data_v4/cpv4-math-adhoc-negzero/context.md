# Maximum product of a contiguous subarray (empty allowed)

## Research question

You are given a sequence of `n` integers `a[0..n-1]` whose values may be negative or zero. Consider
every **contiguous** subarray `a[i..j]` and the product of its elements. The **empty** subarray is
also allowed, and its product is the empty product `1`. Output the **largest** product achievable
over all subarrays (empty or not). Because the empty subarray is always available, the answer is
**at least `1`**.

This is the multiplicative cousin of maximum-subarray-sum, and the multiplication is what makes it
treacherous: a negative element flips the sign of a running product, a zero resets it, and the
"empty product is `1`" rule sets a floor that an all-negative or all-zero input cannot beat with any
single window. Getting the sign bookkeeping and the base case exactly right — including the
all-negative and empty corners — is the whole problem.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 62`); then `n` integers `a[i]`
  (`-2 <= a[i] <= 2`), whitespace-separated. `n = 0` denotes the empty array (no values follow).
- Output (stdout): a single line with the maximum achievable product.
- Time limit: 1 second. Memory: 256 MB.

The value range `|a[i]| <= 2` with `n <= 62` bounds every subarray product by `2^62 < 2^63`, so the
answer always fits in a signed 64-bit integer; you never need bignum, but you do need `long long`.

Example: for `a = [2, -2, -2, 0, 2]` the answer is `8`, from the window `a[0..2] = 2 * (-2) * (-2)`.

## Background

The constraint "contiguous, and multiply" turns this into a running-state problem. Two families of
approach are on the table before committing to one:

- **Track only the running maximum product.** Carry the best product of a subarray ending at the
  current index; extend it by `a[i]` or restart at `a[i]`, mirroring Kadane's sum algorithm. The
  open question is whether a *single* running maximum suffices when a negative `a[i]` can turn a tiny
  (very negative) product into a huge positive one.
- **Track the running maximum and minimum together.** Carry both the largest and smallest product of
  a subarray ending at the current index, because multiplying by a negative swaps their roles. The
  open question is the exact transition and — sharper here — the base case: what the "answer so far"
  starts at, and whether the empty-subarray floor of `1` is applied correctly so that all-negative
  and empty inputs return `1` rather than a negative or a zero.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, the empty
array (`n = 0`), a single element (`n = 1`, including a lone negative and a lone zero), all-negative
arrays of both even and odd length, runs split by interior zeros, and the size extreme `n = 62` with
values `+/-2` (so the product reaches `2^62` and must not overflow a 64-bit accumulator).

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

    // TODO: compute the maximum product over all contiguous subarrays, where the
    // empty subarray is allowed and contributes the empty product 1.
    long long answer = 1;

    cout << answer << "\n";
    return 0;
}
```
