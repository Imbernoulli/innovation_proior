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

Example: for `a = [2, -2, -2, 0, 2]` the answer is `8`, from the window `a[0..2] = 2 * (-2) * (-2)`.

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, the empty
array (`n = 0`), a single element (`n = 1`, including a lone negative and a lone zero), all-negative
arrays of both even and odd length, runs split by interior zeros, and the size extreme `n = 62` with
values `+/-2`.

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
