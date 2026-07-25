# Counting positive-product subsequences, modulo m

## Research question

You are given a sequence of `n` integers `a[0..n-1]`. The values may be **positive, negative, or
zero**. Consider every **non-empty** subsequence (i.e. every non-empty subset of positions) and look
at the **product** of its chosen values. Count how many of these subsequences have a **strictly
positive** product, and report that count **modulo a given integer `m`**.

Getting the corners right — all-negative arrays, arrays full of zeros, the empty array, and
`m = 1` — is the whole game.

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

## Evaluation settings

Judged on hidden tests covering: all-positive arrays, arrays mixing negatives and zeros, all-negative
arrays, arrays that are entirely zeros, the empty array (`n = 0`), single elements, `m = 1`, and
large `n = 2*10^5` with values near the magnitude bounds.

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
        // TODO: count non-empty subsequences with strictly positive product, modulo m.
    }

    long long answer = 0;
    cout << answer << "\n";
    return 0;
}
```
