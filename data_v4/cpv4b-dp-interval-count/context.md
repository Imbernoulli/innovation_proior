# Counting barcode labels with bounded ink runs

## Research question

A label printer lays down a single row of `n` cells from left to right. Each cell is filled with
exactly one of `k` ink colors. The machine has a mechanical quirk: it can only print a *stripe* — a
maximal block of consecutive same-color cells — whose length is between `A` and `B` cells inclusive.
A finished label is just the resulting string of colors; two print jobs that produce the same row of
colors are the *same label*.

Count how many distinct labels of length `n` the printer can produce, i.e. the number of strings of
length `n` over `k` colors in which **every maximal monochromatic run has length between `A` and `B`
inclusive**. Because the count is astronomical, report it modulo a given integer `M`.

## Input / output contract

- Input (stdin): five integers on one line (whitespace-separated):
  `n` `k` `A` `B` `M`, with
  `0 <= n <= 2*10^6`, `1 <= k <= 10^9`, `1 <= A <= B <= 10^9`, `1 <= M <= 10^9`.
- Output (stdout): a single line with the number of valid labels of length `n`, taken modulo `M`.
- The empty label (`n = 0`) has no runs at all, so it is vacuously valid; the count is `1` (reduced
  mod `M`, so `0` when `M = 1`).
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 4`, `k = 3`, `A = 2`, `B = 2`, `M = 1000000007` the answer is `6`. With every
stripe forced to length exactly 2, a length-4 label is two stripes of two cells each whose colors
differ: the first stripe has 3 color choices, the second has 2, giving `3 * 2 = 6`. The six labels are
`0011, 0022, 1100, 1122, 2200, 2211`.

## Evaluation settings

Judged on hidden tests covering: tiny `n` (including `n = 0` and `n = 1`); the single-color case
`k = 1`; narrow windows (`A = B`, so only one stripe length is legal); wide windows (`A = 1`,
`B >= n`, so every string is valid); small moduli including `M = 1` and composite `M`; large `k`
near `10^9`; and large `n = 2*10^6` for time and memory.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, k, A, B, M;
    if (!(cin >> n >> k >> A >> B >> M)) return 0;   // empty input

    // TODO: count length-n strings over k colors in which every maximal
    // monochromatic run has length in [A, B], modulo M. The empty string
    // (n == 0) is vacuously valid.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
