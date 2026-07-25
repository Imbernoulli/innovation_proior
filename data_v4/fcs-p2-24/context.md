# Minimum number of perfect squares summing to n

## Research question

Given a positive integer `n`, write it as a sum of perfect-square numbers
(`1, 4, 9, 16, 25, ...`) using **as few terms as possible**, and output that
minimum count. Squares may be reused (e.g. `12 = 4 + 4 + 4`), and there is no
limit on how many terms you use other than minimizing the count.

Every integer `n >= 1` has at least the trivial decomposition `n = 1 + 1 + ... + 1`
(`n` ones), so an answer always exists; the task is to find the smallest number
of squares.

## Input / output contract

- Input (stdin): a single integer `n` with `0 <= n <= 10^6`.
- Output (stdout): a single line with the minimum number of perfect squares that
  sum to exactly `n`. For `n = 0` the answer is `0` (the empty sum).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `n = 12` the answer is `3`, because `12 = 4 + 4 + 4` (three squares),
and no decomposition into two squares exists.

## Evaluation settings

Judged on hidden tests covering: small values (`n = 0, 1, 2, 3, ...`), perfect
squares themselves, a range of other representative values, and the maximum
`n = 10^6` (to confirm the chosen method fits in time and memory).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // TODO: compute the minimum number of perfect squares summing to n.
    int answer = 0;

    cout << answer << "\n";
    return 0;
}
```
