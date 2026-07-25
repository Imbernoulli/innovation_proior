# Total handshake value over all pairs

## Research question

A conference has `n` attendees standing in a room. Attendee `i` carries an *influence* value `a[i]`
(a non-negative integer). When two distinct attendees `i` and `j` shake hands, the handshake produces
a value equal to the product `a[i] * a[j]`. Every unordered pair of distinct attendees shakes hands
exactly once. Compute the **total handshake value**, i.e. the sum of `a[i] * a[j]` over all unordered
pairs `i < j`, and print it.

This is a small math-adhoc problem. The naive double loop is `O(n^2)` and too slow at the stated
size, so a faster way to compute the sum is needed.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 10^5`); then `n` integers `a[i]`
  (`0 <= a[i] <= 10^4`), whitespace-separated.
- Output (stdout): a single line with the total handshake value (the sum of `a[i] * a[j]` over all
  unordered pairs `i < j`). If `n < 2` there are no pairs and the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 4, 1, 5]` the answer is `72`
(`3*1 + 3*4 + 3*1 + 3*5 + 1*4 + 1*1 + 1*5 + 4*1 + 4*5 + 1*5 = 72`).

## Evaluation settings

Judged on hidden tests covering: tiny arrays (`n = 0`, `n = 1`, `n = 2`), arrays containing zeros,
arrays of equal values, and large `n = 10^5` with values near `10^4`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;

    // TODO: read the n values and compute the sum of a[i]*a[j] over all unordered pairs i<j.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
