# Total handshake value over all pairs

## Research question

A conference has `n` attendees standing in a room. Attendee `i` carries an *influence* value `a[i]`
(a non-negative integer). When two distinct attendees `i` and `j` shake hands, the handshake produces
a value equal to the product `a[i] * a[j]`. Every unordered pair of distinct attendees shakes hands
exactly once. Compute the **total handshake value**, i.e. the sum of `a[i] * a[j]` over all unordered
pairs `i < j`, and print it.

This is a small math-adhoc problem: the entire task is to recognize the closed form for the sum of
all pairwise products and to carry the arithmetic in a wide enough integer type. The naive double
loop is `O(n^2)` and too slow at the stated size, so the interesting work is the algebraic
identity — and the trap is that the answer is astronomically larger than a 32-bit integer can hold.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 10^5`); then `n` integers `a[i]`
  (`0 <= a[i] <= 10^4`), whitespace-separated.
- Output (stdout): a single line with the total handshake value (the sum of `a[i] * a[j]` over all
  unordered pairs `i < j`). If `n < 2` there are no pairs and the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [3, 1, 4, 1, 5]` the answer is `72`
(`3*1 + 3*4 + 3*1 + 3*5 + 1*4 + 1*1 + 1*5 + 4*1 + 4*5 + 1*5 = 72`).

## Background

The quantity wanted is `S = sum_{i<j} a[i] * a[j]`. Two routes are on the table before committing:

- **Direct double loop.** For every pair `(i, j)` with `i < j`, add `a[i] * a[j]`. This is
  unambiguously correct but `O(n^2)`; at `n = 10^5` that is `~5*10^9` multiply-adds, far over the
  time limit. Useful only as a reference brute force on small inputs.
- **Closed-form identity.** Expanding the square of the total, `(sum a[i])^2 = sum a[i]^2 +
  2 * sum_{i<j} a[i] * a[j]`. Rearranging gives `S = ((sum a[i])^2 - sum a[i]^2) / 2`, computable in
  a single `O(n)` pass that accumulates `sum` and `sum of squares`. The open questions are getting
  the `/2` and the `- sum of squares` exactly right, and choosing integer types wide enough that no
  intermediate (in particular `sum^2`) and no final value silently wraps.

## Evaluation settings

Judged on hidden tests covering: tiny arrays (`n = 0`, `n = 1`, `n = 2`), arrays containing zeros,
arrays of equal values, and large `n = 10^5` with values near `10^4` — where the total handshake
value is on the order of `5*10^17`, so it overflows a 32-bit integer by many orders of magnitude and
even the intermediate `sum^2` reaches `~10^18`.

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
