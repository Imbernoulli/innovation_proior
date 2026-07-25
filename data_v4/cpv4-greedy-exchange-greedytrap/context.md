# Fewest stamps to franking an exact amount

## Research question

A postage meter at a mail room stocks `n` distinct stamp denominations `d[0..n-1]` (in cents), each in
**unlimited** supply. A parcel needs *exactly* `A` cents of postage franked onto it. You may stick on as
many stamps as you like, repeating any denomination, in any combination, as long as the values add up
to **exactly** `A`. Output the **minimum number of stamps** that sums to exactly `A`, or `-1` if no
multiset of the available denominations sums to exactly `A`.

This is the unbounded "minimum coins to make change" objective in disguise: given unlimited supply of
each denomination, find the fewest pieces that sum to exactly `A`, or determine that no combination
of the available denominations works.

## Input / output contract

- Input (stdin):
  - the first line has two tokens, `n` and `A` (`1 <= n <= 100`, `0 <= A <= 10^5`);
  - the second line has `n` integers `d[0..n-1]` (`1 <= d[i] <= 10^5`), the distinct denominations,
    whitespace-separated.
- Output (stdout): a single line with the minimum number of stamps that sum to exactly `A`, or `-1`
  if exact franking is impossible.
- Time limit: 1 second. Memory: 256 MB.

Example: for denominations `[1, 3, 4]` and `A = 6`, the answer is `2` (use `3 + 3`).

## Evaluation settings

Judged on hidden tests covering: a range of denomination sets of varying size and composition,
exact-impossibility cases (all denominations even, `A` odd) that must print `-1`, the corner `A = 0`
(answer `0`, the empty multiset), a single denomination, and the largest scale `A = 10^5` with up to
`100` denominations near `10^5`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long A;
    if (!(cin >> n >> A)) return 0;
    vector<long long> d(n);
    for (auto &x : d) cin >> x;

    // TODO: compute the minimum number of denominations (with repetition, unlimited supply)
    //       that sum to exactly A, or -1 if no such multiset exists.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
