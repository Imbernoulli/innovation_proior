# Fewest stamps to franking an exact amount

## Research question

A postage meter at a mail room stocks `n` distinct stamp denominations `d[0..n-1]` (in cents), each in
**unlimited** supply. A parcel needs *exactly* `A` cents of postage franked onto it. You may stick on as
many stamps as you like, repeating any denomination, in any combination, as long as the values add up
to **exactly** `A`. Output the **minimum number of stamps** that sums to exactly `A`, or `-1` if no
multiset of the available denominations sums to exactly `A`.

This is the unbounded "minimum coins to make change" objective in disguise. It is the canonical place
where a locally-optimal greedy — always reach for the largest denomination that still fits — looks
obviously right and is, for general denomination sets, *wrong*. The question is which denominations and
how many of each minimize the stamp count, and how to compute it without falling into the greedy trap.

## Input / output contract

- Input (stdin):
  - the first line has two tokens, `n` and `A` (`1 <= n <= 100`, `0 <= A <= 10^5`);
  - the second line has `n` integers `d[0..n-1]` (`1 <= d[i] <= 10^5`), the distinct denominations,
    whitespace-separated.
- Output (stdout): a single line with the minimum number of stamps that sum to exactly `A`, or `-1`
  if exact franking is impossible.
- Time limit: 1 second. Memory: 256 MB.

Example: for denominations `[1, 3, 4]` and `A = 6`, the answer is `2` (use `3 + 3`). The tempting
greedy that grabs the largest stamp first would take `4 + 1 + 1 = 6`, i.e. `3` stamps — strictly worse.

## Background

The constraint "sum to *exactly* `A` with the fewest pieces" is a constrained optimization over an
unbounded multiset. Two families of approach are on the table before committing to one:

- **Greedy by denomination.** Repeatedly stick on the largest denomination `<= remaining`, subtract,
  repeat until the remainder is `0` (success) or no denomination fits (declare failure). This is
  `O(n log n + A / d_min)` and a handful of lines. The open question — and the whole point of this
  problem — is whether always grabbing the largest stamp is actually optimal for an arbitrary
  denomination set, or only for special "canonical" systems.
- **Unbounded shortest-combination DP.** Define `dp[v]` = fewest stamps summing to exactly `v`, fill
  `v` from `1` up to `A`, relaxing through every denomination. This is `O(n * A)`; the open question is
  the exact recurrence, the base case, and how impossibility (`-1`) propagates.

## Evaluation settings

Judged on hidden tests covering: canonical systems where greedy happens to be optimal (e.g. `1,5,10,25`),
deliberately non-canonical systems where greedy overshoots the stamp count (e.g. `1,3,4` or `1,5,8`),
exact-impossibility cases (all denominations even, `A` odd) that must print `-1`, the corner `A = 0`
(answer `0`, the empty multiset), a single denomination, and the largest scale `A = 10^5` with up to
`100` denominations near `10^5` (so the DP table is the intended `O(n*A)` and stamp counts can be large).

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
