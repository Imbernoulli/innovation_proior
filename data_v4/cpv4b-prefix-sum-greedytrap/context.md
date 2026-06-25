# Splitting a shift log into the most profitable billing blocks

## Research question

A coffee cart logs, for each of `n` consecutive hours of a shift, the **net cash change**
`a[i]` for that hour (tips minus restocking, so values may be negative). At closing the
manager wants to split the **entire** log into **contiguous, non-empty billing blocks**
(every hour belongs to exactly one block, blocks do not overlap, and their order is the
order of the hours). A block is **profitable** if the sum of its hours is **strictly
positive**. The manager is paid a flat reward per profitable block, so the goal is to
choose the cut points that **maximize the number of profitable blocks**.

Output that maximum count. Concretely: over all ways to partition `a[0..n-1]` into
consecutive runs, what is the largest number of runs whose sum is `> 0`?

This is a prefix-sum partition problem: a block `(j, i]` is profitable exactly when the
running total has strictly increased from index `j` to index `i`, i.e.
`prefix[i] > prefix[j]`. The point of interest is that the natural "cut a block off the
moment its running sum turns positive" rule is **not** optimal once hours can be negative.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`); then `n` integers `a[i]`
  (`-10^9 <= a[i] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the maximum number of profitable (strictly positive
  sum) blocks over all partitions of the whole array.
- Time limit: 1 second. Memory: 256 MB.

Example: for `a = [4, -4, 2, -1, 2, -4]` the answer is `3`. The prefix array is
`[0, 4, 0, 2, 1, 3, -1]`. One optimal partition is
`[4] | [-4] | [2] | [-1, 2] | [-4]`, whose block sums are `4, -4, 2, 1, -4`, giving three
strictly positive blocks. (The "cut on first positive total" rule scores only `1` here.)

## Background

Because every hour must be covered, the unprofitable blocks are not waste — they are the
*bridges* that let the running total drop back down so that a later block can climb again.
That coupling between profitable and non-profitable blocks is what defeats local greedy.
Two families of approach are on the table before committing:

- **Greedy by running sum.** Sweep left to right accumulating the current block's sum; the
  instant it becomes strictly positive, close the block, bank a point, and start a fresh
  block. It is `O(n)` and a few lines. The open question is whether closing a block as soon
  as it turns positive can ever cost a later opportunity.
- **Prefix-sum dynamic programming.** Let `prefix[i]` be the sum of the first `i` hours and
  let `dp[i]` be the most profitable blocks achievable in a full partition of `prefix[0..i]`.
  A last block `(j, i]` is profitable iff `prefix[j] < prefix[i]`, which turns the
  transition into a range-max over previously seen prefix values. The open questions are the
  exact recurrence and how to evaluate that range-max fast enough for `n = 2*10^5`.

## Evaluation settings

Judged on hidden tests covering: all-positive logs (every hour its own block), logs with
negatives and zeros (zeros never make a block profitable), the empty log (`n = 0`), a
single hour (positive, zero, negative), all-non-positive logs (answer `0`), logs
engineered to fool the running-sum greedy, and large `n = 2*10^5` with values near `10^9`
(so prefix sums need 64-bit, reaching about `2*10^14` in magnitude).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> prefix(n + 1);
    prefix[0] = 0;
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        prefix[i] = prefix[i - 1] + x;
    }

    // TODO: maximize the number of blocks (j, i] with prefix[i] > prefix[j] over a
    //       partition of the whole array into contiguous non-empty blocks.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
