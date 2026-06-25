# Collapsing a row of crate stacks with least total effort

## Research question

A warehouse robot faces a single row of `n` stacks of crates; stack `i` holds `w[i]` crates. The robot
clears the row by repeatedly **shoving two adjacent stacks together** into one combined stack. Shoving a
stack of `x` crates against an adjacent stack of `y` crates costs `x + y` units of effort (it has to drag
all `x + y` crates), and the result is a single stack of `x + y` crates sitting in their place. The robot
keeps doing this until only one stack remains. Different orders of shoving cost different total effort.

Compute the **minimum total effort** to collapse the whole row into one stack.

Only *adjacent* stacks may be combined, so the order is constrained by position on the line — this is the
"merge along a line" interval-DP problem (a cousin of optimal matrix-chain ordering and the stone-merging
classic). Getting it right means getting the interval recurrence right *and* picking data types that survive
the largest inputs, because the accumulated effort grows much faster than any single stack.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`); then `n` integers `w[i]`
  (`0 <= w[i] <= 10^6`), whitespace-separated.
- Output (stdout): a single line with the minimum total effort.
- If `n <= 1` there are no shoves to make, so the effort is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for `w = [3, 1, 4, 2]` the answer is `20` (shove the `3` and `1` into a `4` for effort `4`, giving
`[4, 4, 2]`; shove the right `4` and `2` into a `6` for effort `6`, giving `[4, 6]`; shove those for effort
`10`; total `4 + 6 + 10 = 20`).

## Background

Two structural facts drive the design:

- **The cost of one merge is the size of the resulting stack**, and the size of any stack that ever exists is
  the sum of a *contiguous* run `w[l..r]` of original stacks (adjacency is never broken). So a contiguous
  range `[l..r]` is the natural subproblem: the least effort to fuse exactly that range into one stack.
- **The last merge of a range `[l..r]` splits it at some boundary `k`**: the range was first reduced to one
  stack covering `[l..k]` and one covering `[k+1..r]`, then those two were shoved together at cost equal to
  the total crates in `[l..r]`. That total does not depend on `k`, so it factors out of the choice.

Approaches on the table before committing:

- **Greedy (always merge the cheapest adjacent pair).** `O(n log n)`-ish and easy to write; the open question
  is whether a locally cheapest shove is ever globally suboptimal on a line.
- **Interval dynamic programming.** Define the least effort to fuse each range, fill ranges by increasing
  length, and split at the best boundary using prefix sums for the range total. `O(n^3)`; the open questions
  are the exact recurrence and the numeric range of the accumulated values.

## Evaluation settings

Judged on hidden tests covering: tiny rows (`n = 0`, `n = 1`, `n = 2`), rows with zero-sized stacks, rows where
greedy and optimal diverge, and large rows (`n = 500`) with weights near `10^6` where the *accumulated* effort
runs to several times `10^9` even though every single stack still fits comfortably in 32 bits — so a 32-bit
accumulator silently overflows while small tests pass.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;
    if (n <= 1) { cout << 0 << "\n"; return 0; }

    // TODO: interval DP — least effort to fuse each contiguous range into one stack,
    //       using prefix sums for the per-range merge cost. Output dp over the whole row.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
