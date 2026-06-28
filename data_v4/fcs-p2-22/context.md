# Burst Balloons: maximum coins

## Research question

You are given `n` balloons in a row, balloon `i` painted with the number `nums[i]`. You burst the
balloons one at a time, in any order, until none remain. When you burst balloon `i`, you collect

```
nums[left] * nums[i] * nums[right]
```

coins, where `left` and `right` are the balloons immediately adjacent to `i` **among those still
unburst at that moment**. If a side has no remaining neighbour (you ran off the end of the row), treat
the missing neighbour as a balloon painted with the number `1`. After balloon `i` is burst it is
removed, so the balloons on its two sides become adjacent for all later bursts.

Choose the order of bursting that **maximizes the total number of coins collected**, and output that
maximum.

This is the classic "Burst Balloons" coin-collection problem. The order matters a great deal: a
balloon's payout depends on which neighbours survive next to it at the instant you pop it, so an early
burst reshapes the multipliers available to every later burst. The interest is that the locally
greedy instinct (pop the cheap ones first, or save the big ones for last) is not optimal, and the
question is to find the genuinely optimal order's value.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 500`), the number of balloons. Then `n` integers
  `nums[i]` (`0 <= nums[i] <= 100`), whitespace-separated.
- Output (stdout): a single line with the maximum total coins collectable.
- If `n = 0` there are no balloons and the answer is `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `nums = [3, 1, 5, 8]` the answer is `167`. One optimal order is burst `1`, then `5`,
then `3`, then `8`: `3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 15 + 120 + 24 + 8 = 167`.

## Background

The payout rule couples every decision to the current neighbourhood, which makes the order a
combinatorial object with `n!` possibilities. Two families of approach are on the table before
committing to one:

- **Greedy by value.** Fix a popping rule based on the painted numbers alone — for instance always
  burst the smallest remaining balloon first (to cash it out while a big neighbour is still present),
  or always burst the largest remaining balloon first (or last). Each is `O(n^2)` or `O(n log n)` and
  a few lines; the open question is whether any such fixed local rule is actually optimal, given that
  the constraint coupling neighbours is global.
- **Interval dynamic programming.** Add the two virtual `1` balloons at the ends, then build up
  answers for sub-rows by ranges. The trick is to reason about which balloon in a range is burst
  **last**, because that balloon's two neighbours are then exactly the fixed endpoints of the range.
  This is `O(n^3)`; the open question is the exact state, the transition, and whether `n^3` at
  `n = 500` is fast enough.

## Evaluation settings

Judged on hidden tests covering: the empty row (`n = 0`), a single balloon, rows containing `0`s,
rows where the maximum value `100` repeats, small rows where an exhaustive over-all-orders oracle can
confirm the value, and large `n = 500` rows (where any exponential or `n!` order search is hopeless
and the chosen method's running time and arithmetic range must both hold up).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (!(cin >> n)) return 0;
    vector<long long> v(n + 2);
    v[0] = 1;            // virtual boundary balloon
    v[n + 1] = 1;        // virtual boundary balloon
    for (int i = 1; i <= n; i++) cin >> v[i];

    // TODO: compute the maximum total coins over all bursting orders.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
