# Gift card with a spending window: maximum joy in a price band

## Research question

A store gift card must be spent so that the **total price of the items you buy lands inside a fixed
band** `[L, R]` — at least `L` and at most `R`, **both endpoints inclusive**. There are `n` items;
item `i` has an integer price `p[i] >= 1` and an integer **joy** `v[i]` that may be negative (some
items are duds you would normally avoid, but a dud might be the only way to nudge the total price up
into the band). You buy each item at most once. Among all subsets whose total price `s` satisfies
`L <= s <= R`, output the **maximum total joy**. If no subset has its price in the band, output
`IMPOSSIBLE`.

The empty subset has price `0` and joy `0`; it is a legal purchase **only when `L = 0`** (so that
`0` falls inside `[L, R]`). This is a bounded-window 0/1 knapsack. The whole difficulty lives at the
two ends of the window: whether `L` and `R` are inclusive, and exactly which prices `s` are eligible.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `L`, `R`
  (`0 <= n <= 2000`, `0 <= L <= R <= 100000`). Then `n` lines follow, line `i` holding two integers
  `p[i]` (`1 <= p[i] <= 100000`) and `v[i]` (`-10^9 <= v[i] <= 10^9`).
- Output (stdout): a single line — the maximum total joy over all subsets whose total price is in
  `[L, R]` inclusive, or the word `IMPOSSIBLE` if there is no such subset.
- Time limit: 1 second. Memory: 256 MB.

Example: for

```
5 10 12
4 3
5 4
6 5
7 -1
3 2
```

the answer is `9`. Buying items 2 and 3 (1-indexed) costs `5 + 6 = 11`, which is inside `[10, 12]`,
for joy `4 + 5 = 9`. (Buying items 1, 2, 5 also reaches price `12` with joy `3 + 4 + 2 = 9`.) No
in-band subset beats `9`.

## Background

This is a 0/1 knapsack whose feasibility region is a *band* of total prices rather than a single
cap. Two design questions have to be settled before coding:

- **What does the state track?** The natural choice is `dp[s]` = the best joy of a subset whose total
  price is **exactly** `s`. Then the answer is an aggregate of `dp[s]` over the eligible prices `s`.
  The alternative — `dp[s]` = best joy with price **at most** `s` — collapses the lower bound `L`
  and is awkward to query on a band, so exact-price states are cleaner here.
- **Where do the window endpoints enter?** They do *not* enter the transitions at all; they only
  decide (a) how large the `dp` array must be and (b) which prices are scanned when reading off the
  answer. Both are inclusive-boundary decisions: the array covers prices `0..R` (that is `R + 1`
  cells), and the answer ranges over `s = L, L+1, ..., R` (both ends included).

Because `v[i]` can be negative, you cannot assume "spend as much as allowed"; the optimizer may need
to include a negative-joy item purely to push the price up to `L`, or to leave value on the table to
stay at or below `R`.

## Evaluation settings

Judged on hidden tests covering: `n = 0`; bands with `L = 0` (empty subset legal) and `L > 0`
(empty subset illegal); single-item inputs where the item's price is below `L`, inside `[L, R]`,
exactly equal to `L` or to `R`, or above `R`; all-negative joys (the answer is the *least bad*
in-band subset, which can be negative); inputs with no in-band subset (`IMPOSSIBLE`); items with
`p[i] > R` (must be ignored); and large `n` and `R` near the limits so an `O(n*R)` table is required
and the running joy can exceed 32 bits.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long L, R;
    if (!(cin >> n >> L >> R)) return 0;

    vector<long long> p(n), v(n);
    for (int i = 0; i < n; i++) cin >> p[i] >> v[i];

    // TODO: among subsets whose total price s satisfies L <= s <= R (inclusive),
    //       output the maximum total joy, or "IMPOSSIBLE" if none exists.

    return 0;
}
```
