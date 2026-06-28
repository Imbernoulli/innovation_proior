# Place k items to maximize the minimum gap

## Research question

You are given `n` positions on a line, `p[0..n-1]` (integers, possibly repeated, in any order),
and an integer `k`. Treat each position as a slot that can hold at most one item. Place exactly
`k` items into `k` of the slots so that the **minimum** distance between any two chosen positions is
as **large** as possible. Output that maximum achievable minimum gap.

This is the canonical *max-min placement* problem (also known as "aggressive cows" / "maximum
minimum distance" / spreading `k` balls to maximize the smallest separation). The difficulty is that
the objective is a maximin: we are not summing or counting, we are pushing up the worst (smallest)
pairwise gap, and the chosen positions interact globally — moving one item changes which pair is the
tightest.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `k` (`1 <= k <= n <= 2*10^5`); then `n`
  integers `p[i]` (`0 <= p[i] <= 10^9`), whitespace-separated, in arbitrary order, possibly with
  duplicates.
- Output (stdout): a single line with the maximum achievable minimum pairwise gap.
- Convention: if `k == 1` there is no pair, so the minimum gap is undefined; report `0`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `p = [1, 2, 8, 4, 9]` and `k = 3`, the answer is `3` (choose positions `1, 4, 8`,
whose consecutive gaps are `3` and `4`, so the minimum gap is `3`; no placement of 3 items does
better).

## Background

Two ingredients matter. First, once the positions are **sorted**, a set of chosen positions has its
minimum *pairwise* gap equal to the minimum of its consecutive (adjacent-in-the-sorted-order) gaps —
so we only ever reason about neighbours, not all `O(k^2)` pairs. Second, the search space of which
`k` of `n` slots to pick is astronomically large (`C(n,k)`), so brute-force subset enumeration is out
for the stated `n`.

Two families of approach are on the table before committing to one:

- **Direct combinatorial optimization / subset search.** Enumerate or cleverly prune subsets of size
  `k`. Obviously correct but exponential; only viable for tiny `n` (used here as the offline oracle).
- **Decision + search reduction.** Replace the optimization "what is the largest gap?" by the
  *decision* "can we place `k` items with all gaps at least `d`?" — and search over `d`. The open
  questions are whether that decision is easy to answer and whether the answers behave regularly
  enough in `d` to search efficiently.

## Evaluation settings

Judged on hidden tests covering: `k == 1` (degenerate, answer `0`); `k == n` (must take every slot,
answer is the minimum adjacent gap of all positions); all-equal positions (answer `0`); two extreme
points (answer is the full span); heavy duplicates; unsorted input; and large `n = 2*10^5` with
coordinates up to `10^9` so the answer and span require 64-bit-safe reasoning.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long k;
    if (!(cin >> n >> k)) return 0;
    vector<long long> p(n);
    for (auto &x : p) cin >> x;

    // TODO: place k items on these positions so the minimum pairwise gap is maximized;
    //       output that maximum minimum gap.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
