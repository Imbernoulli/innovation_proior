# Fewest weights to balance the vault scale

## Research question

A museum vault opens only when a single balance scale is brought to *exact* equilibrium. On the left
pan sits a fixed counterweight of `S` grams. In the storeroom there are `n` calibration weights with
masses `w[0..n-1]` grams; each physical weight may be used at most once, and you place a chosen subset
on the right pan. The vault opens when the right pan's total mass equals `S` exactly.

The curator does not just want *some* working combination — carrying weights up the stairs is slow, so
she wants to use **as few physical weights as possible**. Your job is to **output an actual subset**:
the indices of a minimum-size collection of weights whose masses sum to exactly `S`. If no subset
sums to `S`, report that the vault cannot be opened.

This is the *construction* version of subset-sum, not the decision version. It is not enough to know a
solution exists or to know the minimum count — you must emit a concrete list of indices, and that list
is checked: the chosen masses must sum to exactly `S`, the indices must be distinct and in range, and
the count must equal the true minimum. A construction that "looks right" on four-weight examples but
is suboptimal or infeasible on larger inputs scores zero, because the judge verifies the property on
inputs at the full constraint scale.

## Input / output contract

- Input (stdin): the first line contains two integers `n` and `S`
  (`0 <= n <= 5000`, `0 <= S <= 5000`). The second line contains `n` integers `w[i]`
  (`0 <= w[i] <= 10^9`), whitespace-separated. (Weights may exceed `S`; such a weight can never be on
  the right pan of a solution, but it is still a valid input.)
- Output (stdout):
  - If some subset of the weights sums to exactly `S`: print `k` on the first line (the number of
    chosen weights, `k >= 0`), then on the second line the `k` chosen **1-based** indices in strictly
    ascending order, space-separated. When `k = 0` (only possible when `S = 0`), the second line is
    empty.
  - If no subset sums to `S`: print a single line containing `-1`.
- Any minimum-size subset is accepted; if several exist, output any one of them.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 5`, `S = 19`, `w = [1, 12, 6, 8, 11]` the answer is `2` weights — indices `4 5`
(masses `8 + 11 = 19`). No single weight equals `19`, so `2` is optimal.

## Background

Two ways to attack this are on the table before committing:

- **Largest-first greedy construction.** Sort the weights descending; repeatedly add the largest weight
  that still fits under the remaining target, subtracting as you go, until the remainder hits `0`. It is
  `O(n log n)`, trivial to emit, and produces a concrete subset directly. The open question is whether
  "always grab the biggest weight that fits" ever uses more weights than necessary — or worse, paints
  itself into a corner and declares an openable vault unopenable.
- **Bounded-target knapsack DP.** Because `S <= 5000`, the set of reachable pan-totals lives in
  `[0, S]`. Compute, for every reachable total, the minimum number of weights that produces it, then
  reconstruct one optimal subset. This is `O(n * S)` time. The open question is the exact recurrence
  and — the part that bites — how to recover an actual subset from the DP **without ever reusing a
  physical weight**, since each weight is a distinct object.

The trap this problem is built around: a construction that is correct on tiny inputs can be confidently
wrong at scale. A greedy that opens every four-weight vault in your head can, on a fifty-weight vault,
use one weight too many (still "valid" but rejected for non-minimality) or even falsely report `-1`.
The defense is to verify the constructed subset's property — exact sum, distinctness, minimum count —
on inputs as large as the constraints allow, not just on hand examples.

## Evaluation settings

Judged on hidden tests covering: feasible vaults with a unique minimum subset; feasible vaults with
many tied minimum subsets (repeated weight values); infeasible vaults (`-1`); the `S = 0` corner
(answer is the empty subset, `k = 0`); `n = 0`; weights that exceed `S` (must be ignored, never chosen);
weights equal to `0` (never help reach a positive total and must never pad a subset); and large
`n = S = 5000` instances where an `O(n*S)` DP and its reconstruction must finish within the limit and
within 256 MB.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long S;
    if (!(cin >> n >> S)) return 0;
    vector<long long> w(n);
    for (auto &x : w) cin >> x;

    // TODO: output the 1-based indices of a MINIMUM-size subset of the weights whose
    //       masses sum to exactly S (any minimum-size subset is accepted), or -1 if
    //       no subset sums to S. Print k on line 1, then the k indices ascending.

    return 0;
}
```
