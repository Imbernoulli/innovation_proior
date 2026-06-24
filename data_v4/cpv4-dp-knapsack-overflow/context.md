# Loading the long-haul van (0/1 knapsack with large payoffs)

## Research question

A courier is loading one van for an overnight run. There are `n` parcels in the depot. Parcel `i`
occupies `w[i]` units of volume and, if delivered, earns a payout of `v[i]` cents. The van holds a
total volume of `W`. Each parcel is either loaded whole or left behind — parcels cannot be split, and
each exists in a single copy. Choose a subset of parcels whose total volume does not exceed `W` so
that the **total payout is maximized**, and output that maximum payout (in cents).

This is the classic 0/1 knapsack, but the numbers are chosen so that the payout figures are the trap:
a single parcel can be worth up to `10^9` cents, and a full van can hold up to a thousand parcels, so
the best total can reach the order of `10^{12}` — far outside the range of a 32-bit integer. The point
of the problem is to get the dynamic program *and* the arithmetic width right at the same time.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `W` (`0 <= n <= 1000`,
  `0 <= W <= 10^5`). Then follow `n` lines; line `i` holds two integers `w[i]` and `v[i]`
  (`0 <= w[i] <= 10^5`, `0 <= v[i] <= 10^9`).
- Output (stdout): a single line with the maximum achievable total payout.
- Time limit: 1 second. Memory: 256 MB.

Example: for `W = 10` and the four parcels `(w, v) = (3, 1000000000), (4, 1500000000),
(5, 1200000000), (2, 800000000)`, the answer is `3300000000` — load parcels 1, 2, and 4 (volumes
`3 + 4 + 2 = 9 <= 10`, payouts `1000000000 + 1500000000 + 800000000`). Note the answer already exceeds
`2^31 - 1`.

## Background

The constraint "each parcel used at most once" makes this 0/1 knapsack rather than the unbounded
variant. Two approaches are worth weighing before committing:

- **Greedy by value density.** Sort parcels by `v[i]/w[i]` and load the densest that still fit. This
  is `O(n log n)` and tempting, but greedy is known to be optimal only for the *fractional* knapsack
  where parcels can be split; with whole-parcel (0/1) loading the open question is whether density
  ordering ever loads a suboptimal set.
- **Capacity dynamic programming.** Maintain, for every volume budget `c` from `0` to `W`, the best
  payout achievable within that budget, and fold parcels in one at a time. This is `O(n*W)`; the open
  questions are the update direction (which enforces the at-most-once rule) and — crucially here — the
  width of the numbers being accumulated.

## Evaluation settings

Judged on hidden tests covering: all parcels fitting, none fitting (every `w[i] > W`), zero-volume
parcels (which should always be loaded if `v[i] > 0`), `W = 0`, `n = 0`, ties in value density, and
large instances with `n = 1000`, `W = 10^5`, and payouts near `10^9` so the optimal total exceeds a
32-bit integer.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long W;
    if (!(cin >> n >> W)) return 0;

    vector<long long> wt(n), val(n);
    for (int i = 0; i < n; i++) cin >> wt[i] >> val[i];

    // TODO: compute the maximum total payout of a subset of parcels with total volume <= W,
    //       each parcel used at most once.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
