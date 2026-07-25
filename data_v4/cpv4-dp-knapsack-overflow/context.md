# Loading the long-haul van (0/1 knapsack with large payoffs)

## Research question

A courier is loading one van for an overnight run. There are `n` parcels in the depot. Parcel `i`
occupies `w[i]` units of volume and, if delivered, earns a payout of `v[i]` cents. The van holds a
total volume of `W`. Each parcel is either loaded whole or left behind — parcels cannot be split, and
each exists in a single copy. Choose a subset of parcels whose total volume does not exceed `W` so
that the **total payout is maximized**, and output that maximum payout (in cents).

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `W` (`0 <= n <= 1000`,
  `0 <= W <= 10^5`). Then follow `n` lines; line `i` holds two integers `w[i]` and `v[i]`
  (`0 <= w[i] <= 10^5`, `0 <= v[i] <= 10^9`).
- Output (stdout): a single line with the maximum achievable total payout.
- Time limit: 1 second. Memory: 256 MB.

Example: for `W = 10` and the four parcels `(w, v) = (3, 1000000000), (4, 1500000000),
(5, 1200000000), (2, 800000000)`, the answer is `3300000000` — load parcels 1, 2, and 4 (volumes
`3 + 4 + 2 = 9 <= 10`, payouts `1000000000 + 1500000000 + 800000000`).

## Evaluation settings

Judged on hidden tests covering: all parcels fitting, none fitting (every `w[i] > W`), zero-volume
parcels (which should always be loaded if `v[i] > 0`), `W = 0`, `n = 0`, ties in value density, and
large instances with `n = 1000`, `W = 10^5`, and payouts near `10^9`.

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
