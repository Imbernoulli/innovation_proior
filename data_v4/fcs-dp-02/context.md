# Irrigation pumps along a straight canal

## Research question

A long straight irrigation canal runs past `n` water wells dug by farmers. Well `t`
sits at integer position `x[t]` measured in metres from the canal head, and the
positions are given **sorted** in non-decreasing order. The water authority will
install exactly `p` electric pumps somewhere along the canal (a pump may be placed
at any real coordinate, and several pumps may share a coordinate). Once the pumps
are fixed, every well is served by the **nearest** pump, and the daily effort of
well `t` is the distance from `x[t]` to that pump.

Choose pump positions to **minimise the total served distance** summed over all
wells, and output that minimum.

This is the one-dimensional `p`-facility median (the "post office on a line" family).
It is the kind of clustering-on-a-line subproblem that appears inside districting,
warehouse siting, and time-series segmentation, so getting both the geometry (where a
single pump should sit for a block of wells) and the partition DP exactly right —
fast enough for `n` in the thousands — is the whole task.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `p`
  (`1 <= n <= 8000`, `1 <= p <= n`). The second line holds the `n` well positions
  `x[0], x[1], ..., x[n-1]` (`0 <= x[t] <= 10^9`), whitespace-separated and given in
  non-decreasing order.
- Output (stdout): a single line with the minimum achievable total served distance.
- Time limit: 2 seconds. Memory: 256 MB.

Example:

```
6 2
1 2 3 100 101 102
```

The wells split into the cluster `{1,2,3}` (one pump at 2, effort `1+0+1 = 2`) and
the cluster `{100,101,102}` (one pump at 101, effort `1+0+1 = 2`), for a total of
`4`. No other placement of two pumps does better, so the answer is `4`.

## Background

A pump serving a *contiguous* block of wells should sit at that block's **median**:
on a line the sum of absolute deviations is minimised at the median, and the median
cost of a sorted block is computable in `O(1)` from prefix sums. Because an optimal
assignment of wells to nearest pumps is always a partition of the sorted wells into
contiguous blocks (sorting the pumps, the served sets are intervals), the problem
reduces to: **partition the sorted wells into `p` contiguous blocks to minimise the
sum of per-block median costs.**

That partition is a classic layered dynamic program. Let `C(j, i)` be the median cost
of the block of wells `[j, i]`. With `dp_k[i]` = minimum cost to cover the first `i`
wells using `k` pumps, the recurrence is

```
dp_k[i] = min over j of  dp_{k-1}[j] + C(j, i-1).
```

Two routes to evaluating it are on the table before committing:

- **Plain `O(p n^2)` DP.** For every `(k, i)` scan all split points `j`. Simple and
  obviously correct, but `p` and `n` can both be `8000`, so `p n^2` is far past a
  two-second budget.
- **Faster transition via structure of `C`.** The block-cost matrix `C` satisfies the
  quadrangle (Monge) inequality. Whether that structure lets the inner `min` be
  evaluated in amortised sub-linear time — and exactly which acceleration applies — is
  the open question this task turns on.

## Evaluation settings

Judged on hidden tests covering: `p = 1` (one block, the bare median cost of the whole
array, where the running distance approaches `8000 * 10^9` and overflows 32 bits);
`p = n` (a pump per well, cost `0`); many duplicate / equal positions; clustered
inputs where the optimal split points are far from evenly spaced; and full-size
`n = p = 8000` adversarial cases that punish any `O(p n^2)` implementation.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, p;
    if (!(cin >> n >> p)) return 0;
    vector<ll> x(n);
    for (auto &v : x) cin >> v;
    sort(x.begin(), x.end());

    // TODO: partition the sorted wells into p contiguous blocks minimising the
    // sum of per-block median costs; print the minimum total served distance.
    ll answer = 0;

    cout << answer << "\n";
    return 0;
}
```
