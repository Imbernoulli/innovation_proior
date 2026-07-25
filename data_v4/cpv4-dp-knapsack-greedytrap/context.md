# Provisioning a sequencing run under a fixed reagent budget

## Research question

A genomics core has one sequencing run left this quarter and a fixed amount of a shared reagent,
measured in integer **micrograms**, of which exactly `B` are available. There are `n` candidate
assays queued. Assay `i` consumes `c[i]` micrograms of reagent (an all-or-nothing amount: you either
run the assay in full or not at all) and, if run, produces `r[i]` usable data points. The reagent is
the only binding resource and each assay can be run at most once.

Choose a subset `S` of the assays with total reagent cost `sum_{i in S} c[i] <= B` so as to
**maximize** the total yield `sum_{i in S} r[i]`. Output that maximum yield. Running no assays is
allowed, so the answer is always at least `0`.

This is the 0/1 knapsack problem in disguise: indivisible items (assays), a single capacity (the
reagent budget `B`), an additive objective (data points).

## Input / output contract

- Input (stdin):
  - The first line contains two integers `n` and `B` (`0 <= n <= 2000`, `0 <= B <= 2*10^5`).
  - Each of the next `n` lines contains two integers `c[i]` and `r[i]`
    (`1 <= c[i] <= 2*10^5`, `0 <= r[i] <= 10^9`): the reagent cost and the data-point yield of
    assay `i`.
- Output (stdout): a single line with the maximum achievable total yield.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with `B = 10` and assays `(c,r) = (6,8), (5,6), (5,6)`, the answer is `12` — run the two
cost-5 assays. (Grabbing the cost-6 assay first, which is the most "data per microgram", strands 4
micrograms and yields only `8`.)

## Evaluation settings

Judged on hidden tests covering: instances engineered so that efficiency-greedy and raw-yield-greedy
both miss the optimum (the fractional intuition fails on indivisible items); `n = 0` and `B = 0`;
assays whose cost exceeds `B` (never runnable); assays with `r[i] = 0`; exact-fit instances where the
optimal subset uses the budget to the last microgram; and large instances with `n = 2000`,
`B = 2*10^5`, and yields near `10^9`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long B;
    if (!(cin >> n >> B)) return 0;

    vector<long long> c(n), r(n);
    for (int i = 0; i < n; i++) cin >> c[i] >> r[i];

    // TODO: maximize total yield over subsets with total cost <= B (empty subset allowed).
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
