# Richest fuel blend at or above a target mass

## Research question

A rover refuelling depot holds `n` distinct fuel canisters. Canister `i` carries `cal[i]` units of
usable energy and has mass `mass[i]`. The loader must select a **non-empty** subset of canisters whose
**total mass is at least `L`** (a minimum payload the engine needs to fire). Among all such selections,
it wants the **richest blend**: the one maximizing the *energy density*

    density(S) = (sum of cal[i] over i in S) / (sum of mass[i] over i in S).

Output the maximum achievable density as an **exact reduced fraction** `p/q` (with `q > 0` and
`gcd(p, q) = 1`). Note you cannot just grab the single densest canister: it might be too light to meet
the mass floor `L`, so the choice of *which* canisters to combine is genuinely a knapsack-style search.

This is a fractional (ratio) objective sitting on top of a 0/1 subset-sum constraint. The subtlety is
not the search — it is that the answer is a *ratio of two DP-computed sums*, and choosing the best ratio
means comparing fractions whose cross-products are enormous.

## Input / output contract

- Input (stdin): the first line holds two integers `n` and `L`
  (`1 <= n <= 50`, `1 <= L <= sum of all mass[i]`).
  Each of the next `n` lines holds two integers `cal[i]` and `mass[i]`
  (`1 <= cal[i] <= 10^13`, `1 <= mass[i] <= 1000`).
- Output (stdout): a single line `p/q`, the maximum density in lowest terms (`gcd(p, q) = 1`, `q > 0`).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for

    4 5
    100 4
    30 2
    60 3
    12 1

the answer is `160/7`. Canister 4 alone is densest (`12/1`), but its mass `1` is below `L = 5`. The best
selection meeting the floor is canisters 1 and 3: energy `100 + 60 = 160`, mass `4 + 3 = 7`, density
`160/7 ≈ 22.857`, beating e.g. canisters 1 and 4 (`112/5 = 22.4`).

## Background

Two facts shape the design:

- **The feasible total masses are bounded and small.** With `n <= 50` and `mass[i] <= 1000`, the total
  mass of any subset lies in `[0, 50000]`. So we can do a 0/1 *subset-sum DP over total mass*: for each
  reachable mass `W`, keep the maximum total energy `best[W]` achievable by some subset of that exact
  mass. That is an `O(n * sumMass)` table, at most `50 * 50000 = 2.5*10^6` updates.
- **The objective is a ratio, and the numerators are huge.** Total energy can reach
  `50 * 10^13 = 5*10^14`; total mass up to `5*10^4`. To pick the best `best[W]/W` we compare two
  fractions `P1/W1` and `P2/W2` by their cross-products `P1*W2` vs `P2*W1`. Those products can reach
  `5*10^14 * 5*10^4 = 2.5*10^19`, which is **larger than the signed 64-bit ceiling**
  `LLONG_MAX ≈ 9.2*10^18`. So the comparison itself, not the DP, is where correctness is won or lost.

## Evaluation settings

Judged on hidden tests covering: the small worked sample; cases where the globally densest canister is
too light to meet `L` (so the floor actually bites); `L = 1` and `L = sumMass` (whole-set forced);
`n = 1`; ties where several subsets share a density and the reduced form must match; and large
adversarial cases with `cal[i]` near `10^13` engineered so the decisive fraction comparison overflows
signed 64-bit arithmetic. A solution that divides in floating point, or cross-multiplies in `long long`,
fails those last cases.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long L;
    if (!(cin >> n >> L)) return 0;
    vector<long long> cal(n), mass(n);
    long long sumMass = 0;
    for (int i = 0; i < n; i++) { cin >> cal[i] >> mass[i]; sumMass += mass[i]; }

    // TODO: 0/1 subset-sum DP over total mass; among masses W >= L, return the
    // maximum energy/W as a reduced fraction p/q, comparing fractions exactly.
    long long p = 0, q = 1;

    cout << p << "/" << q << "\n";
    return 0;
}
```
