# Dispatching couriers to zones at minimum total cost (assignment by bitmask DP)

## Research question

A small same-day delivery hub has `n` couriers on shift and `n` delivery zones that must be covered
this hour. Exactly one courier is sent to each zone, and each courier covers exactly one zone (it is a
one-to-one assignment). Sending **courier `i` to zone `j`** costs `c[i][j]` — fuel, tolls, and the
detour penalty for that particular courier reaching that particular zone. The dispatcher wants the
assignment that **minimizes the total cost** summed over all `n` courier-zone pairs.

Formally: choose a permutation `p` of `{0, ..., n-1}` (courier `i` goes to zone `p[i]`) minimizing
`sum over i of c[i][p[i]]`. Output that minimum total cost. This is the classic assignment problem,
and with `n` small it is the textbook setting for a **bitmask dynamic program** over the set of
already-covered zones.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 20`). Then `n` lines each with `n` integers; the
  `j`-th integer on line `i` is `c[i][j]` (`0 <= c[i][j] <= 10^9`), whitespace-separated.
- Output (stdout): a single line with the minimum total assignment cost. For `n = 0` the only
  assignment is empty and the answer is `0`.
- Time limit: 2 seconds. Memory: 256 MB.

Example: for

```
3
4 2 8
4 3 7
3 1 6
```

the answer is `12` (courier 0 -> zone 0 costs 4, courier 1 -> zone 2 costs 7, courier 2 -> zone 1
costs 1; `4 + 7 + 1 = 12`, and no permutation does better).

## Background

With `n <= 20`, enumerating all `n!` permutations is hopeless (`20!` is about `2.4*10^18`). The
standard tool is a DP indexed by a **bitmask of which zones are already used**. Process the couriers in
a fixed order `0, 1, 2, ...`; the number of couriers placed so far equals the number of set bits in the
mask, so the mask alone determines *which* courier is being placed next. That makes the state just the
mask:

- `dp[mask]` = minimum cost to assign couriers `0 .. popcount(mask) - 1` to exactly the zones in
  `mask`.

There are `2^n` masks and each tries up to `n` zones for the next courier, so the algorithm is
`O(2^n * n)` time and `O(2^n)` memory — for `n = 20` that is about `2^20 * 20 ≈ 2*10^7` transitions,
comfortable inside the limit. The alternative, the Hungarian algorithm at `O(n^3)`, is overkill for
`n <= 20` and harder to get right; the bitmask DP is the natural fit and what this task targets.

The reason the constraints are drawn the way they are: with `c[i][j]` up to `10^9` and `n` up to `20`,
a full assignment's cost can reach `20 * 10^9 = 2*10^10`. That is roughly **ten times** the maximum a
signed 32-bit integer can hold (`2^31 - 1 ≈ 2.147*10^9`), so the running cost inside the DP must be
accumulated in 64-bit arithmetic. Doing the additions in `int` overflows silently and yields a wrong
answer with no crash and no warning.

## Evaluation settings

Judged on hidden tests covering: tiny cases (`n = 0`, `n = 1`) checked against the obvious answer;
small `n` cross-checked against brute-force permutation search; cost matrices with many equal entries
(so several permutations tie); matrices where the cheapest single edges form an *illegal* assignment
(two couriers wanting the same zone), so a greedy per-courier pick is wrong; and large `n = 20`
matrices with entries near `10^9` engineered so the optimal total exceeds `2^31` — these are the cases
that expose a 32-bit accumulator.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<vector<long long>> c(n, vector<long long>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> c[i][j];

    // TODO: bitmask DP over the set of used zones to compute the minimum-cost
    //       one-to-one assignment of couriers to zones; print the minimum total cost.
    long long answer = 0;

    cout << answer << "\n";
    return 0;
}
```
