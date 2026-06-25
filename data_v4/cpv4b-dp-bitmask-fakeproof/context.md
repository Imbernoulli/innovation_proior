# Relay Crews: cheapest exact partition of the module set

## Research question

An orbital station has `N` maintenance modules, numbered `0 .. N-1`. A roster of `M` repair
crews is available. Crew `j` can service exactly one fixed set of modules in a single shift —
that set is given as a bitmask `mask[j]` over the `N` modules — and hiring it costs `c[j]`.

You must service **every module exactly once**: choose a collection of crews whose module-sets
are pairwise disjoint and whose union is the whole set `{0, .., N-1}`. (Equivalently: the chosen
crews partition the module set.) A crew may be hired at most once. Among all valid partitions,
output the **minimum total cost**, or `-1` if no collection of crews exactly partitions the
modules.

This is the exact-cover-by-disjoint-sets problem in its `N <= 16` bitmask regime: the kind of
core routine that appears inside team-formation, tiling, and assignment solvers. The interesting
part is not the recurrence but the **cost of evaluating it** — the work is governed by a sum over
submasks whose closed form is a classic bit-counting identity that is very easy to assert wrongly.

## Input / output contract

- Input (stdin):
  - The first line holds two integers `N` and `M` (`0 <= N <= 16`, `0 <= M <= 2^N`).
  - Then `M` lines follow, each with two integers `mask[j]` and `c[j]`
    (`1 <= mask[j] <= 2^N - 1`, `0 <= c[j] <= 10^9`). `mask[j]` is the integer whose set bits
    are exactly the modules crew `j` services (bit `i` set means module `i`).
- Output (stdout): a single line — the minimum total cost of an exact partition, or `-1` if none
  exists. (When `N = 0` the module set is empty, the empty collection partitions it, and the
  answer is `0`.)
- Time limit: 2 seconds. Memory: 256 MB.

Example. `N = 4`, `M = 6` with crews
`(mask=3,c=3) (mask=12,c=4) (mask=1,c=5) (mask=6,c=2) (mask=8,c=5) (mask=15,c=11)`.
The cheapest exact partition is `{0,1}` (mask 3, cost 3) together with `{2,3}` (mask 12, cost 4),
covering all four modules disjointly for total cost `7`. The answer is `7`.

## Background

Two facts frame the design.

- **The recurrence is a subset DP.** Let `best[m]` be the minimum cost to cover exactly the
  module-set `m` using disjoint crews. Then `best[0] = 0`, and `best[m]` is the minimum over a
  nonempty submask `s` of `m` (with a crew servicing exactly `s`) of `cost1[s] + best[m ^ s]`,
  where `cost1[s]` is the cheapest single crew whose mask equals `s`. The full answer is
  `best[2^N - 1]`.

- **The work is a sum over (mask, submask) pairs.** Evaluating `best[m]` for every `m` means, for
  each mask `m`, iterating over its submasks. The total number of such pairs is
  `Σ_m 2^popcount(m)`, and whether this DP fits the time limit depends entirely on the closed form
  of that sum — a quantity it is tempting to estimate from "average popcount" and get badly wrong.

The candidate concern before committing is therefore not *which* recurrence but *how expensive it
is to run* at `N = 16`, which hinges on correctly identifying that submask-iteration cost.

## Evaluation settings

Judged on hidden tests covering: small `N` with random crew rosters (checked against an
independent exact-cover search); the empty station `N = 0`; rosters that cannot tile the modules
(answer `-1`); duplicate masks at different costs (cheapest must win); singletons-only rosters;
and the dense worst case `N = 16` with up to `2^16 - 1` crews, where an algorithm whose true cost
was mis-estimated would time out.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;
    const int FULL = (1 << n) - 1;
    const long long INF = (long long)4e18;

    vector<long long> cost1(1 << n, INF);   // cheapest single crew servicing exactly s
    for (int j = 0; j < m; j++) {
        int mk; long long c;
        cin >> mk >> c;
        if (mk >= 1 && mk <= FULL) cost1[mk] = min(cost1[mk], c);
    }

    // TODO: subset DP best[m] = cheapest exact cover of module-set m; print best[FULL] or -1.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
