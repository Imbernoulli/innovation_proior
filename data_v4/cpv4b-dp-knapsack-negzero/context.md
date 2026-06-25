# Drone manifest: load exactly K parcels under a weight cap, signed net profit

## Research question

A delivery drone flies one route tonight. There are `n` candidate parcels in the depot. Parcel `i`
has an integer **weight** `w[i] >= 0` (some "parcels" are paperwork-only manifests that weigh `0`)
and an integer **net profit** `v[i]` that may be **negative, zero, or positive** (a loss-leader you
are contractually obliged to consider can drag the night's profit down yet still has to be carried to
make the count). Operations requires the drone to leave with **exactly `K` parcels** loaded, and the
combined weight of the chosen parcels must **not exceed** the payload cap `C`. Among all ways to load
exactly `K` parcels within the cap, you want to **maximize the total net profit**. If there is **no**
way to load exactly `K` parcels within the cap, the run is cancelled.

This is a two-dimensional (cardinality x weight) 0/1 knapsack. The fixed-count requirement is what
makes it interesting: because you are *forced* to take exactly `K` parcels, the optimum can
legitimately be a **negative** number, and "the optimum is negative" must stay strictly distinct from
"no valid load exists." That distinction is exactly where a careless base case (initialising the
whole table to `0`, or confusing "profit 0" with "unreachable") silently produces a wrong answer.

## Input / output contract

- Input (stdin): the first line has three integers `n`, `K`, `C`
  (`0 <= n <= 200`, `0 <= K <= 10^9`, `0 <= C <= 1000`). Then `n` lines follow, the `i`-th containing
  `w[i]` and `v[i]` (`0 <= w[i] <= 10^9`, `-10^9 <= v[i] <= 10^9`).
- Output (stdout): if some subset of **exactly `K`** parcels has total weight `<= C`, print the
  maximum total net profit over all such subsets (this value may be negative or zero). Otherwise
  print the single token `INFEASIBLE`.
- Time limit: 1 second. Memory: 256 MB.

Example: for `n = 5`, `K = 2`, `C = 7` and parcels `(w,v) = (3,5), (4,-2), (2,0), (5,4), (0,-1)` the
answer is `5`: you must pick exactly two parcels of combined weight at most `7`. The pair `{(3,5),
(2,0)}` has weight `5 <= 7` and profit `5 + 0 = 5`, which beats `{(3,5),(4,-2)}` (weight `7`, profit
`3`) and `{(3,5),(5,4)}` (weight `8`, over the cap). So `5`.

## Background

Two families of approach are on the table before committing to one.

- **Subset enumeration / meet-in-the-middle.** Enumerate every size-`K` subset (or split the parcels
  in half and merge by weight). Exhaustive enumeration is `O(C(n,K))` and obviously correct, but only
  tractable for tiny `n`; meet-in-the-middle reaches `n` near 40 but is fiddly to merge under the
  *maximize-profit-at-exactly-K-parcels-and-weight-at-most-C* objective. Neither scales to `n = 200`.
- **Two-dimensional capacity DP (count x weight).** Build a table indexed by `(k, c)` = number of
  parcels chosen so far and their exact total weight, storing the best profit reachable, and relax it
  parcel by parcel in the 0/1 manner. This is `O(n * K * C)` time and `O(K * C)` memory. The open
  questions are the exact base case (which `(k, c)` are reachable before any parcel is placed), the
  transition order that keeps each parcel usable at most once, and how to encode "this `(k, c)` is
  unreachable" so it never masquerades as a real profit of `0`.

## Evaluation settings

Judged on hidden tests covering: loads with a mix of positive, zero, and negative profits; cases
where every feasible size-`K` load gives a **negative** optimum (must print the negative number, not
`INFEASIBLE`); the empty-load corner `K = 0` (answer `0`, even when every parcel has a negative
profit); `n = 0`; all-negative profit arrays; counts `K` that **exceed `n`** or that cannot be met
within the cap (answer `INFEASIBLE`); many zero-weight parcels (so several can be loaded without
spending any capacity); parcels heavier than `C` (never choosable); and large `n = 200`, `K = 200`,
`C = 1000` with `|v[i]|` near `10^9` (so the accumulated profit can exceed the 32-bit range).

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    long long K, C;
    if (!(cin >> n >> K >> C)) return 0;
    vector<long long> w(n), v(n);
    for (int i = 0; i < n; i++) cin >> w[i] >> v[i];

    // TODO: maximize total net profit over subsets of EXACTLY K parcels with total weight <= C;
    //       print INFEASIBLE if no such subset exists (the optimum itself may be negative or zero).

    return 0;
}
```
