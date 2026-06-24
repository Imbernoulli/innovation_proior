# Earliest arrival under station curfews

## Research question

A courier moves through a city modelled as a directed graph of `n` stations and `m` one-way
connections. Connection `(u, v, w)` lets the courier travel from station `u` to station `v` in `w`
minutes. The courier starts **at station `1` at time `0`** and wants to reach station `n` as early as
possible.

Every station `v` has a **curfew time** `c[v]`: the courier is forbidden to *be present* at station
`v` at any moment `t` with `t >= c[v]`. Travel along a connection is continuous, but the only thing
the rules check is the instant the courier is *standing at* a station — so an arrival at station `v`
at time `t` is permitted exactly when `t < c[v]` (strictly before the curfew). Waiting at a station is
allowed but never helps, because arriving somewhere earlier can only keep more options open.

Output the earliest time the courier can legally be present at station `n`, or `-1` if no legal route
exists (including the degenerate case where the courier may not even legally stand at station `1` at
time `0`).

The whole difficulty lives at the boundary `t = c[v]`: arriving *exactly* at the curfew is forbidden,
not allowed. This is a shortest-path problem with a per-node validity window, and whether the
inequality is `<` or `<=` changes the answer.

## Input / output contract

- Input (stdin):
  - The first line has two integers `n` and `m` (`1 <= n <= 2*10^5`, `0 <= m <= 2*10^5`).
  - The second line has `n` integers `c[1], ..., c[n]` (`0 <= c[v] <= 10^9`), the curfew of each
    station.
  - Each of the next `m` lines has three integers `u v w` (`1 <= u, v <= n`, `u != v`,
    `1 <= w <= 10^9`): a one-way connection from `u` to `v` taking `w` minutes. Parallel edges may
    appear; there are no self-loops.
- Output (stdout): a single line with the earliest legal arrival time at station `n`, or `-1` if it
  is unreachable under the curfew rule.
- Time limit: 2 seconds. Memory: 256 MB.

Example: with stations `1..4`, curfews `c = [100, 9, 12, 8]`, and connections
`1->4 (8)`, `1->2 (3)`, `2->4 (4)`, `3->4 (1)`, the answer is `7`. The direct hop `1->4` would land at
time `8`, but `c[4] = 8` and `8 < 8` is false, so that arrival is illegal; the route `1->2->4` arrives
at time `3 + 4 = 7 < 8`, which is legal and earliest.

## Background

Two design questions must be settled before writing code.

- **Does Dijkstra still apply?** Edge weights are positive and the per-node constraint depends only on
  the *arrival time*, never on the path taken to get there. So "earliest legal arrival at `v`" is a
  well-defined label that can only improve, and a smaller label is always at least as useful as a
  larger one (it satisfies strictly more curfews downstream and produces strictly smaller successor
  times). That monotonicity is exactly what Dijkstra needs: pruning a node when we have already
  finalized a smaller legal arrival there is sound.

- **Where is the boundary checked?** The constraint "present at `v` only while `t < c[v]`" must be
  enforced at the moment of *arrival* at `v`, including the start (the courier is present at station
  `1` at time `0`, which requires `0 < c[1]`). Whether the comparison is strict (`<`) or inclusive
  (`<=`) flips the verdict on every arrival that lands exactly on a curfew, and the constraints are
  chosen so such exact landings are common.

## Evaluation settings

Judged on hidden tests covering: graphs where the raw-cheapest route lands exactly on a curfew and a
slightly longer route is the real answer; graphs where every route into station `n` is blocked
(answer `-1`); the start station being illegal at time `0` (`c[1] = 0`); `n = 1` (the courier is
already at the destination at time `0`, answer `0` iff `c[1] > 0`); parallel edges with different
weights; and large instances `n, m = 2*10^5` with weights near `10^9`, so accumulated times exceed the
32-bit range and demand 64-bit arithmetic.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<long long> c(n + 1);
    for (int v = 1; v <= n; v++) cin >> c[v];

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    // TODO: Dijkstra for the earliest legal arrival at station n.
    // An arrival time t at station v is legal only when t < c[v] (strict boundary),
    // including the start: the courier is present at station 1 at time 0.
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
