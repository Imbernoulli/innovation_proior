# Cheapest toll route across a freight network

## Research question

A national freight operator runs trucks over a one-way toll network of `n` depots (numbered `1..n`)
connected by `m` directed toll roads. Road `i` lets a truck drive **from** depot `u_i` **to** depot
`v_i` for a toll of `w_i` currency units. Tolls are charged per road and simply add up along a route.
A dispatcher must move a shipment from depot `1` (the central hub) to depot `n` (the export port) as
cheaply as possible. Output the minimum total toll of any directed route from `1` to `n`, or `-1` if
no such route exists.

This is single-source single-target shortest path on a directed graph with non-negative edge weights.
What makes it more than a textbook drill is the money scale: individual tolls are already near `10^9`,
and a cheapest route can legitimately chain many of them, so the **answer itself routinely exceeds the
range of a 32-bit signed integer**. Getting the distances into the right integer type — and proving the
unreachable and trivial corners — is the whole game.

## Input / output contract

- Input (stdin): the first line has two integers `n` and `m`
  (`1 <= n <= 2*10^5`, `0 <= m <= 5*10^5`).
  Each of the next `m` lines has three integers `u_i v_i w_i`
  (`1 <= u_i, v_i <= n`, `0 <= w_i <= 10^9`) describing one directed toll road `u_i -> v_i` of toll
  `w_i`. Self-loops (`u_i = v_i`) and several parallel roads between the same pair may appear.
- Output (stdout): a single line with the minimum total toll of a directed route from depot `1` to
  depot `n`, or `-1` if depot `n` is unreachable from depot `1`. If `n = 1` the answer is `0` (the
  shipment is already at the port).
- Time limit: 2 seconds. Memory: 256 MB.

Example: for the network

```
5 6
1 2 800000000
2 3 800000000
3 5 800000000
2 4 900000000
4 5 900000000
3 4 50000000
```

the answer is `2400000000` — the route `1 -> 2 -> 3 -> 5` costs `800000000 * 3`, which beats the
detour `1 -> 2 -> 4 -> 5` (`= 2600000000`) and `1 -> 2 -> 3 -> 4 -> 5` (`= 2550000000`). Note the
optimum `2400000000` is already larger than the maximum 32-bit signed integer `2147483647`.

## Background

With non-negative weights, the canonical tool is **Dijkstra's algorithm**: keep a tentative distance
`dist[v]` for every vertex, repeatedly extract the unsettled vertex of smallest tentative distance,
and relax its outgoing edges. With a binary heap this runs in `O((n + m) log n)`, comfortably inside
the limits for `n <= 2*10^5`, `m <= 5*10^5`. Two design questions are open before committing:

- **What integer type holds a distance?** A route can use up to `n - 1` roads, each costing up to
  `10^9`, so a distance can reach roughly `2*10^14`. That overflows a 32-bit `int` (cap ~`2.1*10^9`);
  the heap keys, the `dist[]` array, and every relaxation `dist[u] + w` must be 64-bit.
- **How is "unreachable" represented and tested?** A sentinel `INF` must be large enough never to be
  beaten by a real distance, yet `dist[u] + w` must never be evaluated for a vertex still at `INF`
  (Dijkstra only relaxes from popped, finite-distance vertices, so this is automatic — but it must be
  argued, not assumed).

A greedy "always take the cheapest outgoing road" walk is *not* on the table: it can wander into a
dead end or loop and gives no optimality guarantee. Dijkstra's settled-vertex invariant is what makes
the answer provably minimal.

## Evaluation settings

Judged on hidden tests covering: tiny graphs solved by hand; `n = 1` (answer `0`); disconnected
graphs where depot `n` is unreachable (answer `-1`); zero-weight roads; self-loops and parallel
roads; long heavy chains where the optimum exceeds `2^31 - 1` (so a 32-bit accumulator silently
wraps to a negative answer); and large `n = 2*10^5`, `m = 5*10^5` instances for time.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<pair<int,int>>> adj(n + 1); // (neighbor, weight)
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    // TODO: run Dijkstra from depot 1; print dist[n] (or -1 if unreachable).
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
