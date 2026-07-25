# Cheapest toll route across a freight network

## Research question

A national freight operator runs trucks over a one-way toll network of `n` depots (numbered `1..n`)
connected by `m` directed toll roads. Road `i` lets a truck drive **from** depot `u_i` **to** depot
`v_i` for a toll of `w_i` currency units. Tolls are charged per road and simply add up along a route.
A dispatcher must move a shipment from depot `1` (the central hub) to depot `n` (the export port) as
cheaply as possible. Output the minimum total toll of any directed route from `1` to `n`, or `-1` if
no such route exists.

This is single-source single-target shortest path on a directed graph with non-negative edge weights.

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
detour `1 -> 2 -> 4 -> 5` (`= 2600000000`) and `1 -> 2 -> 3 -> 4 -> 5` (`= 2550000000`).

## Evaluation settings

Judged on hidden tests covering: tiny graphs solved by hand; `n = 1` (answer `0`); disconnected
graphs where depot `n` is unreachable (answer `-1`); zero-weight roads; self-loops and parallel
roads; long heavy chains stressing the numeric range of the answer; and large `n = 2*10^5`,
`m = 5*10^5` instances for time.

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

    // TODO: compute the minimum-toll distance from depot 1; print dist[n] (or -1 if unreachable).
    long long answer = -1;

    cout << answer << "\n";
    return 0;
}
```
