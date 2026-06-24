# Broadcast coverage in a relay tree

## Research question

A communications network is laid out as a rooted tree of `n` relay stations, numbered `0..n-1`. The
root is the network gateway; every other station hangs below it. Station `u` can transmit, and its
signal travels *downward only* (toward its own descendants): it reaches a station `v` exactly when `v`
is a **strict descendant** of `u` and the number of edges between them is **at most** `p[u]`, the
power level of `u`. Formally, `u` covers `v` iff `u` is a proper ancestor of `v` and
`depth(v) - depth(u) <= p[u]`, where `depth` counts edges from the root.

A station `v` is **covered** if at least one station on the path strictly above it (any proper
ancestor) can reach it. The gateway (root) has no station above it, so it can never be covered. Count
how many of the `n` stations are covered.

This is a rooted-DFS problem whose answer lives entirely on the boundaries: whether the reach is `<=`
or `<` `p[u]`, and whether a station may cover *itself* (it may not — coverage is strict-descendant
only). Both choices flip the count, so the inclusive/exclusive bookkeeping has to be exact.

## Input / output contract

- Input (stdin): the first token is `n` (`0 <= n <= 2*10^5`). Then `n` lines follow, one per station
  `i = 0..n-1`, each containing two integers `par[i]` and `p[i]`:
  - `par[i]` is the parent of station `i`, or `-1` if `i` is the root. Exactly one station has
    `par[i] = -1`. Every non-root parent satisfies `0 <= par[i] < n` and the edges form a tree.
  - `p[i]` is the power level, `0 <= p[i] <= 10^9`.
- Output (stdout): a single line with the number of covered stations.
- Time limit: 1 second. Memory: 256 MB.

Example: for the 5-station tree below the answer is `3`.

```
5
-1 1
0 0
1 1
2 0
0 0
```

Stations: `0` is the root (power 1); `1` hangs off `0` (power 0); `2` off `1` (power 1); `3` off `2`
(power 0); `4` off `0` (power 0). Depths are `0,1,2,3,1`. Station `1` (depth 1) is reached by the root
(`0+1 = 1 >= 1`); station `3` (depth 3) is reached by station `2` (`2+1 = 3 >= 3`); station `4`
(depth 1) is reached by the root. Station `2` (depth 2) is reached by nobody (`0+1 = 1 < 2`,
`1+0 = 1 < 2`), and the root has no ancestor. Covered set `{1,3,4}`, so the answer is `3`.

## Background

The reach of a station decays with downward distance, so the natural question at each node `v` is:
*does any ancestor still reach this depth?* Two framings are on the table before committing to one:

- **Per-node upward walk.** For each station `v`, climb the chain of proper ancestors and test
  `depth(v) - depth(u) <= p[u]` for each `u`, stopping at the first hit. Correct and obvious, but a
  chain of length `n` makes this `O(n^2)` in the worst case — too slow at `n = 2*10^5`, though it is
  the perfect independent reference for small cases.
- **Single DFS carrying a running reach.** Each ancestor `u` "reaches down to" absolute depth
  `depth(u) + p[u]`. The deepest absolute depth covered by *any* proper ancestor of `v` is the maximum
  of that quantity over the ancestors, and it only ever grows as we descend. So one DFS that threads a
  running maximum `reach` down each root-to-node path answers every node in `O(n)`. The open questions
  are exactly the boundary ones: is the test `depth(v) <= reach` or `<`, and does `reach` at node `v`
  include `v`'s own contribution or only the strict ancestors'?

## Evaluation settings

Judged on hidden tests covering: deep chains (path graphs of length up to `2*10^5`, which stress both
the running-max idea and the recursion-depth limit), wide stars, balanced trees, power levels of `0`
(a station that reaches nobody) and up to `10^9` (a station that reaches its whole subtree), exact
boundary cases where `depth(v) - depth(u) == p[u]`, the single-node tree (`n = 1`, answer `0`), and
`n = 0`.

## Code framework

A single self-contained C++17 program that reads stdin and writes stdout.

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    if (n <= 0) { printf("0\n"); return 0; }

    vector<int> par(n), p(n);
    vector<vector<int>> ch(n);
    int root = -1;
    for (int i = 0; i < n; i++) {
        int pa;
        if (scanf("%d %d", &pa, &p[i]) != 2) return 0; // parent (-1 for root) and power
        par[i] = pa;
        if (pa == -1) root = i;
        else ch[pa].push_back(i);
    }

    // TODO: count stations v for which some PROPER ancestor u satisfies
    //       depth(v) - depth(u) <= p[u], via one DFS carrying a running reach.
    long long covered = 0;

    printf("%lld\n", covered);
    return 0;
}
```
