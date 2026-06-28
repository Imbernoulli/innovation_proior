**Problem.** Given an `R x C` grid of `.` (free), `#` (wall), `S` (start), `T` (target), you walk
4-directionally from `S` to `T`. You may break up to `K` walls in total: stepping onto a free cell
costs no break, stepping onto a `#` costs one break and needs remaining budget. Output the minimum
number of walls broken to reach `T`, or `-1` if `T` is unreachable using at most `K` breaks. The
"flip at most one wall" case is `K = 1`. Read `R C K` then the grid from stdin; print one integer.

**Why the obvious grid BFS is wrong.** A plain BFS that stores one distance per cell cannot model
this, because *how many breaks remain* is part of the search state, not a property of a cell. The same
cell reached with budget left versus budget spent leads to different futures, and a flat search that
marks the cell visited on the first arrival discards the better-resourced one. Concretely, on the
`1 x 5` corridor `S#.#T` the answer is `-1` at `K = 1` (two separate walls, one break) but `2` at
`K = 2` (break both); a flat model cannot even represent the difference between "at the middle cell
with one break left" and "with none," which is exactly what distinguishes the two answers.

**Key idea — layered state-product graph + 0-1 BFS.** Make the node `(cell, breaks_used)` with
`breaks_used in [0..K]`: think of `K + 1` stacked copies of the grid. A free move stays in the same
layer; stepping onto a `#` drops one layer (one more break) and is forbidden out of the bottom layer
`k = K`. Every edge now has a natural cost in `{0, 1}`: `0` for a free step, `1` for a wall-break. The
answer is the shortest-path distance from `(S, 0)` to any `(T, k)`. Because all weights are `0` or
`1`, the SOTA engine is **0-1 BFS** with a deque (`O(V + E)`), not Dijkstra (`O(E log V)`): relax a
weight-`0` edge with `push_front`, a weight-`1` edge with `push_back`; the deque stays sorted by
distance so a front-pop is final. With `V = R * C * (K + 1)` up to `~1.1 * 10^7`, the `log` factor
Dijkstra carries is what 0-1 BFS removes — the difference between comfortably passing and timing out.

**Pitfalls.**
1. *Charging free moves.* A `.`/`S`/`T` step is weight `0`; only a `#` step is weight `1`. Charging
   `1` for ordinary moves silently computes *steps* instead of *breaks* (a trace of `S...T,K=1`
   returning `4` instead of `0` exposes it).
2. *Deque push side.* Weight-`1` relaxations must `push_back`; pushing everything to the front breaks
   the "front is final" invariant and can finalize a non-minimal distance.
3. *Early exit on `T`.* The first popped `(T, k)` is provably minimal under a correct deque, but it is
   fragile against the lazy duplicates 0-1 BFS creates; safer and equally fast to run to completion
   and take `min over k of dist[(T, k)]`.
4. *State explosion.* Do not forget the `K + 1` layers in the distance array size; the flattened id
   is `((r * C + c) * (K + 1)) + k`, computed in 64-bit to avoid any intermediate overflow.

**Edge cases.** `K = 0` reduces to plain reachability (`0` or `-1`). A free detour always beats a
break, so a wall-free route yields `0` even when budget is available (0-1 BFS reaches those nodes
first via weight-`0` edges). A target enclosed by a one-thick wall costs `1`; by two-thick costs `2`
(only if `K` allows). Unreachable within budget prints `-1`. `1 x 2` adjacent `S`/`T` prints `0`.

**Complexity.** `O(V + E) = O(R * C * (K + 1))` time and memory; about `44 MB` and well under one
second at the maximum `1000 x 1000`, `K = 10`. A Dijkstra over the identical state graph is the
cross-check oracle.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int R, C, K;
    if (!(cin >> R >> C >> K)) return 0;

    vector<string> g(R);
    for (int i = 0; i < R; i++) cin >> g[i];

    int sr = -1, sc = -1, tr = -1, tc = -1;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (g[i][j] == 'S') { sr = i; sc = j; }
            else if (g[i][j] == 'T') { tr = i; tc = j; }
        }

    // Layered state graph: node = (row, col, breaks_used), breaks_used in [0..K].
    // Edge weight 0  : step onto a free cell ('.', 'S', 'T')  -> breaks unchanged.
    // Edge weight 1  : step onto a wall ('#')                 -> breaks_used + 1 (needs budget).
    // dist(node) = minimum number of walls broken to arrive at that node.
    // Because every edge weight is 0 or 1, 0-1 BFS (deque) computes all distances in O(V+E):
    //   relax with weight 0 -> push_front, weight 1 -> push_back, and the deque stays
    //   sorted by distance. A pop is processed only if it matches the stored dist (lazy skip).
    const int INF = INT_MAX;
    int layer = K + 1;
    auto idx = [&](int r, int c, int k) -> long long {
        return (((long long)r * C + c) * layer + k);
    };
    vector<int> dist((long long)R * C * layer, INF);

    deque<long long> dq;
    dist[idx(sr, sc, 0)] = 0;
    dq.push_back(idx(sr, sc, 0));

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    while (!dq.empty()) {
        long long cur = dq.front();
        dq.pop_front();
        int k = (int)(cur % layer);
        long long rc = cur / layer;
        int c = (int)(rc % C);
        int r = (int)(rc / C);
        int d = dist[cur];

        for (int dir = 0; dir < 4; dir++) {
            int nr = r + dr[dir];
            int nc = c + dc[dir];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            char ch = g[nr][nc];
            int nk = k, w;
            if (ch == '#') {
                if (k == K) continue;       // no budget left to break a wall
                nk = k + 1;
                w = 1;
            } else {
                w = 0;                      // '.', 'S' or 'T' : free move
            }
            int nd = d + w;
            long long nstate = idx(nr, nc, nk);
            if (nd < dist[nstate]) {
                dist[nstate] = nd;
                if (w == 0) dq.push_front(nstate);
                else        dq.push_back(nstate);
            }
        }
    }

    int best = INF;
    for (int k = 0; k <= K; k++)
        best = min(best, dist[idx(tr, tc, k)]);

    cout << (best == INF ? -1 : best) << "\n";
    return 0;
}
```
