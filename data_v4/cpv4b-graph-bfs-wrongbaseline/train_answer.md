**Problem.** On an `n x m` grid of heights `h[i][j]`, a drone goes from `(0,0)` to `(n-1,m-1)` by orthogonal moves. Stepping to a strictly higher neighbour (`h[dest] > h[cur]`) is a *boost* costing `1` energy; any other step (level or downhill) is a *glide* costing `0`. Read the grid from stdin, print the minimum total energy. `n, m <= 1000`, heights up to `10^9`.

**Why the obvious plain BFS is wrong.** Plain BFS is the reflexive "shortest path on a grid" tool, but it assumes every edge has weight `1`, so it returns the minimum **number of moves**, not the minimum **energy**. On the `3 x 3` sample

```
1 2 2
3 2 1
3 3 1
```

plain BFS returns `4` (the Manhattan move-distance), while the true energy is `1`: one boost `(0,0)->(1,0)` for `1->3`, then three free glides `3->3->3->1` down to `(2,2)`. A second case `2x5` `[[1,0,1,1,1],[1,0,0,1,0]]` gives BFS `5` against truth `1`. Free glides are charged as moves, so the objective is wrong. Plain BFS is discarded.

**Key idea — 0-1 BFS.** The edges are weighted `0` (glide) or `1` (boost), and the weight is *directional* (up costs 1, the reverse down costs 0), so this is a directed shortest-path problem with weights in `{0,1}`. Use 0-1 BFS: a deque-based Dijkstra specialization. Keep `dist[][]` initialized to `+inf`, set `dist[0][0]=0`, push `(0,0)`. Pop from the front; for each neighbour with edge weight `w = (h[nr][nc] > h[r][c]) ? 1 : 0`, relax only on strict improvement and push to the **front** if `w == 0`, the **back** if `w == 1`. The deque stays sorted within two consecutive distance values, so the front is always a global minimum — `O(nm)` total. The answer is `dist[n-1][m-1]`.

**Pitfalls.**
1. *Objective mismatch.* Plain BFS minimizes move count; you need minimum energy. They differ whenever a free glide path is longer in moves than a boost-heavy path. This is the whole point of the problem.
2. *Boost direction.* The weight-1 edge is moving to a strictly *higher* cell: `h[nr][nc] > h[r][c]`. Writing `h[r][c] > h[nr][nc]` inverts it (charging for glides, free-riding boosts). Trace one move to confirm: `1 -> 3` must give `w = 1`.
3. *Stale deque entries.* A vertex can be pushed more than once before it is finalized. Do **not** rely on a fragile `done`-on-first-pop flag; instead read `d = dist[r][c]` when you pop and guard every relaxation with `if (d + w < dist[nr][nc])`. Then a stale pop only re-attempts relaxations from an already-final distance and the `<` guard rejects all of them — harmless.
4. *Front/back discipline.* Pushing a weight-0 relaxation to the back (or weight-1 to the front) breaks the sorted-deque invariant and yields wrong answers. Weight 0 -> front, weight 1 -> back.

**Edge cases.** `1 x 1`: answer `0` (start is the goal). Flat grid: every move is a glide, answer `0`. Strictly increasing `1 x m` corridor: every step a boost, answer `m-1`. `n x 1` columns: handled by the four-direction loop with two off-grid moves skipped. Heights up to `10^9` are only compared with `>`, never summed, so no overflow; distances are at most `nm <= 10^6`.

**Complexity.** `O(nm)` time and `O(nm)` space — each cell finalized once, each of its four edges relaxed a constant number of times. Comfortable for `10^6` cells within 1 second.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    if (!(cin >> n >> m)) return 0;
    vector<vector<int>> h(n, vector<int>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            cin >> h[i][j];

    // 0-1 BFS: gliding to a cell with height <= current costs 0, boosting up costs 1.
    const long long INF = LLONG_MAX;
    vector<vector<long long>> dist(n, vector<long long>(m, INF));
    deque<pair<int,int>> dq;
    dist[0][0] = 0;
    dq.push_back({0, 0});
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    while (!dq.empty()) {
        auto [r, c] = dq.front();
        dq.pop_front();
        long long d = dist[r][c];
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= n || nc < 0 || nc >= m) continue;
            int w = (h[nr][nc] > h[r][c]) ? 1 : 0; // boost up = 1, glide level/down = 0
            if (d + w < dist[nr][nc]) {
                dist[nr][nc] = d + w;
                if (w == 0) dq.push_front({nr, nc});
                else        dq.push_back({nr, nc});
            }
        }
    }

    cout << dist[n-1][m-1] << "\n";
    return 0;
}
```
