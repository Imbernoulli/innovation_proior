// TIER: strong
// Insight: the web-safety constraint defines a geometry-induced poset over the
// job cells -- "cut inland before coast." A multi-source BFS from the frame
// (over the ORIGINAL, unpunched sheet) gives every job cell a depth = its
// distance to the nearest clamp. Punching in DECREASING depth order is a
// linear extension of that poset: the deepest (most inland) cells -- e.g. a
// room -- are always punched before the shallower corridor cells that are
// their only remaining link to the frame, so a corridor is only ever punched
// once nothing behind it still depends on it. Within each equal-depth layer
// (where no such dependency exists) we are free to reorder for turret cost,
// so we greedily grab whichever remaining same-depth cell is closest, on the
// tool dial, to the turret's current position.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, N, T;
    scanf("%d %d %d %d", &R, &C, &N, &T);
    vector<string> grid(R);
    for (int r = 0; r < R; r++) { char buf[64]; scanf("%s", buf); grid[r] = buf; }
    vector<int> jr(N), jc(N), jt(N);
    for (int i = 0; i < N; i++) scanf("%d %d %d", &jr[i], &jc[i], &jt[i]);

    // multi-source BFS from every '#' frame cell, over present cells ('#' + '.').
    vector<vector<int>> dist(R, vector<int>(C, -1));
    queue<pair<int,int>> q;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (grid[r][c] == '#') { dist[r][c] = 0; q.push({r, c}); }
    const int dr[4] = {1, -1, 0, 0}, dc[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop();
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            if (grid[nr][nc] == 'x' || dist[nr][nc] != -1) continue;
            dist[nr][nc] = dist[r][c] + 1;
            q.push({nr, nc});
        }
    }

    vector<int> depthOf(N);
    int maxDepth = 0;
    for (int i = 0; i < N; i++) {
        int d = dist[jr[i]][jc[i]];
        if (d < 0) d = 0; // defensive: input is guaranteed frame-connected, but never index by -1
        depthOf[i] = d;
        maxDepth = max(maxDepth, depthOf[i]);
    }

    vector<vector<int>> layer(maxDepth + 1);
    for (int i = 0; i < N; i++) layer[depthOf[i]].push_back(i);

    vector<int> order;
    order.reserve(N);
    int curTool = 1; // home / parked position
    for (int d = maxDepth; d >= 0; d--) {
        vector<int>& rem = layer[d];
        vector<char> taken(rem.size(), 0);
        for (size_t step = 0; step < rem.size(); step++) {
            int best = -1, bestDist = INT_MAX;
            for (size_t k = 0; k < rem.size(); k++) {
                if (taken[k]) continue;
                int dd = abs(jt[rem[k]] - curTool);
                dd = min(dd, T - dd);
                if (dd < bestDist) { bestDist = dd; best = (int)k; }
            }
            taken[best] = 1;
            order.push_back(rem[best]);
            curTool = jt[rem[best]];
        }
    }

    for (int i : order) printf("%d\n", i + 1);
    return 0;
}
