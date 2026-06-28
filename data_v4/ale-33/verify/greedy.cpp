// Trivial baseline for ale-33: route nets one-by-one in INPUT ORDER with a BFS
// shortest path over cells not yet claimed by an earlier net (hard blocking,
// NO rip-up, NO negotiation).  This is the obvious greedy: each net grabs a
// shortest free corridor; later nets must squeeze around it.  If any net cannot
// route (its corridor is boxed in), this baseline has no valid output for that
// net and emits a single cell, which the scorer floors to 0 -- exactly the
// "conflicts count as failures" baseline the normalisation refers to.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W, N;
    if (scanf("%d %d %d", &H, &W, &N) != 3) return 0;
    int HW = H * W;
    vector<array<int,4>> P(N);
    for (auto& p : P) scanf("%d %d %d %d", &p[0], &p[1], &p[2], &p[3]);
    auto id = [&](int r, int c){ return r * W + c; };

    vector<int> endptOwner(HW, -1);
    for (int k = 0; k < N; k++) {
        endptOwner[id(P[k][0], P[k][1])] = k;
        endptOwner[id(P[k][2], P[k][3])] = k;
    }
    vector<char> blocked(HW, 0);        // claimed through-cells
    vector<vector<int>> route(N);
    const int DR[4] = {-1,1,0,0}, DC[4] = {0,0,-1,1};

    for (int k = 0; k < N; k++) {
        int s = id(P[k][0], P[k][1]), t = id(P[k][2], P[k][3]);
        vector<int> prevc(HW, -2);
        queue<int> q; q.push(s); prevc[s] = -1;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            if (v == t) break;
            int r = v / W, c = v % W;
            for (int d = 0; d < 4; d++) {
                int nr = r + DR[d], nc = c + DC[d];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
                int u = nr * W + nc;
                if (prevc[u] != -2) continue;
                if (u != t && blocked[u]) continue;
                int ow = endptOwner[u];
                if (ow != -1 && ow != k && u != t) continue;
                prevc[u] = v; q.push(u);
            }
        }
        if (prevc[t] == -2) { route[k].clear(); continue; }   // failed -> empty
        vector<int> path;
        for (int v = t; v != -1; v = prevc[v]) path.push_back(v);
        reverse(path.begin(), path.end());
        for (int v : path) blocked[v] = 1;
        route[k] = path;
    }

    for (int k = 0; k < N; k++) {
        if (route[k].empty()) { printf("1 %d %d\n", P[k][0], P[k][1]); continue; }
        printf("%d", (int)route[k].size());
        for (int v : route[k]) printf(" %d %d", v / W, v % W);
        printf("\n");
    }
    return 0;
}
