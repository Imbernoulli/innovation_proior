// TIER: greedy
// Obvious recipe: spatial districting. Spread K seeds evenly across the grid
// and grow territories by multi-source BFS (each block joins whichever seed's
// wavefront reaches it first). This always yields a valid, edge-connected
// partition and roughly recovers broad geographic recipe regions -- but it
// never looks at the recipe/spend data at all, so it can never notice or
// isolate a thin, scattered same-recipe pocket embedded inside a region.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int R, C, K;
    scanf("%d %d %d", &R, &C, &K);
    // read + discard recipe and spend grids (geometry-only heuristic)
    long long total = (long long)R * C * 2;
    for (long long i = 0; i < total; i++) { long long tmp; scanf("%lld", &tmp); }

    // ---- place K seeds spread across a ceil(sqrt(K)) x ceil(sqrt(K)) layout ----
    int axis = (int)ceil(sqrt((double)K));
    vector<pair<int,int>> seeds;
    set<pair<int,int>> used;
    for (int i = 0; i < axis && (int)seeds.size() < K; i++) {
        for (int j = 0; j < axis && (int)seeds.size() < K; j++) {
            int r = (int)((ll)R * (2 * i + 1) / (2 * axis));
            int c = (int)((ll)C * (2 * j + 1) / (2 * axis));
            r = max(0, min(R - 1, r));
            c = max(0, min(C - 1, c));
            if (used.count({r, c})) continue;
            used.insert({r, c});
            seeds.push_back({r, c});
        }
    }
    // fallback: fill remaining seeds via raster scan over unused cells
    for (int r = 0; r < R && (int)seeds.size() < K; r++)
        for (int c = 0; c < C && (int)seeds.size() < K; c++) {
            if (used.count({r, c})) continue;
            used.insert({r, c});
            seeds.push_back({r, c});
        }

    // ---- multi-source BFS territory growth ----
    vector<vector<int>> owner(R, vector<int>(C, 0));
    vector<vector<int>> dist(R, vector<int>(C, -1));
    deque<pair<int,int>> q;
    for (int i = 0; i < (int)seeds.size(); i++) {
        auto [r, c] = seeds[i];
        owner[r][c] = i + 1;
        dist[r][c] = 0;
        q.push_back({r, c});
    }
    static const int dr[4] = {1, -1, 0, 0};
    static const int dc[4] = {0, 0, 1, -1};
    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop_front();
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            if (dist[nr][nc] != -1) continue;
            dist[nr][nc] = dist[r][c] + 1;
            owner[nr][nc] = owner[r][c];
            q.push_back({nr, nc});
        }
    }

    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++) printf("%d%c", owner[r][c], c + 1 == C ? '\n' : ' ');
    }
    return 0;
}
