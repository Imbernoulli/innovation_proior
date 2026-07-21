#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, L, U;
    if (!(cin >> R >> C >> L >> U)) return 0;
    vector<string> g(R);
    for (auto &row : g) cin >> row;

    const int INF = INT_MAX;
    vector<vector<int>> dist(R, vector<int>(C, INF));
    deque<pair<int,int>> q;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (g[r][c] == '*') {
                dist[r][c] = 0;
                q.push_back({r, c});
            }

    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};
    while (!q.empty()) {
        auto [r, c] = q.front(); q.pop_front();
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            if (g[nr][nc] == '#') continue;            // walls block spread
            if (dist[nr][nc] != INF) continue;
            dist[nr][nc] = dist[r][c] + 1;
            q.push_back({nr, nc});
        }
    }

    long long cnt = 0;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            if (g[r][c] == '#') continue;              // walls are never "damaged plants"
            int d = dist[r][c];
            if (d == INF) continue;                    // unreachable open cell
            if (d >= L && d <= U) cnt++;               // inclusive band [L, U]
        }

    cout << cnt << "\n";
    return 0;
}
