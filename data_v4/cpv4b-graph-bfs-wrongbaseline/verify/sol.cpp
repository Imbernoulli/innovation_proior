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
