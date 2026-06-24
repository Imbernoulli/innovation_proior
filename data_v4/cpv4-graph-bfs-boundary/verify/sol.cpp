#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int H, W;
    if (!(cin >> H >> W)) return 0;
    int L, R;
    cin >> L >> R;

    vector<string> g(H);
    for (int i = 0; i < H; i++) cin >> g[i];

    const long long INF = LLONG_MAX / 4;
    vector<vector<long long>> dist(H, vector<long long>(W, INF));
    deque<pair<int,int>> q;

    // Multi-source BFS: every open cell that is a station ('S') starts at distance 0.
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (g[i][j] == 'S') {
                dist[i][j] = 0;
                q.push_back({i, j});
            }

    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};
    while (!q.empty()) {
        auto [x, y] = q.front();
        q.pop_front();
        for (int d = 0; d < 4; d++) {
            int nx = x + dx[d], ny = y + dy[d];
            if (nx < 0 || nx >= H || ny < 0 || ny >= W) continue;
            if (g[nx][ny] == '#') continue;          // blocked: not traversable
            if (dist[nx][ny] > dist[x][y] + 1) {
                dist[nx][ny] = dist[x][y] + 1;
                q.push_back({nx, ny});
            }
        }
    }

    // Count open cells whose nearest-station distance d satisfies L <= d <= R (inclusive band).
    long long count = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            if (g[i][j] == '#') continue;            // blocked cells are never counted
            long long d = dist[i][j];
            if (d == INF) continue;                  // unreachable cells excluded
            if (d >= L && d <= R) count++;           // inclusive on BOTH ends
        }

    cout << count << "\n";
    return 0;
}
