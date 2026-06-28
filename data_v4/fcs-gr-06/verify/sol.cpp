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
