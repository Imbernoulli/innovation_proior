#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C;
    if (!(scanf("%d %d", &R, &C) == 2)) return 0;

    vector<string> g(R);
    for (int i = 0; i < R; i++) {
        // read a row token (length C); skip leading whitespace robustly
        char buf[2100];
        if (scanf("%2099s", buf) != 1) { g[i] = string(C, '#'); continue; }
        g[i] = string(buf);
    }

    const int INF = -1; // unvisited marker
    vector<int> dist((long long)R * C, INF);
    deque<int> q; // BFS queue of flattened indices
    // multi-source: every '*' (tower) starts at distance 0
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            if (g[i][j] == '*') {
                int id = i * C + j;
                dist[id] = 0;
                q.push_back(id);
            }
        }
    }

    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};
    while (!q.empty()) {
        int id = q.front(); q.pop_front();
        int r = id / C, c = id % C;
        int d = dist[id];
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            char ch = g[nr][nc];
            if (ch == '#') continue;          // blocked cell does not transmit
            int nid = nr * C + nc;
            if (dist[nid] != INF) continue;   // already activated earlier (BFS = optimal)
            dist[nid] = d + 1;
            q.push_back(nid);
        }
    }

    // Total energy = sum over all OPEN cells ('.' or '*') of their activation time.
    // Unreachable open cells (never flooded) contribute nothing.
    long long total = 0;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            if (g[i][j] == '#') continue;
            int id = i * C + j;
            if (dist[id] == INF) continue;    // unreachable -> excluded
            total += dist[id];
        }
    }

    printf("%lld\n", total);
    return 0;
}
