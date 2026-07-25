#include <bits/stdc++.h>
using namespace std;

/*
  Shortest delivery route on a grid where the robot may BLAST through at most
  K boulder cells. Grid chars:
    'S' start, 'T' target, '.' open, '#' boulder.
  Each move to an orthogonally adjacent in-bounds cell costs 1 minute. Moving
  INTO a boulder cell is allowed only if at least one blast charge remains;
  doing so consumes one charge (the boulder is pulverised for that traversal
  but we model charge usage, not permanent removal). 'S' and 'T' are open.

  BFS over augmented state (row, col, used) where used in [0..K] is the number
  of charges already spent. Plain BFS works because every move costs exactly 1
  minute. A state is the position together with how many charges are gone; we
  visit each such state at most once. Output the fewest minutes to reach T, or
  -1 if T is unreachable within the blast budget.
*/

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int R, C, K;
    if (!(cin >> R >> C >> K)) return 0;
    vector<string> g(R);
    for (auto &row : g) cin >> row;

    int sr = -1, sc = -1, tr = -1, tc = -1;
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++) {
            if (g[r][c] == 'S') { sr = r; sc = c; }
            else if (g[r][c] == 'T') { tr = r; tc = c; }
        }

    int layers = K + 1;
    auto id = [&](int r, int c, int k) {
        return (r * C + c) * layers + k;
    };

    vector<int> dist(R * C * layers, -1);

    int start = id(sr, sc, 0);
    dist[start] = 0;
    queue<int> q;
    q.push(start);

    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    int answer = -1;
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        int k = cur % layers;
        int cell = cur / layers;
        int c = cell % C;
        int r = cell / C;
        int d = dist[cur];

        if (r == tr && c == tc) { answer = d; break; }

        for (int dir = 0; dir < 4; dir++) {
            int nr = r + dr[dir], nc = c + dc[dir];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            int nk = k;
            if (g[nr][nc] == '#') {
                if (k + 1 > K) continue;   // out of blast charges
                nk = k + 1;
            }
            int nid = id(nr, nc, nk);
            if (dist[nid] == -1) {
                dist[nid] = d + 1;
                q.push(nid);
            }
        }
    }

    cout << answer << "\n";
    return 0;
}
