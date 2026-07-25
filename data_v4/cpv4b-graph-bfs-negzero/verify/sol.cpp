#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int R, C;
    if (!(cin >> R >> C)) return 0;          // empty input -> no grid

    int N = R * C;
    vector<long long> t(N);
    for (int i = 0; i < N; i++) cin >> t[i];

    // Multi-source BFS over the grid.
    //   source   : t <  0   (strictly cold cells inject frost, start at tick 0)
    //   conductor: t <= 0   (cold AND exactly-zero cells let frost pass through)
    //   wall     : t >  0    (warm cells block frost entirely)
    // A zero cell is a conductor but NOT a source: it only frosts if reached.
    // Frost spreads to 4-adjacent conductor cells one tick at a time. The
    // answer is the largest first-frost tick over all frosted cells; if no
    // cell ever frosts (no strictly-cold source), the answer is -1.
    vector<int> dist(N, -1);
    deque<int> q;
    for (int i = 0; i < N; i++) {
        if (t[i] < 0) {                      // < 0 : strictly cold => a source
            dist[i] = 0;
            q.push_back(i);
        }
    }

    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};
    long long answer = -1;                    // -1 when nothing ever frosts

    while (!q.empty()) {
        int cur = q.front(); q.pop_front();
        answer = max(answer, (long long)dist[cur]);
        int r = cur / C, c = cur % C;
        for (int k = 0; k < 4; k++) {
            int nr = r + dr[k], nc = c + dc[k];
            if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
            int nxt = nr * C + nc;
            if (dist[nxt] != -1) continue;    // already frosted
            if (t[nxt] > 0) continue;         // <= 0 conducts; > 0 is a wall
            dist[nxt] = dist[cur] + 1;
            q.push_back(nxt);
        }
    }

    cout << answer << "\n";
    return 0;
}
