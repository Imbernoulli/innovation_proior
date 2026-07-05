// TIER: greedy
// Max-coverage greedy: repeatedly install the relay that warms the most STILL-COLD sectors
// (ignoring cost); tie-break to lower id. One pass, no cost awareness, no pruning.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) return 0;
    vector<string> grid(H);
    for (int i = 0; i < H; i++) { char buf[64]; scanf("%s", buf); grid[i] = buf; }
    int n; scanf("%d", &n);
    vector<int> cost(n + 1), rad(n + 1);
    for (int v = 1; v <= n; v++) scanf("%d", &cost[v]);
    for (int v = 1; v <= n; v++) scanf("%d", &rad[v]);

    // ids + adjacency
    vector<vector<int>> id(H, vector<int>(W, -1));
    int m = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (grid[i][j] == '.') id[i][j] = ++m;
    vector<vector<int>> adj(n + 1);
    const int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            if (id[i][j] < 0) continue;
            for (int d = 0; d < 4; d++) {
                int ni = i + dx[d], nj = j + dy[d];
                if (ni < 0 || ni >= H || nj < 0 || nj >= W) continue;
                if (id[ni][nj] < 0) continue;
                adj[id[i][j]].push_back(id[ni][nj]);
            }
        }

    // precompute coverage ball of each sector (BFS bounded by rad)
    vector<vector<int>> ball(n + 1);
    vector<int> dist(n + 1, -1), stamp(n + 1, -1);
    for (int s = 1; s <= n; s++) {
        queue<int> q; dist[s] = 0; stamp[s] = s; q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop();
            ball[s].push_back(x);
            if (dist[x] == rad[s]) continue;
            for (int y : adj[x]) if (stamp[y] != s) { stamp[y] = s; dist[y] = dist[x] + 1; q.push(y); }
        }
    }

    vector<char> warm(n + 1, 0);
    int coldLeft = n;
    vector<int> chosen;
    while (coldLeft > 0) {
        int best = -1, bestNew = -1;
        for (int s = 1; s <= n; s++) {
            int nw = 0;
            for (int c : ball[s]) if (!warm[c]) nw++;
            if (nw > bestNew) { bestNew = nw; best = s; }
        }
        if (best < 0 || bestNew <= 0) break;   // safety
        chosen.push_back(best);
        for (int c : ball[best]) if (!warm[c]) { warm[c] = 1; coldLeft--; }
    }

    printf("%d\n", (int)chosen.size());
    for (size_t i = 0; i < chosen.size(); i++)
        printf("%d%c", chosen[i], i + 1 == chosen.size() ? '\n' : ' ');
    if (chosen.empty()) printf("\n");
    return 0;
}
