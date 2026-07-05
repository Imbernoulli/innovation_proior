#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int H = inf.readInt();
    int W = inf.readInt();
    vector<string> grid(H);
    for (int i = 0; i < H; i++) grid[i] = inf.readWord();

    // number habitable sectors row-major, build adjacency (open 4-neighbours)
    vector<vector<int>> id(H, vector<int>(W, -1));
    int n = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (grid[i][j] == '.') id[i][j] = ++n;

    int nread = inf.readInt();     // stated count of habitable sectors
    vector<long long> cost(n + 1);
    for (int v = 1; v <= n; v++) cost[v] = inf.readInt();
    vector<int> rad(n + 1);
    for (int v = 1; v <= n; v++) rad[v] = inf.readInt();
    (void)nread;

    // adjacency lists (0-indexed offset by 1 into ids)
    vector<vector<int>> adj(n + 1);
    const int dx[4] = {-1, 1, 0, 0};
    const int dy[4] = {0, 0, -1, 1};
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

    // baseline B = install a relay on every habitable sector
    long long B = 0;
    for (int v = 1; v <= n; v++) B += cost[v];

    // ---- read participant output ----
    int K = ouf.readInt(0, n, "K");
    vector<int> chosen(K);
    vector<char> used(n + 1, 0);
    long long F = 0;
    for (int i = 0; i < K; i++) {
        int v = ouf.readInt(1, n, "sector");
        if (used[v]) quitf(_wa, "sector %d listed twice", v);
        used[v] = 1;
        chosen[i] = v;
        F += cost[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after relay list");

    // ---- coverage: depth-limited BFS from each installed relay ----
    vector<char> warm(n + 1, 0);
    vector<int> dist(n + 1, -1);
    vector<int> stamp(n + 1, -1);
    for (int idx = 0; idx < K; idx++) {
        int s = chosen[idx];
        queue<int> q;
        dist[s] = 0; stamp[s] = idx; q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop();
            warm[x] = 1;
            if (dist[x] == rad[s]) continue;
            for (int y : adj[x]) {
                if (stamp[y] != idx) {
                    stamp[y] = idx;
                    dist[y] = dist[x] + 1;
                    q.push(y);
                }
            }
        }
    }
    for (int v = 1; v <= n; v++)
        if (!warm[v]) quitf(_wa, "habitable sector %d is not warmed by any relay", v);

    if (B <= 0) B = 1;  // safety; B is a sum of positive costs, so never triggers
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
