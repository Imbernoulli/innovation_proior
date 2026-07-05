#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<vector<int>> adj(N + 1);
    for (int i = 0; i < M; i++) {
        int u = inf.readInt(), v = inf.readInt();
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<long long> c(N + 1);
    for (int i = 1; i <= N; i++) c[i] = inf.readInt();
    vector<int> r(N + 1);
    for (int i = 1; i <= N; i++) r[i] = inf.readInt();
    int D = inf.readInt();
    vector<int> dem(D);
    for (int i = 0; i < D; i++) dem[i] = inf.readInt();

    // baseline B = install one ladder at every spawning ground
    long long B = 0;
    for (int d : dem) B += c[d];

    // ---- read participant output ----
    int K = ouf.readInt(0, N, "K");
    vector<int> chosen(K);
    vector<char> used(N + 1, 0);
    long long F = 0;
    for (int i = 0; i < K; i++) {
        int v = ouf.readInt(1, N, "node");
        if (used[v]) quitf(_wa, "pool %d listed twice", v);
        used[v] = 1;
        chosen[i] = v;
        F += c[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after ladder list");

    // ---- coverage: depth-limited BFS from each built ladder ----
    vector<char> cov(N + 1, 0);
    vector<int> dist(N + 1, -1);
    vector<int> stamp(N + 1, -1);   // reuse dist array across ladders
    for (int idx = 0; idx < K; idx++) {
        int s = chosen[idx];
        // BFS bounded by r[s]
        queue<int> q;
        dist[s] = 0; stamp[s] = idx; q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop();
            cov[x] = 1;
            if (dist[x] == r[s]) continue;
            for (int y : adj[x]) {
                if (stamp[y] != idx) {
                    stamp[y] = idx;
                    dist[y] = dist[x] + 1;
                    q.push(y);
                }
            }
        }
    }
    for (int d : dem)
        if (!cov[d]) quitf(_wa, "spawning ground %d is not reachable by any ladder", d);

    if (B <= 0) B = 1;  // safety; B is a sum of positive costs so this never triggers
    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
