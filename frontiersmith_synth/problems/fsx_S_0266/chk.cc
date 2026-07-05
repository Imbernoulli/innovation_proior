#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, r;
vector<vector<int>> g;
vector<ll> cost;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    r = inf.readInt();
    g.assign(N + 1, {});
    for (int i = 0; i < M; i++) {
        int u = inf.readInt(), v = inf.readInt();
        g[u].push_back(v);
        g[v].push_back(u);
    }
    cost.assign(N + 1, 0);
    ll B = 0;
    for (int i = 1; i <= N; i++) {
        cost[i] = inf.readInt();
        B += cost[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's relay set ----
    int q = ouf.readInt(0, N, "q");
    vector<char> chosen(N + 1, 0);
    vector<int> relays;
    ll F = 0;
    for (int i = 0; i < q; i++) {
        int idx = ouf.readInt(1, N, "systemIndex");
        if (chosen[idx]) quitf(_wa, "system %d chosen more than once", idx);
        chosen[idx] = 1;
        relays.push_back(idx);
        F += cost[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- feasibility: multi-source BFS bounded by radius r ----
    vector<int> dist(N + 1, INT_MAX);
    deque<int> bfs;
    for (int u : relays) { dist[u] = 0; bfs.push_back(u); }
    while (!bfs.empty()) {
        int u = bfs.front(); bfs.pop_front();
        if (dist[u] >= r) continue;             // do not expand past radius
        for (int v : g[u]) {
            if (dist[u] + 1 < dist[v]) {
                dist[v] = dist[u] + 1;
                bfs.push_back(v);
            }
        }
    }
    for (int j = 1; j <= N; j++) {
        if (dist[j] > r)
            quitf(_wa, "system %d is not illuminated (nearest relay > %d hops)", j, r);
    }

    if (F <= 0) quitf(_wa, "no relays built but coverage claimed");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
