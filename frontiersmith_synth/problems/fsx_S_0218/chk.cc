#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, r;
vector<ll> cost;
vector<vector<int>> g;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    r = inf.readInt();

    cost.assign(n + 1, 0);
    ll B = 0;
    for (int v = 1; v <= n; v++) {
        cost[v] = inf.readInt();
        B += cost[v];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    g.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        g[u].push_back(v);
        g[v].push_back(u);
    }

    // ---- read & validate participant's regulator set ----
    int p = ouf.readInt(0, n, "p");
    vector<char> chosen(n + 1, 0);
    vector<int> Z;
    Z.reserve(p);
    for (int i = 0; i < p; i++) {
        int idx = ouf.readInt(1, n, "zoneIndex");
        if (chosen[idx]) quitf(_wa, "zone %d installed more than once", idx);
        chosen[idx] = 1;
        Z.push_back(idx);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- verify coverage: depth-limited BFS from each installed regulator ----
    vector<char> covered(n + 1, 0);
    vector<int> dist(n + 1, -1);
    vector<int> touched;
    for (int f : Z) {
        touched.clear();
        // simple BFS to depth r
        queue<int> q;
        dist[f] = 0;
        touched.push_back(f);
        covered[f] = 1;
        q.push(f);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (dist[u] == r) continue;
            for (int w : g[u]) {
                if (dist[w] == -1) {
                    dist[w] = dist[u] + 1;
                    covered[w] = 1;
                    touched.push_back(w);
                    q.push(w);
                }
            }
        }
        for (int x : touched) dist[x] = -1;
    }

    for (int v = 1; v <= n; v++)
        if (!covered[v])
            quitf(_wa, "zone %d is not within radius %d of any installed regulator", v, r);

    ll F = 0;
    for (int f : Z) F += cost[f];
    if (F <= 0) F = 0; // empty set only feasible if n==0, which cannot happen (n>=1)

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
