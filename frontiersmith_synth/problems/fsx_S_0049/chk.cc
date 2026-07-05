#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll B;
vector<ll> ec;
vector<vector<array<ll, 3>>> adj; // per node: (neighbor, weight, edgeIndex)

ll dijkstra(const vector<char>& removed) {
    vector<ll> dist(n + 1, LLONG_MAX);
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : adj[u]) {
            int v = (int)e[0];
            ll w = e[1];
            int idx = (int)e[2];
            if (removed[idx]) continue;
            if (d + w < dist[v]) { dist[v] = d + w; pq.push({dist[v], v}); }
        }
    }
    return dist[t];
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);
    n = inf.readInt();
    m = inf.readInt();
    s = inf.readInt();
    t = inf.readInt();
    B = inf.readLong();
    ec.assign(m + 1, 0);
    adj.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readLong(), c = inf.readLong();
        ec[i] = c;
        adj[u].push_back({(ll)v, w, (ll)i});
        adj[v].push_back({(ll)u, w, (ll)i});
    }

    vector<char> none(m + 1, 0);
    ll base = dijkstra(none); // do-nothing baseline
    if (base == LLONG_MAX || base < 1) quitf(_fail, "bad instance: baseline unreachable/nonpositive");

    // participant output
    int r = ouf.readInt(0, m, "r");
    vector<char> removed(m + 1, 0);
    ll costsum = 0;
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "idx");
        if (removed[idx]) quitf(_wa, "duplicate road index %d", idx);
        removed[idx] = 1;
        costsum += ec[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");
    if (costsum > B) quitf(_wa, "closure effort %lld exceeds budget %lld", costsum, B);

    ll F = dijkstra(removed);
    if (F == LLONG_MAX) quitf(_wa, "s and t disconnected after closures");

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, base));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, base, sc / 1000.0);
}
