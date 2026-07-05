#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, q, k;
vector<int> targets;
vector<vector<array<ll, 3>>> adj; // per node: (neighbor, weight, edgeIndex)

// Sum of shortest-path distances from s to every target. Returns {sum, allReachable}.
pair<ll, bool> totalLatency(const vector<char>& removed) {
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
    ll sum = 0;
    for (int t : targets) {
        if (dist[t] == LLONG_MAX) return {0, false};
        sum += dist[t];
    }
    return {sum, true};
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);
    n = inf.readInt();
    m = inf.readInt();
    s = inf.readInt();
    q = inf.readInt();
    k = inf.readInt();
    targets.resize(q);
    for (int i = 0; i < q; i++) targets[i] = inf.readInt();
    adj.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readLong();
        adj[u].push_back({(ll)v, w, (ll)i});
        adj[v].push_back({(ll)u, w, (ll)i});
    }

    vector<char> none(m + 1, 0);
    auto [base, baseOk] = totalLatency(none);
    if (!baseOk || base < 1) quitf(_fail, "bad instance: baseline unreachable/nonpositive");

    // participant output
    int r = ouf.readInt(0, k, "r");
    vector<char> removed(m + 1, 0);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "idx");
        if (removed[idx]) quitf(_wa, "duplicate link index %d", idx);
        removed[idx] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    auto [F, feasible] = totalLatency(removed);
    if (!feasible) quitf(_wa, "a science antenna is disconnected from the hub after the cuts");

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, base));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, base, sc / 1000.0);
}
