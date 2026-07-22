#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int n, m, r, k;
vector<vector<int>> adj;

// Bootstrap percolation from a seed set: seeds are forced active; any node with
// >= r active neighbours becomes active; run to fixed point. Returns adopters.
long long percolate(const vector<int>& seeds) {
    vector<char> act(n + 1, 0);
    vector<int>  cnt(n + 1, 0);
    vector<int>  q;
    q.reserve(n);
    for (int s : seeds) if (!act[s]) { act[s] = 1; q.push_back(s); }
    for (size_t i = 0; i < q.size(); i++) {
        int u = q[i];
        for (int w : adj[u]) if (!act[w]) {
            if (++cnt[w] >= r) { act[w] = 1; q.push_back(w); }
        }
    }
    long long c = 0;
    for (int v = 1; v <= n; v++) c += act[v];
    return c;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    r = inf.readInt();
    k = inf.readInt();

    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u = inf.readInt();
        int v = inf.readInt();
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ---- internal baseline B: the "scatter" reference (k smallest-degree nodes) ----
    vector<int> order(n);
    for (int v = 1; v <= n; v++) order[v - 1] = v;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (adj[a].size() != adj[b].size()) return adj[a].size() < adj[b].size();
        return a < b;
    });
    int kk = min(k, n);
    vector<int> baseSeeds(order.begin(), order.begin() + kk);
    long long B = percolate(baseSeeds);
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate the participant's seed set ----
    int s = ouf.readInt(0, k, "s");
    vector<char> seen(n + 1, 0);
    vector<int> seeds;
    seeds.reserve(s);
    for (int i = 0; i < s; i++) {
        int v = ouf.readInt(1, n, "seed");
        if (seen[v]) quitf(_wa, "seed %d listed more than once", v);
        seen[v] = 1;
        seeds.push_back(v);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    long long F = percolate(seeds);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((long long)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
