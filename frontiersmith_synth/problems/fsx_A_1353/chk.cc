// checker for "Sandpile Relief Under a Firing Budget" (fsx_A_1353)
// Reads: graph (n piles + 1 sink=n, m weighted edges), initial chip counts h[],
// criticality weights c[]. Participant reports f[v] = total times pile v fires.
//
// Feasibility: sum f <= K, and f must be REALIZABLE by some legal firing order
// (only fire a currently-unstable pile). We verify this with a queue-driven
// simulation. This is safe (order-independent) because of the abelian /
// least-action property of chip-firing under a fixed per-vertex firing cap:
// any maximal legal-and-under-cap firing process either exhausts the whole
// cap vector or gets permanently stuck, and both the outcome and the final
// chip counts are identical regardless of the order chosen -- so checking
// with ANY single fixed order (here: a FIFO activation queue) is conclusive.
//
// Objective (minimize): R = sum_v c_v * max(0, finalHeight_v - deg_v + 1).
// Baseline B = R for the do-nothing output. Score = B / (B + 9*R).
#include "testlib.h"
#include <vector>
#include <algorithm>
using namespace std;
typedef long long ll;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt(1, 2000, "n");
    int m = inf.readInt(1, 4000, "m");
    ll K  = inf.readLong(0LL, 4000000LL, "K");
    int sink = n;

    vector<vector<pair<int,ll>>> adj(n + 1);
    vector<ll> deg(n, 0);
    for (int i = 0; i < m; i++) {
        int a = inf.readInt(0, n, "a");
        int b = inf.readInt(0, n, "b");
        ll w  = inf.readLong(1LL, 60LL, "w");
        if (a == b) quitf(_fail, "generator produced a self loop (a=%d)", a);
        adj[a].push_back({b, w});
        adj[b].push_back({a, w});
        if (a < n) deg[a] += w;
        if (b < n) deg[b] += w;
    }
    (void)sink;

    vector<ll> h(n), c(n);
    for (int v = 0; v < n; v++) h[v] = inf.readLong(0LL, 300000LL, "h");
    for (int v = 0; v < n; v++) c[v] = inf.readLong(1LL, 25LL, "c");

    for (int v = 0; v < n; v++)
        if (deg[v] < 1) quitf(_fail, "generator bug: pile %d has degree 0", v);

    auto riskSum = [&](vector<ll>& height) -> ll {
        ll s = 0;
        for (int v = 0; v < n; v++) {
            ll e = height[v] - deg[v] + 1;
            if (e > 0) s += c[v] * e;
        }
        return s;
    };

    vector<ll> h0 = h;
    ll B = riskSum(h0);
    if (B < 1) B = 1;

    // ---- participant output: f_0 .. f_{n-1} ----
    vector<ll> f(n);
    ll sumF = 0;
    for (int v = 0; v < n; v++) {
        f[v] = ouf.readLong(0LL, 2000000000LL, "f");
        sumF += f[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");
    if (sumF > K) quitf(_wa, "sum f = %lld exceeds budget K=%lld", sumF, K);

    // ---- realizability simulation (queue-driven, order-independent outcome) ----
    vector<ll> height = h;
    vector<ll> remaining = f;
    ll totalRemaining = sumF;

    vector<char> inQueue(n, 0);
    vector<int> q;
    q.reserve(n);
    auto tryEnqueue = [&](int v) {
        if (!inQueue[v] && remaining[v] > 0 && height[v] >= deg[v]) {
            inQueue[v] = 1;
            q.push_back(v);
        }
    };
    for (int v = 0; v < n; v++) tryEnqueue(v);

    size_t qi = 0;
    while (qi < q.size()) {
        int v = q[qi++];
        inQueue[v] = 0;
        while (remaining[v] > 0 && height[v] >= deg[v]) {
            // batch: fire v repeatedly as long as it stays legal, without needing
            // to touch neighbors in between (v's own future legality depends
            // only on v's own height, which only v's own firings change).
            ll times = (height[v] - deg[v]) / deg[v] + 1;
            if (times > remaining[v]) times = remaining[v];
            height[v] -= times * deg[v];
            remaining[v] -= times;
            totalRemaining -= times;
            for (auto& pr : adj[v]) {
                int u = pr.first; ll w = pr.second;
                if (u == sink) continue;
                height[u] += w * times;
                tryEnqueue(u);
            }
        }
    }

    if (totalRemaining > 0)
        quitf(_wa, "firing vector not realizable: %lld claimed firings can never legally happen", totalRemaining);

    ll R = riskSum(height);
    double denom = (double)B + 9.0 * (double)R;
    double ratio = (denom > 0.0) ? (double)B / denom : 0.0;
    if (ratio < 0.0) ratio = 0.0;
    if (ratio > 1.0) ratio = 1.0;

    quitp(ratio, "OK R=%lld B=%lld Ratio: %.6f", R, B, ratio);
}
