#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for VLIW latency-constrained bundle scheduling (minimization).
// Validates a participant issue-cycle assignment s_1..s_n against:
//   * dependence latencies: for every edge u->v, s_v >= s_u + L_u
//   * issue width W: at most W ops share a cycle
//   * per-kind unit caps cap_t: at most cap_t kind-t ops share a cycle
//   * read-port budget P: sum of rp_{t_i} over a cycle <= P
// Objective F = 1 + max s_i (makespan).  Internal baseline B = makespan of the
// one-op-per-cycle program-order (topological) schedule -- always feasible, positive.
//   ratio = min(1, (B / max(1,F)) / 10).

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int W = inf.readInt();
    int T = inf.readInt();
    int P = inf.readInt();

    vector<int> cap(T + 1), rp(T + 1);
    for (int t = 1; t <= T; t++) cap[t] = inf.readInt();
    for (int t = 1; t <= T; t++) rp[t] = inf.readInt();

    vector<int> type(n + 1), L(n + 1);
    for (int i = 1; i <= n; i++) {
        type[i] = inf.readInt();
        L[i] = inf.readInt();
    }

    vector<int> eu(m + 1), ev(m + 1);
    vector<vector<int>> preds(n + 1);
    for (int j = 1; j <= m; j++) {
        int u = inf.readInt();
        int v = inf.readInt();
        eu[j] = u; ev[j] = v;
        preds[v].push_back(u);
    }

    // ---- read participant schedule ----
    vector<long long> s(n + 1);
    for (int i = 1; i <= n; i++)
        s[i] = ouf.readLong(0LL, 100000000LL, format("s_%d", i).c_str());
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the schedule");

    // ---- resource feasibility (per-cycle counters via hash maps) ----
    unordered_map<long long, int> cntCycle;
    unordered_map<long long, long long> portCycle;
    unordered_map<long long, int> cntType;   // key = cycle*(T+1) + kind
    cntCycle.reserve(2 * n + 16);
    portCycle.reserve(2 * n + 16);
    cntType.reserve(2 * n + 16);

    long long maxS = 0;
    for (int i = 1; i <= n; i++) {
        long long c = s[i];
        int cc = ++cntCycle[c];
        if (cc > W)
            quitf(_wa, "cycle %lld issues %d ops > issue width W=%d", c, cc, W);
        long long kt = c * (long long)(T + 1) + type[i];
        int ct = ++cntType[kt];
        if (ct > cap[type[i]])
            quitf(_wa, "cycle %lld issues %d kind-%d ops > cap_%d=%d",
                  c, ct, type[i], type[i], cap[type[i]]);
        long long pv = (portCycle[c] += rp[type[i]]);
        if (pv > P)
            quitf(_wa, "cycle %lld read-port demand %lld > budget P=%d", c, pv, P);
        if (c > maxS) maxS = c;
    }

    // ---- dependence latencies ----
    for (int j = 1; j <= m; j++) {
        int u = eu[j], v = ev[j];
        if (s[v] < s[u] + (long long)L[u])
            quitf(_wa, "dependence %d->%d violated: s_%d=%lld < s_%d+L_%d=%lld",
                  u, v, v, s[v], u, u, s[u] + (long long)L[u]);
    }

    long long F = maxS + 1;

    // ---- baseline B: one op per cycle in program order ----
    vector<long long> bs(n + 1, 0);
    long long cur = -1;
    for (int i = 1; i <= n; i++) {
        long long e = 0;
        for (int u : preds[i]) e = max(e, bs[u] + (long long)L[u]);
        long long si = max(e, cur + 1);
        bs[i] = si;
        cur = si;
    }
    long long B = cur + 1;
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
