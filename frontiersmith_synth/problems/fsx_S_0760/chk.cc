#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- Refreeze Window: Convoy Wave Scheduling (minimization).
//
// Input:  N M K
//         M lines: u v L C r f     (edge i = i-th of these lines, 1-indexed)
//         K lines: o d ready
//
// Output: K lines, one per convoy in input order:
//         t P e_1 e_2 ... e_P
//         t = start time (>= ready), P = number of edges in the walk (1<=P<=MAXP),
//         e_1..e_P a sequence of edge ids forming a walk from o to d.
//
// Feasibility: bounded reads, chain-connectivity from o to d, seekEof.
//
// Objective: every edge use is a scheduled EVENT (time, edge). All events (over ALL
// convoys) are processed in nondecreasing global time order. An edge's thickness at the
// moment of use is C (if never broken before) or min(C, r * elapsed) where elapsed = time
// since it was last used (elapsed>=0 by construction). Using it resets its "last broken"
// time to now. Cost of an event = f + thickness^1.5 (rounded to nearest integer). F = sum
// of all event costs.
//
// Baseline B: the checker's own BFS fewest-hop path per convoy (ties broken by smallest
// edge id), costed as if every edge it touches were fully virgin (f + C^1.5) -- i.e. the
// naive "just get there, ignore the ice" construction. B is always positive.
//
// Score: sc = min(1000, 100*B/max(1,F));  print "Ratio: sc/1000".

static const int MAXP = 200;
static const long long TMAX = 2000000;

static inline long long r_edge_cost(long long C, long long f) {
    double th = (double)C;
    return f + (long long)llround(pow(th, 1.5));
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt(2, 100000, "N");
    int M = inf.readInt(1, 100000, "M");
    int K = inf.readInt(1, 100000, "K");

    vector<int> eu(M + 1), ev(M + 1), eL(M + 1), eC(M + 1), er(M + 1), ef(M + 1);
    vector<vector<pair<int,int>>> adj(N + 1); // node -> (neighbor, edgeId), in ascending edgeId order
    for (int i = 1; i <= M; i++) {
        eu[i] = inf.readInt(1, N, "u");
        ev[i] = inf.readInt(1, N, "v");
        eL[i] = inf.readInt(1, 1000000, "L");
        eC[i] = inf.readInt(1, 1000000, "C");
        er[i] = inf.readInt(1, 1000000, "r");
        ef[i] = inf.readInt(0, 1000000, "f");
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }

    vector<int> co(K + 1), cd(K + 1), cready(K + 1);
    for (int i = 1; i <= K; i++) {
        co[i] = inf.readInt(1, N, "o");
        cd[i] = inf.readInt(1, N, "d");
        cready[i] = inf.readInt(0, (int)TMAX, "ready");
    }

    // ---- participant output: validate + collect events ----
    struct Event { long long time; int seq; int edgeId; };
    vector<Event> events;
    events.reserve((size_t)K * 8);
    int globalSeq = 0;

    for (int i = 1; i <= K; i++) {
        long long t = ouf.readLong(0, TMAX, "t");
        if (t < cready[i])
            quitf(_wa, "convoy %d starts at %lld before its ready time %d", i, t, cready[i]);
        int P = ouf.readInt(1, MAXP, "P");
        int cur = co[i];
        long long tt = t;
        for (int k = 0; k < P; k++) {
            int e = ouf.readInt(1, M, "edge");
            int other;
            if (eu[e] == cur) other = ev[e];
            else if (ev[e] == cur) other = eu[e];
            else { quitf(_wa, "convoy %d step %d: edge %d is not incident to current node %d", i, k, e, cur); return 0; }
            events.push_back({tt, globalSeq++, e});
            cur = other;
            tt += eL[e];
        }
        if (cur != cd[i])
            quitf(_wa, "convoy %d ends at node %d, expected destination %d", i, cur, cd[i]);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the last convoy's plan");

    // ---- simulate in global chronological order ----
    sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        if (a.time != b.time) return a.time < b.time;
        return a.seq < b.seq;
    });

    vector<char> brokenBefore(M + 1, 0);
    vector<long long> lastBreak(M + 1, 0);
    long long F = 0;
    for (auto& ev : events) {
        int e = ev.edgeId;
        double th;
        if (!brokenBefore[e]) {
            th = (double)eC[e];
        } else {
            long long elapsed = ev.time - lastBreak[e];
            if (elapsed < 0) elapsed = 0;
            th = min((double)eC[e], (double)er[e] * (double)elapsed);
        }
        long long cost = (long long)ef[e] + (long long)llround(pow(th, 1.5));
        F += cost;
        brokenBefore[e] = 1;
        lastBreak[e] = ev.time;
    }
    if (F < 0) F = 0; // defensive; costs are all nonnegative so this cannot actually trigger

    // ---- baseline B: BFS fewest-hop path per convoy, costed at virgin thickness ----
    long long B = 0;
    vector<int> distHop(N + 1), parentEdge(N + 1), parentNode(N + 1);
    for (int i = 1; i <= K; i++) {
        fill(distHop.begin(), distHop.end(), -1);
        queue<int> q;
        distHop[co[i]] = 0;
        q.push(co[i]);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (u == cd[i]) break;
            for (auto& pr : adj[u]) {
                int v = pr.first, eid = pr.second;
                if (distHop[v] == -1) {
                    distHop[v] = distHop[u] + 1;
                    parentEdge[v] = eid;
                    parentNode[v] = u;
                    q.push(v);
                }
            }
        }
        // distHop[cd[i]] is guaranteed >=0 since the generated graph is connected
        long long pathCost = 0;
        int cur = cd[i];
        while (cur != co[i]) {
            int eid = parentEdge[cur];
            pathCost += r_edge_cost(eC[eid], ef[eid]);
            cur = parentNode[cur];
        }
        B += pathCost;
    }
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
