#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for "Rush-Hour Rerouting on the Convex-Congestion Metro".
// Minimization.  Multi-commodity routing on a capacitated undirected graph; each segment
// has a CONVEX (quadratic) congestion cost a_e * x_e^2 where x_e is the shared load; leaving
// a passenger unserved costs P.  Feasibility: well-formed origin->destination path flows,
// per-demand served <= volume, per-segment load <= capacity.  Objective
//   F = sum_e a_e * x_e^2  +  P * sum_d unserved_d.
// Baseline B = do-nothing (route nobody) = P * sum_d vol_d  (always feasible, positive).
//   ratio = min(1, (B / max(1,F)) / 10).

static const long long KMAX = 2000000;       // max number of path-flow lines
static const long long VERTCAP = 20000000;   // max total path vertices across all lines

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int D = inf.readInt();
    long long P = inf.readInt();

    // segments; edge key = (long long)min*(N+1)+max -> edge id
    vector<long long> cap(M), acoef(M), load(M, 0);
    unordered_map<long long, int> edgeId;
    edgeId.reserve(M * 2 + 16);
    for (int e = 0; e < M; e++) {
        int u = inf.readInt();
        int v = inf.readInt();
        cap[e] = inf.readInt();
        acoef[e] = inf.readInt();
        long long a = min(u, v), b = max(u, v);
        long long key = a * (long long)(N + 1) + b;
        edgeId[key] = e;
    }

    vector<int> sd(D), td(D);
    vector<long long> vol(D);
    long long totalVol = 0;
    for (int d = 0; d < D; d++) {
        sd[d] = inf.readInt();
        td[d] = inf.readInt();
        vol[d] = inf.readInt();
        totalVol += vol[d];
    }

    // ---- read participant routing ----
    vector<long long> served(D, 0);
    long long totalVerts = 0;
    long long K = ouf.readLong(0, KMAX, "K");

    vector<int> pathEdges;  // reused per line
    for (long long line = 0; line < K; line++) {
        int d = ouf.readInt(1, D, "d") - 1;
        int L = ouf.readInt(2, N, "L");
        totalVerts += L;
        if (totalVerts > VERTCAP) quitf(_wa, "total path length exceeds output limit");

        pathEdges.clear();
        int prev = ouf.readInt(1, N, "w");
        int first = prev;
        for (int j = 1; j < L; j++) {
            int cur = ouf.readInt(1, N, "w");
            if (cur == prev)
                quitf(_wa, "path flow %lld repeats station %d consecutively", line, cur);
            long long a = min(prev, cur), b = max(prev, cur);
            long long key = a * (long long)(N + 1) + b;
            auto it = edgeId.find(key);
            if (it == edgeId.end())
                quitf(_wa, "path flow %lld uses non-existent segment %d-%d", line, prev, cur);
            pathEdges.push_back(it->second);
            prev = cur;
        }
        int last = prev;
        if (first != sd[d])
            quitf(_wa, "path flow %lld: first station %d != origin %d of demand %d",
                  line, first, sd[d], d + 1);
        if (last != td[d])
            quitf(_wa, "path flow %lld: last station %d != destination %d of demand %d",
                  line, last, td[d], d + 1);

        long long f = ouf.readLong(1, vol[d], "f");
        served[d] += f;
        if (served[d] > vol[d])
            quitf(_wa, "demand %d routes %lld > volume %lld", d + 1, served[d], vol[d]);
        for (int eid : pathEdges) {
            load[eid] += f;
            if (load[eid] > cap[eid])
                quitf(_wa, "segment load %lld exceeds capacity %lld", load[eid], cap[eid]);
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the path flows");

    // ---- participant objective F ----
    long long crowd = 0;
    for (int e = 0; e < M; e++) crowd += acoef[e] * load[e] * load[e];
    long long unserved = 0;
    for (int d = 0; d < D; d++) unserved += (vol[d] - served[d]);
    long long F = crowd + P * unserved;

    // ---- baseline: do-nothing (route nobody) ----
    long long B = P * totalVol;
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
