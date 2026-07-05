// Checker/scorer for "Grid Hardening Attack" (interdiction-effective-resistance).
// Reads the network from inf, the participant's cut set from ouf, validates feasibility
// strictly, then scores by weighted s-t effective resistance vs the do-nothing baseline.
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

static int n, m, s, t;
static long long Bbud;
static vector<int> eu, ev, er, ec;

// Weighted s-t effective resistance of the subgraph given by 'alive' edges.
// Restricts to the connected component of s (t is guaranteed to be in it by the caller).
// R(s,t) = det(L^{(s,t)}) / det(L^{(s)}) computed in log-space via Cholesky (the reduced
// Laplacian minors are symmetric positive definite for a connected component grounded at s).
static double effRes(const vector<char>& alive) {
    // adjacency for BFS on the alive subgraph
    vector<vector<int>> g(n + 1);
    for (int e = 0; e < m; e++) if (alive[e]) {
        g[eu[e]].push_back(ev[e]);
        g[ev[e]].push_back(eu[e]);
    }
    vector<int> loc(n + 1, -1), comp;
    comp.push_back(s); loc[s] = 0;
    for (size_t h = 0; h < comp.size(); h++) {
        int u = comp[h];
        for (int w : g[u]) if (loc[w] < 0) { loc[w] = (int)comp.size(); comp.push_back(w); }
    }
    int p = (int)comp.size();
    // dense weighted Laplacian on the component (p x p)
    vector<double> L((size_t)p * p, 0.0);
    auto AT = [&](int i, int j) -> double& { return L[(size_t)i * p + j]; };
    for (int e = 0; e < m; e++) if (alive[e]) {
        int a = loc[eu[e]], b = loc[ev[e]];
        if (a < 0 || b < 0) continue;            // edge in another component
        double gc = 1.0 / (double)er[e];
        AT(a, b) -= gc; AT(b, a) -= gc;
        AT(a, a) += gc; AT(b, b) += gc;
    }
    int si = loc[s], ti = loc[t];
    // log-determinant of the principal submatrix of L that keeps all indices not in 'excl'
    auto logdet = [&](vector<int> excl) -> double {
        vector<char> rm(p, 0);
        for (int x : excl) rm[x] = 1;
        vector<int> keep;
        for (int i = 0; i < p; i++) if (!rm[i]) keep.push_back(i);
        int q = (int)keep.size();
        if (q == 0) return 0.0;                  // empty matrix -> det = 1
        vector<double> A((size_t)q * q);
        for (int i = 0; i < q; i++)
            for (int j = 0; j < q; j++)
                A[(size_t)i * q + j] = AT(keep[i], keep[j]);
        // in-place Cholesky (lower); logdet = 2 * sum log(diag)
        double ld = 0.0;
        for (int i = 0; i < q; i++) {
            double d = A[(size_t)i * q + i];
            for (int k = 0; k < i; k++) { double v = A[(size_t)i * q + k]; d -= v * v; }
            if (d <= 1e-300) d = 1e-300;         // numerical guard (SPD in exact arithmetic)
            double di = sqrt(d);
            A[(size_t)i * q + i] = di;
            ld += log(di);
            for (int j = i + 1; j < q; j++) {
                double sum = A[(size_t)j * q + i];
                for (int k = 0; k < i; k++) sum -= A[(size_t)j * q + k] * A[(size_t)i * q + k];
                A[(size_t)j * q + i] = sum / di;
            }
        }
        return 2.0 * ld;
    };
    double ld_s  = logdet({si});
    double ld_st = logdet({si, ti});
    return exp(ld_st - ld_s);
}

// Is t reachable from s in the alive subgraph?
static bool stConnected(const vector<char>& alive) {
    vector<vector<int>> g(n + 1);
    for (int e = 0; e < m; e++) if (alive[e]) {
        g[eu[e]].push_back(ev[e]);
        g[ev[e]].push_back(eu[e]);
    }
    vector<char> seen(n + 1, 0);
    vector<int> st = {s}; seen[s] = 1;
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        if (u == t) return true;
        for (int w : g[u]) if (!seen[w]) { seen[w] = 1; st.push_back(w); }
    }
    return seen[t];
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);
    n = inf.readInt(); m = inf.readInt(); s = inf.readInt(); t = inf.readInt();
    Bbud = inf.readLong();
    eu.resize(m); ev.resize(m); er.resize(m); ec.resize(m);
    for (int e = 0; e < m; e++) {
        eu[e] = inf.readInt(); ev[e] = inf.readInt();
        er[e] = inf.readInt(); ec[e] = inf.readInt();
    }

    // --- read + strictly validate participant output ---
    int K = ouf.readInt(0, m, "K");
    vector<char> removed(m, 0);
    long long cost = 0;
    for (int i = 0; i < K; i++) {
        int e = ouf.readInt(1, m, "line");
        if (removed[e - 1]) quitf(_wa, "line %d cut more than once", e);
        removed[e - 1] = 1;
        cost += ec[e - 1];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");
    if (cost > Bbud) quitf(_wa, "total cut cost %lld exceeds budget %lld", cost, Bbud);

    vector<char> alive(m);
    for (int e = 0; e < m; e++) alive[e] = (char)!removed[e];
    if (!stConnected(alive)) quitf(_wa, "s and t are disconnected after the cuts");

    double F = effRes(alive);
    vector<char> all(m, 1);
    double B = effRes(all);
    if (!(B > 0.0) || !isfinite(B)) quitf(_fail, "bad baseline B=%.6g", B);
    if (!isfinite(F)) quitf(_wa, "non-finite objective");

    double sc = min(1000.0, 100.0 * F / max(1e-12, B));
    if (sc < 0) sc = 0;
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
}
