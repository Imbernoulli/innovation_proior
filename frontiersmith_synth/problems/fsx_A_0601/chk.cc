#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Quarantine Cuts on a Drifting Contact Graph".
//
// Input:  n m k ; then m lines "u v" (1-indexed undirected edges, u<v, connected).
// Output: k lines "u v", each a DISTINCT existing edge to delete; the residual graph on all
//         n nodes must stay CONNECTED.
//
// Objective (MIN): lambda(G') = residual adjacency spectral radius of the graph after the k
//   deletions, via a fixed deterministic power iteration (all-ones start, 400 iterations,
//   Rayleigh quotient). Floor L = 2(m-k)/n (residual average degree, a provable lower bound
//   on lambda). F = max(lambda(G') - L, 1e-6).
//
// Baseline B (checker-built reference cut = "cut the least-central links"): delete the k edges
//   of smallest endpoint degree-product that lie OUTSIDE a BFS spanning tree (so connectivity
//   is guaranteed), measure B = lambda(G_R) - L. The trivial reference reproduces this -> ~0.1.
// Score (min): sc = min(1000, 100 * B / F); ratio = sc/1000, capped at 1.0.
// -----------------------------------------------------------------------------

static int N;
static vector<vector<int>> ADJ; // reusable adjacency (neighbor lists) of a graph on N nodes

// residual spectral radius via deterministic power iteration on ADJ.
static double specRad() {
    vector<double> x(N, 1.0), y(N, 0.0);
    for (int it = 0; it < 400; it++) {
        for (int u = 0; u < N; u++) y[u] = 0.0;
        for (int u = 0; u < N; u++) {
            double xu = x[u];
            for (int v : ADJ[u]) y[v] += xu;
        }
        double nrm = 0.0;
        for (int u = 0; u < N; u++) nrm += y[u] * y[u];
        nrm = sqrt(nrm);
        if (nrm < 1e-12) return 0.0;
        double inv = 1.0 / nrm;
        for (int u = 0; u < N; u++) x[u] = y[u] * inv;
    }
    // Rayleigh quotient x^T A x  (||x||_2 = 1)
    for (int u = 0; u < N; u++) y[u] = 0.0;
    for (int u = 0; u < N; u++) {
        double xu = x[u];
        for (int v : ADJ[u]) y[v] += xu;
    }
    double lam = 0.0;
    for (int u = 0; u < N; u++) lam += x[u] * y[u];
    return lam;
}

static void buildAdjFromMask(int m, const vector<pair<int,int>>& ev, const vector<char>& del) {
    for (int u = 0; u < N; u++) ADJ[u].clear();
    for (int i = 0; i < m; i++) {
        if (del[i]) continue;
        ADJ[ev[i].first].push_back(ev[i].second);
        ADJ[ev[i].second].push_back(ev[i].first);
    }
}

// connectivity of the graph currently in ADJ (over all N nodes).
static bool connectedADJ() {
    vector<char> seen(N, 0);
    vector<int> st;
    st.push_back(0);
    seen[0] = 1;
    int cnt = 1;
    while (!st.empty()) {
        int u = st.back(); st.pop_back();
        for (int v : ADJ[u]) if (!seen[v]) { seen[v] = 1; cnt++; st.push_back(v); }
    }
    return cnt == N;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    ll k = inf.readLong();
    N = n;
    ADJ.assign(N, {});

    vector<pair<int,int>> ev(m);
    map<pair<int,int>, int> emap;
    vector<int> deg(n, 0);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt() - 1;
        int v = inf.readInt() - 1;
        if (u > v) swap(u, v);
        ev[i] = {u, v};
        emap[{u, v}] = i;
        deg[u]++; deg[v]++;
    }

    // ---- read & validate the participant's k deletions ----
    vector<char> del(m, 0);
    for (ll j = 0; j < k; j++) {
        int a = ouf.readInt(1, n, "u");
        int b = ouf.readInt(1, n, "v");
        a--; b--;
        if (a == b) quitf(_wa, "deletion %lld is a self-loop (%d)", j + 1, a + 1);
        if (a > b) swap(a, b);
        auto it = emap.find({a, b});
        if (it == emap.end())
            quitf(_wa, "deletion %lld = (%d,%d) is not an edge of the graph", j + 1, a + 1, b + 1);
        int id = it->second;
        if (del[id]) quitf(_wa, "edge (%d,%d) deleted more than once", a + 1, b + 1);
        del[id] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %lld deletions", k);

    // ---- feasibility: residual graph must stay connected ----
    buildAdjFromMask(m, ev, del);
    if (!connectedADJ())
        quitf(_wa, "residual graph is disconnected after the deletions");

    // ---- participant objective ----
    double L = 2.0 * (double)(m - k) / (double)n;   // provable floor on lambda
    double lamP = specRad();
    double F = lamP - L;
    if (F < 1e-6) F = 1e-6;

    // ---- baseline B: least-central non-tree cut ----
    // BFS spanning tree in edge-index order (deterministic; matches the trivial reference).
    vector<vector<pair<int,int>>> tadj(n); // (neighbor, edge-id) in input order
    for (int i = 0; i < m; i++) {
        tadj[ev[i].first].push_back({ev[i].second, i});
        tadj[ev[i].second].push_back({ev[i].first, i});
    }
    vector<char> intree(m, 0), seen(n, 0);
    for (int s = 0; s < n; s++) {
        if (seen[s]) continue;
        seen[s] = 1;
        queue<int> q; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto &pr : tadj[u]) {
                int v = pr.first, id = pr.second;
                if (!seen[v]) { seen[v] = 1; intree[id] = 1; q.push(v); }
            }
        }
    }
    vector<int> cand;
    cand.reserve(m);
    for (int i = 0; i < m; i++) if (!intree[i]) cand.push_back(i);
    // smallest endpoint degree-product first (tie-break edge id) = least-central links.
    sort(cand.begin(), cand.end(), [&](int a, int b) {
        ll pa = (ll)deg[ev[a].first] * deg[ev[a].second];
        ll pb = (ll)deg[ev[b].first] * deg[ev[b].second];
        if (pa != pb) return pa < pb;
        return a < b;
    });
    vector<char> delR(m, 0);
    ll take = min<ll>(k, (ll)cand.size());
    for (ll j = 0; j < take; j++) delR[cand[j]] = 1;
    buildAdjFromMask(m, ev, delR);
    double lamR = specRad();
    double B = lamR - L;
    if (B < 1e-6) B = 1e-6;

    double sc = min(1000.0, 100.0 * B / F);
    quitp(sc / 1000.0, "OK lamP=%.4f lamR=%.4f L=%.4f F=%.4f B=%.4f Ratio: %.6f",
          lamP, lamR, L, F, B, sc / 1000.0);
    return 0;
}
