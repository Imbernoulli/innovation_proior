// checker for "Splitting the Grove: k-Way Orchard Partition"
#include "testlib.h"
#include <vector>
using namespace std;
typedef long long ll;

struct DSU {
    vector<int> par, sz;
    DSU(int n) : par(n + 1), sz(n + 1, 1) { for (int i = 0; i <= n; i++) par[i] = i; }
    int find(int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; }
    void unite(int a, int b) { a = find(a); b = find(b); if (a != b) { if (sz[a] < sz[b]) swap(a, b); par[b] = a; sz[a] += sz[b]; } }
};

int n, K, LAMBDA;
vector<int> w, p;
vector<int> eu, ev, ec; // 1-indexed edges

ll totalS = 0;

// Given a "removed" mask and a "cutEdge" mask, computes F = COST + BALANCE_PENALTY.
// Also validates exactly K surviving components (caller decides whether to enforce that).
struct EvalResult { bool ok; ll F; int comps; };

EvalResult evaluate(const vector<char> &removed, const vector<char> &cutEdge, ll extraCost) {
    DSU dsu(n);
    for (int i = 1; i <= n - 1; i++) {
        if (removed[eu[i]] || removed[ev[i]]) continue;
        if (cutEdge[i]) continue;
        dsu.unite(eu[i], ev[i]);
    }
    vector<ll> compW; // yield per surviving component, indexed by a compressed id
    vector<int> rootToIdx(n + 1, -1);
    int comps = 0;
    for (int v = 1; v <= n; v++) {
        if (removed[v]) continue;
        int r = dsu.find(v);
        if (rootToIdx[r] == -1) { rootToIdx[r] = comps++; compW.push_back(0); }
        compW[rootToIdx[r]] += w[v];
    }
    if (comps != K) return {false, 0, comps};

    ll P = 0;
    for (ll s : compW) {
        ll dev = (ll)K * s - totalS;
        P += dev * dev;
    }
    ll denom = (ll)K * totalS;
    ll penalty = (denom > 0) ? (LAMBDA * P) / denom : 0;
    ll F = extraCost + penalty;
    return {true, F, comps};
}

int main(int argc, char *argv[]) {
    setName("checker for Splitting the Grove: k-Way Orchard Partition");
    registerTestlibCmd(argc, argv);

    n = inf.readInt(); K = inf.readInt(); LAMBDA = inf.readInt();
    w.assign(n + 1, 0); p.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) w[v] = inf.readInt();
    for (int v = 1; v <= n; v++) p[v] = inf.readInt();
    eu.assign(n, 0); ev.assign(n, 0); ec.assign(n, 0); // 1-indexed 1..n-1
    for (int i = 1; i <= n - 1; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ec[i] = inf.readInt();
    }
    totalS = 0;
    for (int v = 1; v <= n; v++) totalS += w[v];

    // ---- read and validate participant output ----
    int x = ouf.readInt(0, n, "x");
    vector<char> removed(n + 1, 0);
    ll clearCost = 0;
    for (int i = 0; i < x; i++) {
        int r = ouf.readInt(1, n, "r_i");
        if (removed[r]) quitf(_wa, "plot %d cleared more than once", r);
        removed[r] = 1;
        clearCost += p[r];
    }

    int y = ouf.readInt(0, n - 1, "y");
    vector<char> cutEdge(n, 0); // 1-indexed 1..n-1
    ll cutCost = 0;
    for (int i = 0; i < y; i++) {
        int idx = ouf.readInt(1, n - 1, "idx_j");
        if (cutEdge[idx]) quitf(_wa, "path %d cut more than once", idx);
        if (removed[eu[idx]] || removed[ev[idx]])
            quitf(_wa, "path %d touches a cleared plot -- must not be listed", idx);
        cutEdge[idx] = 1;
        cutCost += ec[idx];
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing tokens in output");

    ll extraCost = clearCost + cutCost;
    EvalResult res = evaluate(removed, cutEdge, extraCost);
    if (!res.ok)
        quitf(_wa, "produced %d surviving blocks, need exactly K=%d", res.comps, K);
    ll F = res.F;

    // ---- baseline: cut paths 1..K-1 by input index, clear nothing ----
    vector<char> removedB(n + 1, 0);
    vector<char> cutEdgeB(n, 0);
    ll baseCost = 0;
    for (int i = 1; i <= K - 1; i++) { cutEdgeB[i] = 1; baseCost += ec[i]; }
    EvalResult resB = evaluate(removedB, cutEdgeB, baseCost);
    if (!resB.ok)
        quitf(_fail, "internal: baseline construction did not yield K components (%d != %d)", resB.comps, K);
    ll B = resB.F;
    if (B <= 0) quitf(_fail, "internal: baseline objective B=%lld not positive", B);

    // saturating score: 0.1 exactly when F==B (the baseline), asymptotically -> 1 as F -> 0,
    // never hard-capped -- avoids pileup at 1.0 when a good solution beats the naive
    // baseline by a large, input-dependent factor.
    double ratio = (double)B / ((double)B + 9.0 * (double)F);
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
}
