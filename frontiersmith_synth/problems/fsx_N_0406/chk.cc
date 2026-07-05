#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

int n, m, q, H;
struct Edge { int u, v; ll w; u64 mask; };   // mask bit r set => r forbidden
vector<Edge> E;
vector<int> hubNode;
vector<ll>  hubBonus;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    q = inf.readInt();
    H = inf.readInt();

    E.resize(m);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        int k = inf.readInt();
        u64 mask = 0;
        for (int j = 0; j < k; j++) {
            int f = inf.readInt();
            mask |= (1ULL << f);
        }
        E[i] = {u, v, w, mask};
    }
    hubNode.resize(H);
    hubBonus.resize(H);
    for (int i = 0; i < H; i++) {
        hubNode[i]  = inf.readInt();
        hubBonus[i] = inf.readInt();
    }

    // ---------- internal baseline B = best CONSTANT labeling ----------
    // For constant label c, every edge has sum r = (2c) mod q.
    //   clearedWeight(c) = totalW - forbW[r]
    //   bonus(c)         = totalHub - forbHubBonus[r]
    // As c ranges over GF(q) and 2 is invertible, r ranges over all residues,
    // so B = totalW + totalHub - min_r ( forbW[r] + forbHubBonus[r] ).
    ll totalW = 0, totalHub = 0;
    vector<ll> forbW(q, 0), forbHubBonus(q, 0);
    for (auto& e : E) {
        totalW += e.w;
        for (int r = 0; r < q; r++) if (e.mask >> r & 1ULL) forbW[r] += e.w;
    }
    // per-hub OR of incident edge masks
    // (build node -> incident masks OR; only for hub nodes to stay light)
    {
        unordered_map<int, u64> hubMaskOf;
        hubMaskOf.reserve(H * 2 + 4);
        vector<char> isHub(n + 1, 0);
        for (int i = 0; i < H; i++) { isHub[hubNode[i]] = 1; hubMaskOf[hubNode[i]] = 0ULL; }
        for (auto& e : E) {
            if (e.u >= 1 && e.u <= n && isHub[e.u]) hubMaskOf[e.u] |= e.mask;
            if (e.v >= 1 && e.v <= n && isHub[e.v]) hubMaskOf[e.v] |= e.mask;
        }
        for (int i = 0; i < H; i++) {
            totalHub += hubBonus[i];
            u64 hm = hubMaskOf[hubNode[i]];
            for (int r = 0; r < q; r++) if (hm >> r & 1ULL) forbHubBonus[r] += hubBonus[i];
        }
    }
    ll minForb = LLONG_MAX;
    for (int r = 0; r < q; r++) minForb = min(minForb, forbW[r] + forbHubBonus[r]);
    ll B = totalW + totalHub - minForb;
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---------- read & validate participant labeling ----------
    vector<int> x(n + 1);
    for (int i = 1; i <= n; i++)
        x[i] = ouf.readInt(0, q - 1, "label");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---------- objective F ----------
    vector<char> isHub(n + 1, 0);
    vector<ll> bonusOf(n + 1, 0);
    vector<char> harmonized(n + 1, 1);
    for (int i = 0; i < H; i++) { isHub[hubNode[i]] = 1; bonusOf[hubNode[i]] = hubBonus[i]; }

    ll F = 0;
    for (auto& e : E) {
        int r = (x[e.u] + x[e.v]) % q;
        bool clear = !((e.mask >> r) & 1ULL);
        if (clear) F += e.w;
        else {
            if (isHub[e.u]) harmonized[e.u] = 0;
            if (isHub[e.v]) harmonized[e.v] = 0;
        }
    }
    for (int i = 0; i < H; i++)
        if (harmonized[hubNode[i]]) F += hubBonus[i];

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
