#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
ll C;
vector<ll> p, c;
vector<array<ll,3>> edges; // u, v, w

// Weighted global min cut (Stoer-Wagner) on a symmetric matrix g of size n.
// Returns the minimum total crossing weight; 0 if the graph is disconnected.
ll stoerWagner(int n, vector<vector<ll>> g) {
    if (n < 2) return 0;
    ll best = LLONG_MAX;
    vector<int> vert(n);
    for (int i = 0; i < n; i++) vert[i] = i;
    int cur = n;
    while (cur > 1) {
        vector<ll> w(cur, 0);
        vector<char> in(cur, 0);
        int prev = -1, last = -1;
        for (int it = 0; it < cur; it++) {
            int sel = -1;
            for (int j = 0; j < cur; j++)
                if (!in[j] && (sel < 0 || w[j] > w[sel])) sel = j;
            in[sel] = 1;
            prev = last; last = sel;
            if (it == cur - 1) {
                best = min(best, w[sel]);
            } else {
                for (int j = 0; j < cur; j++)
                    if (!in[j]) w[j] += g[vert[sel]][vert[j]];
            }
        }
        // canonical Stoer-Wagner: merge the last selected supernode into the
        // second-to-last selected one (prev).
        for (int j = 0; j < cur; j++) {
            g[vert[prev]][vert[j]] += g[vert[last]][vert[j]];
            g[vert[j]][vert[prev]] += g[vert[j]][vert[last]];
        }
        vert.erase(vert.begin() + last);
        cur--;
    }
    return best == LLONG_MAX ? 0 : best;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    C = inf.readLong();
    p.assign(N + 1, 0);
    c.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) { p[i] = inf.readInt(); c[i] = inf.readInt(); }
    edges.resize(M);
    // pooled weight per unordered adjacent pair, for the baseline B
    map<pair<int,int>, ll> pairW;
    for (int j = 0; j < M; j++) {
        int u = inf.readInt(), v = inf.readInt();
        ll w = inf.readInt();
        edges[j] = {u, v, w};
        int a = min(u, v), b = max(u, v);
        pairW[{a, b}] += w;
    }

    // ---- internal baseline B: best affordable cable-connected pair ----
    ll B = 0;
    for (auto& kv : pairW) {
        int u = kv.first.first, v = kv.first.second;
        if (c[u] + c[v] <= C) {
            ll cand = kv.second * (p[u] + p[v]);
            if (cand > B) B = cand;
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: no affordable connected pair (B=%lld)", B);

    // ---- read & validate the participant's built set S ----
    int r = ouf.readInt(0, N, "r");
    vector<char> built(N + 1, 0);
    vector<int> S;
    ll cost = 0, cover = 0;
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, N, "hubIndex");
        if (built[idx]) quitf(_wa, "hub %d built more than once", idx);
        built[idx] = 1;
        S.push_back(idx);
        cost += c[idx];
        cover += p[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (cost > C) quitf(_wa, "budget exceeded: cost %lld > C %lld", cost, C);

    // ---- objective F = lambda(S) * cover ----
    ll F = 0;
    if ((int)S.size() >= 2) {
        // local reindex + induced weighted adjacency matrix
        int n = (int)S.size();
        vector<int> loc(N + 1, -1);
        for (int i = 0; i < n; i++) loc[S[i]] = i;
        vector<vector<ll>> g(n, vector<ll>(n, 0));
        for (auto& e : edges) {
            int u = (int)e[0], v = (int)e[1];
            if (built[u] && built[v]) {
                int a = loc[u], b = loc[v];
                if (a != b) { g[a][b] += e[2]; g[b][a] += e[2]; }
            }
        }
        ll lam = stoerWagner(n, g);
        F = lam * cover;
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
