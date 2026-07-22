#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Grid Spine Under Targeted Outages".
//
// Input:  n m M K D Bbudget P ; m links (u v w) ; D demands (s e) ; K scenarios (c d..).
// Output: q ; then q distinct link indices you keep (the kept subgraph S).
//
// Objective (MIN): F = sum over scenarios k, demand pairs (s,e) of
//    P                    if s,e disconnected in G_k = S minus scenario k's destroyed links
//    dist_{G_k}(s,e)      otherwise  (weighted shortest path).
//
// Baseline B (checker-computed): the spine-only reference, keep exactly links 0..M-1.
//   This is what the trivial reference reproduces -> ratio 0.1.
// Score (min): sc = min(1000, 100 * B / max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int n, m, M, K, D, Bbudget;
ll P;
vector<int> eu, ev;
vector<ll> ew;
vector<int> ds, de;
vector<vector<int>> scen;

ll scoreSelection(const vector<char>& sel){
    static vector<vector<array<ll,3>>> adj;   // per node: {to, w, edgeIndex}
    adj.assign(n, {});
    for (int i = 0; i < m; i++) if (sel[i]){
        adj[eu[i]].push_back({(ll)ev[i], ew[i], (ll)i});
        adj[ev[i]].push_back({(ll)eu[i], ew[i], (ll)i});
    }
    const ll INF = (ll)4e18;
    vector<char> del(m, 0);
    vector<ll> dist(n);
    ll F = 0;
    for (int k = 0; k < K; k++){
        for (int idx : scen[k]) del[idx] = 1;
        for (int j = 0; j < D; j++){
            fill(dist.begin(), dist.end(), INF);
            priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
            dist[ds[j]] = 0; pq.push({0LL, ds[j]});
            while (!pq.empty()){
                auto pr = pq.top(); pq.pop();
                ll d = pr.first; int u = pr.second;
                if (d > dist[u]) continue;
                if (u == de[j]) break;
                for (auto& e : adj[u]){
                    if (del[(int)e[2]]) continue;
                    ll nd = d + e[1];
                    int to = (int)e[0];
                    if (nd < dist[to]){ dist[to] = nd; pq.push({nd, to}); }
                }
            }
            if (dist[de[j]] >= INF) F += P; else F += dist[de[j]];
        }
        for (int idx : scen[k]) del[idx] = 0;
    }
    return F;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    n = inf.readInt(); m = inf.readInt(); M = inf.readInt();
    K = inf.readInt(); D = inf.readInt(); Bbudget = inf.readInt();
    P = inf.readLong();
    eu.resize(m); ev.resize(m); ew.resize(m);
    for (int i = 0; i < m; i++){ eu[i] = inf.readInt(); ev[i] = inf.readInt(); ew[i] = inf.readLong(); }
    ds.resize(D); de.resize(D);
    for (int j = 0; j < D; j++){ ds[j] = inf.readInt(); de[j] = inf.readInt(); }
    scen.assign(K, {});
    for (int k = 0; k < K; k++){
        int c = inf.readInt();
        scen[k].resize(c);
        for (int t = 0; t < c; t++) scen[k][t] = inf.readInt();
    }

    // ---- baseline B: spine-only ----
    vector<char> base(m, 0);
    for (int i = 0; i < M; i++) base[i] = 1;
    ll B = scoreSelection(base);
    if (B <= 0) B = 1;

    // ---- read participant selection ----
    vector<char> sel(m, 0);
    int q = ouf.readInt(0, m, "count");
    if (q > Bbudget) quitf(_wa, "budget exceeded: q=%d > Bbudget=%d", q, Bbudget);
    for (int t = 0; t < q; t++){
        int e = ouf.readInt(0, m - 1, "edge");
        if (sel[e]) quitf(_wa, "duplicate link index %d", e);
        sel[e] = 1;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = scoreSelection(sel);
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
