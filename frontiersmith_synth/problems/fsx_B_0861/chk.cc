#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = (ll)4e18;

// -----------------------------------------------------------------------------
// Checker / scorer for "Blind Delivery Through Road Closures".
//
// Input:
//   N M K
//   M lines: u v len            (edge i (1-based, in input order) connects u,v)
//   K blocks, each two lines:
//     B_s w_s
//     B_s edge ids blocked in scenario s
//
// Output: N-1 integers -- a permutation of {2,...,N}, the delivery order after
//   leaving depot 1.
//
// FIXED detour-recovery rule (identical for every order, including the
// checker's own reference order): under scenario s, the driver ALWAYS drives
// the shortest path available in the residual graph (edges of scenario s
// removed) between consecutive stops. If the plan wasn't touched by s's
// closures the shortest residual path equals the full-graph shortest path;
// otherwise the driver is forced onto a (possibly much longer) detour. This
// requires no notion of a single canonical "planned path" -- only distances.
//
// Objective (MIN): F = sum over scenarios s of w_s * (sum of the N-1 leg
//   distances under scenario s's residual graph, walking depot -> stop order).
//
// Baseline B (checker-computed): F evaluated on the IDENTITY order
//   (2,3,...,N) -- exactly what solutions/trivial.cpp prints, so F=B and
//   ratio=0.1 for the trivial reference.
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int N, M, K;
vector<array<int,3>> edgeList;               // edgeList[id-1] = {u, v, w}
vector<vector<pair<int,int>>> adj;           // adj[u] = {(v, edgeId)}  (edgeId is 1-based)
vector<vector<int>> blockedIds;              // per scenario, list of blocked edge ids
vector<ll> scenW;

// Dijkstra on the residual graph for scenario s, single source; returns dist[] (size N+1).
vector<ll> dijkstra(int src, const vector<char>& blocked){
    vector<ll> dist(N + 1, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[src] = 0;
    pq.push({0, src});
    while (!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto& [v, eid] : adj[u]){
            if (blocked[eid]) continue;
            ll w = edgeList[eid - 1][2];
            if (dist[u] + w < dist[v]){
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

// Evaluate F for a given visiting order (order[0..N-2] are stops 2..N in some sequence,
// order does NOT include the depot). tour[0] = depot = 1 implicitly.
ll evalOrder(const vector<int>& order){
    ll F = 0;
    for (int s = 0; s < K; s++){
        vector<char> blocked(M + 1, 0);
        for (int id : blockedIds[s]) blocked[id] = 1;
        ll Cs = 0;
        int cur = 1;
        for (int nxt : order){
            vector<ll> dist = dijkstra(cur, blocked);
            if (dist[nxt] >= INF){
                // generator guarantees the backbone spanning tree is never blocked,
                // so this cannot happen; defensive fallback only.
                Cs += (ll)8e8;
            } else {
                Cs += dist[nxt];
            }
            cur = nxt;
        }
        F += scenW[s] * Cs;
    }
    return F;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    K = inf.readInt();

    edgeList.assign(M, {0,0,0});
    adj.assign(N + 1, {});
    for (int i = 1; i <= M; i++){
        int u = inf.readInt(1, N, "u");
        int v = inf.readInt(1, N, "v");
        int w = inf.readInt(1, 1000000, "len");
        edgeList[i - 1] = {u, v, w};
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
    }

    blockedIds.assign(K, {});
    scenW.assign(K, 0);
    for (int s = 0; s < K; s++){
        int Bs = inf.readInt(1, M, "B_s");
        ll w = inf.readLong(1, 1000000000LL, "w_s");
        scenW[s] = w;
        vector<int> ids(Bs);
        for (int j = 0; j < Bs; j++) ids[j] = inf.readInt(1, M, "blocked_edge_id");
        blockedIds[s] = ids;
    }

    // ---- read participant output: permutation of {2,...,N} ----
    vector<char> seen(N + 1, 0);
    vector<int> order(N - 1);
    for (int i = 0; i < N - 1; i++){
        int x = ouf.readInt(2, N, "stop");
        if (seen[x]) quitf(_wa, "stop %d printed more than once", x);
        seen[x] = 1;
        order[i] = x;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the route");

    ll F = evalOrder(order);

    // ---- internal baseline: identity order 2,3,...,N ----
    vector<int> idOrder(N - 1);
    for (int i = 0; i < N - 1; i++) idOrder[i] = i + 2;
    ll B = evalOrder(idOrder);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
