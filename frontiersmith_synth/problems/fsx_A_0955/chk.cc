#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Placards Printed Once, Fires Vary".
//
// Input:
//   N M E X K
//   X exit node ids
//   N populations (room 1..N)
//   E lines: u v cap len            (undirected corridor, 1-indexed edge id = order)
//   K scenario blocks: D_s, then D_s lines "edgeId pct"  (effective cap in that
//   scenario = max(1, cap*pct/100); edges not listed keep pct=100)
//
// Output: N lines, room i (in order): "k v_1 ... v_k"  a SIMPLE path in the base
//   graph with v_1=i and v_k an exit node.
//
// Objective (MIN): for a fixed routing, load[e] = sum of pop_i over rooms whose
//   path uses edge e (load does not depend on the scenario -- routes are static).
//   In scenario s, edge e's transit contribution to any room using it is
//   len_e * ceil(load[e] / effCap_s(e)); a room's clearance time in scenario s is
//   the sum of that over its path; F(s) = max room clearance in scenario s;
//   objective G = max_s F(s) (worst scenario). Minimize G.
//
// Baseline B (checker-computed "do nothing"): route every room to the NEAREST
//   exit by HOP COUNT (unweighted BFS, ignorant of the actual transit weights and
//   of every scenario) -- in this topology that always lands on the bypass exit
//   (2 hops) over the hub exit (3 hops), a naive, capacity/scenario-blind rule.
//   Score: ratio = min(1000, 100*B/max(1,G)) / 1000.
// -----------------------------------------------------------------------------

int N, M, Ecnt, X, K;
vector<int> exits;
vector<char> isExit;
vector<ll> pop_;
struct Edge{ int u, v; ll cap, len; };
vector<Edge> edges;                       // 1-indexed via edges[1..Ecnt]
vector<vector<pair<int,int>>> adj;        // adj[node] = (neighbor, edgeId)
unordered_map<ll,int> emap;               // canonical (u<v) key -> edgeId
vector<vector<pair<int,int>>> scen;       // scen[s] = (edgeId, pct)

ll key(int u, int v){ if (u > v) swap(u, v); return (ll)u * (M + 5) + v; }

// Compute the worst-scenario clearance time G for a fixed set of per-room edge
// sequences (routes). load[e] is scenario-independent (routes are static).
ll computeG(vector<vector<int>>& routeEdges){
    vector<ll> load(Ecnt + 1, 0);
    for (int i = 1; i <= N; i++)
        for (int eid : routeEdges[i]) load[eid] += pop_[i];

    ll G = 0;
    vector<int> pct(Ecnt + 1);
    for (int s = 1; s <= K; s++){
        fill(pct.begin(), pct.end(), 100);
        for (auto &pr : scen[s]) pct[pr.first] = pr.second;
        ll Fs = 0;
        for (int i = 1; i <= N; i++){
            ll tt = 0;
            for (int eid : routeEdges[i]){
                ll effCap = max(1LL, edges[eid].cap * (ll)pct[eid] / 100);
                ll waves = (load[eid] + effCap - 1) / effCap;
                tt += edges[eid].len * waves;
            }
            Fs = max(Fs, tt);
        }
        G = max(G, Fs);
    }
    return G;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt(1, 1000000, "N");
    M = inf.readInt(1, 2000000, "M");
    Ecnt = inf.readInt(1, 4000000, "E");
    X = inf.readInt(1, 1000, "X");
    K = inf.readInt(1, 1000, "K");

    exits.resize(X);
    isExit.assign(M + 1, 0);
    for (int i = 0; i < X; i++){
        exits[i] = inf.readInt(1, M, "exit");
        isExit[exits[i]] = 1;
    }

    pop_.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) pop_[i] = inf.readLong(1, (ll)1e12, "pop");

    edges.assign(Ecnt + 1, Edge{0,0,0,0});
    adj.assign(M + 1, {});
    for (int e = 1; e <= Ecnt; e++){
        int u = inf.readInt(1, M, "u");
        int v = inf.readInt(1, M, "v");
        ll cap = inf.readLong(1, (ll)1e15, "cap");
        ll len = inf.readLong(1, (ll)1e9, "len");
        edges[e] = {u, v, cap, len};
        adj[u].push_back({v, e});
        adj[v].push_back({u, e});
        emap[key(u, v)] = e;
    }

    scen.assign(K + 1, {});
    for (int s = 1; s <= K; s++){
        int D = inf.readInt(0, 1000000, "D_s");
        scen[s].resize(D);
        for (int j = 0; j < D; j++){
            int eid = inf.readInt(1, Ecnt, "edgeId");
            int pct = inf.readInt(1, 100, "pct");
            scen[s][j] = {eid, pct};
        }
    }

    // ---- internal baseline B: nearest exit by HOP COUNT (multi-source BFS) ----
    vector<int> dist(M + 1, -1), par(M + 1, -1), parEdge(M + 1, -1);
    deque<int> q;
    for (int x : exits){ dist[x] = 0; q.push_back(x); }
    while (!q.empty()){
        int u = q.front(); q.pop_front();
        for (auto &pr : adj[u]){
            int v = pr.first, eid = pr.second;
            if (dist[v] == -1){
                dist[v] = dist[u] + 1;
                par[v] = u; parEdge[v] = eid;
                q.push_back(v);
            }
        }
    }
    vector<vector<int>> baseRoute(N + 1);
    for (int i = 1; i <= N; i++){
        if (dist[i] < 0) quitf(_fail, "generator bug: room %d unreachable from any exit", i);
        int cur = i;
        while (!isExit[cur]){
            baseRoute[i].push_back(parEdge[cur]);
            cur = par[cur];
        }
    }
    ll B = computeG(baseRoute);
    if (B <= 0) B = 1;

    // ---- read & validate the participant's routes ----
    vector<vector<int>> route(N + 1);
    vector<int> stampArr(M + 1, 0);
    int stampCtr = 0;
    for (int i = 1; i <= N; i++){
        int k = ouf.readInt(2, M, "path_len");
        vector<int> path(k);
        stampCtr++;
        for (int j = 0; j < k; j++){
            int v = ouf.readInt(1, M, "node");
            if (stampArr[v] == stampCtr) quitf(_wa, "room %d: node %d repeated in path (not simple)", i, v);
            stampArr[v] = stampCtr;
            path[j] = v;
        }
        if (path[0] != i) quitf(_wa, "room %d: path must start at the room's own node, got %d", i, path[0]);
        if (!isExit[path[k - 1]]) quitf(_wa, "room %d: path must end at an exit, got %d", i, path[k - 1]);
        for (int j = 0; j + 1 < k; j++){
            auto it = emap.find(key(path[j], path[j + 1]));
            if (it == emap.end())
                quitf(_wa, "room %d: no corridor edge between %d and %d", i, path[j], path[j + 1]);
            route[i].push_back(it->second);
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after all N room routes");

    ll F = computeG(route);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
