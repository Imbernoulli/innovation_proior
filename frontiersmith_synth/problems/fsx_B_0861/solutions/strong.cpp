// TIER: strong
// The insight: reformulate "will scenario closures hurt this leg" as a PRICE, not a
// binary. Give every road a risk-adjusted cost = raw length + LAMBDA * (sum of the
// weights w_s of every scenario that blocks it). Roads many/heavy scenarios rely on
// closing now look expensive even though they're nominally cheap shortcuts, so a
// nearest-neighbour construction over this adjusted graph naturally avoids repeatedly
// depending on scenario-fragile roads -- turning "shorter tour" into "structure that
// dodges the closure ensemble". We then run an exact-delta 2-opt local search that
// evaluates moves against the REAL scenario-weighted objective (via precomputed
// per-scenario all-pairs distances), which the risk price alone cannot fully capture.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = (ll)4e18;
const ll LAMBDA = 3;

int N, M, K;
vector<array<ll,3>> edgeList;                 // {u,v,len}, 1-based edge id = index+1
vector<vector<pair<int,int>>> adj;            // adj[u] = {(v, edgeId)}
vector<vector<int>> blockedIds;
vector<ll> scenW;

vector<ll> dijkstraOn(int src, const vector<ll>& wgt, const vector<char>* blocked){
    vector<ll> dist(N+1, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[src] = 0; pq.push({0, src});
    while (!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto& [v,eid] : adj[u]){
            if (blocked && (*blocked)[eid]) continue;
            ll w = wgt[eid];
            if (dist[u]+w < dist[v]){ dist[v]=dist[u]+w; pq.push({dist[v],v}); }
        }
    }
    return dist;
}

int main(){
    scanf("%d %d %d", &N, &M, &K);
    edgeList.assign(M, {0,0,0});
    adj.assign(N+1, {});
    for (int i = 1; i <= M; i++){
        ll u,v,w; scanf("%lld %lld %lld", &u,&v,&w);
        edgeList[i-1] = {u,v,w};
        adj[u].push_back({(int)v, i});
        adj[v].push_back({(int)u, i});
    }
    blockedIds.assign(K, {});
    scenW.assign(K, 0);
    for (int s = 0; s < K; s++){
        int B; ll w; scanf("%d %lld", &B, &w);
        scenW[s] = w;
        vector<int> ids(B);
        for (int j = 0; j < B; j++) scanf("%d", &ids[j]);
        blockedIds[s] = ids;
    }

    // ---- per-scenario all-pairs distance matrices (real objective, O(1) lookup later) ----
    vector<vector<char>> blockedArr(K, vector<char>(M+1, 0));
    for (int s = 0; s < K; s++) for (int id : blockedIds[s]) blockedArr[s][id] = 1;
    vector<ll> rawW(M+1);
    for (int i = 1; i <= M; i++) rawW[i] = edgeList[i-1][2];

    vector<vector<vector<ll>>> distMat(K, vector<vector<ll>>(N+1));
    for (int s = 0; s < K; s++)
        for (int u = 1; u <= N; u++)
            distMat[s][u] = dijkstraOn(u, rawW, &blockedArr[s]);

    // ---- risk-adjusted graph for construction ----
    vector<ll> risk(M+1, 0);
    for (int s = 0; s < K; s++) for (int id : blockedIds[s]) risk[id] += scenW[s];
    vector<ll> adjW(M+1);
    for (int i = 1; i <= M; i++) adjW[i] = rawW[i] + LAMBDA * risk[i];

    vector<vector<ll>> adjDist(N+1);
    for (int u = 1; u <= N; u++) adjDist[u] = dijkstraOn(u, adjW, nullptr);

    // ---- nearest-neighbour construction on the risk-adjusted graph ----
    vector<char> visited(N+1, 0);
    visited[1] = 1;
    int cur = 1;
    vector<int> order;
    for (int step = 0; step < N-1; step++){
        int best = -1; ll bd = INF;
        for (int v = 2; v <= N; v++) if (!visited[v] && adjDist[cur][v] < bd){ bd = adjDist[cur][v]; best = v; }
        order.push_back(best);
        visited[best] = 1;
        cur = best;
    }

    // ---- exact-delta 2-opt local search against the REAL objective ----
    vector<int> T(N);           // T[0]=depot=1, T[1..N-1]=order
    T[0] = 1;
    for (int i = 0; i < (int)order.size(); i++) T[i+1] = order[i];

    auto legCostSum = [&](int a, int b)->ll{
        ll s = 0;
        for (int sIdx = 0; sIdx < K; sIdx++) s += scenW[sIdx] * distMat[sIdx][a][b];
        return s;
    };

    long long evalBudget = 1200000;   // candidate-move cap, keeps this well under the time limit
    bool improved = true;
    int sweeps = 0;
    while (improved && sweeps < 12 && evalBudget > 0){
        improved = false;
        sweeps++;
        for (int i = 1; i <= N-2 && evalBudget > 0; i++){
            for (int j = i+1; j <= N-1 && evalBudget > 0; j++){
                evalBudget -= K;
                ll before = legCostSum(T[i-1], T[i]) + (j+1 <= N-1 ? legCostSum(T[j], T[j+1]) : 0);
                ll after  = legCostSum(T[i-1], T[j]) + (j+1 <= N-1 ? legCostSum(T[i], T[j+1]) : 0);
                if (after < before){
                    reverse(T.begin()+i, T.begin()+j+1);
                    improved = true;
                }
            }
        }
    }

    for (int i = 1; i < N; i++) printf("%d%c", T[i], i+1<N?' ':'\n');
    return 0;
}
