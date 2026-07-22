// TIER: strong
// The insight: a static route is exposed to EVERY scenario in the ensemble, so the
// right per-edge capacity to plan against is its WORST case across all K scenarios,
// not its nominal value. For each room, compute the worst-case-bottlenecked route
// to EVERY exit (not just the nearest one); then assign rooms to exits one at a
// time (largest population first) picking whichever candidate route minimizes the
// MARGINAL worst-case congestion time given what has already been committed to
// each corridor. This is a load-balancing / equalization argument across the
// worst-case-capacitated graph -- it routes some rooms of the same neighborhood
// through a nominally-slower, less-shared corridor specifically because the fast
// shared one is what the ensemble is planted to damage, and it naturally splits a
// cluster's population across corridors rather than committing all of it to one.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF64 = (ll)4e18;

int main(){
    int N, M, E, X, K;
    scanf("%d %d %d %d %d", &N, &M, &E, &X, &K);
    vector<int> exits(X);
    vector<char> isExit(M + 1, 0);
    for (int i = 0; i < X; i++){ scanf("%d", &exits[i]); isExit[exits[i]] = 1; }
    vector<ll> pop_(N + 1, 0);
    for (int i = 1; i <= N; i++) scanf("%lld", &pop_[i]);

    vector<vector<pair<int,ll>>> adjW(M + 1);
    vector<vector<pair<int,int>>> adjE(M + 1);
    vector<ll> ecap(E + 1), elen(E + 1);
    for (int e = 1; e <= E; e++){
        int u, v; ll cap, len;
        scanf("%d %d %lld %lld", &u, &v, &cap, &len);
        ecap[e] = cap; elen[e] = len;
        adjW[u].push_back({v, len}); adjW[v].push_back({u, len});
        adjE[u].push_back({v, e});   adjE[v].push_back({u, e});
    }
    vector<ll> worstCap(E + 1);
    for (int e = 1; e <= E; e++) worstCap[e] = ecap[e];
    for (int s = 1; s <= K; s++){
        int D; scanf("%d", &D);
        for (int j = 0; j < D; j++){
            int eid, pct; scanf("%d %d", &eid, &pct);
            ll cand = max(1LL, ecap[eid] * (ll)pct / 100);
            worstCap[eid] = min(worstCap[eid], cand);
        }
    }

    // Dijkstra (by nominal length) from every exit -> per-exit dist/parent trees.
    vector<vector<ll>> dist(X, vector<ll>(M + 1, INF64));
    vector<vector<int>> par(X, vector<int>(M + 1, -1)), parEdge(X, vector<int>(M + 1, -1));
    for (int xi = 0; xi < X; xi++){
        int src = exits[xi];
        auto &ds = dist[xi]; auto &pa = par[xi]; auto &pe = parEdge[xi];
        priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
        ds[src] = 0; pq.push({0, src});
        while (!pq.empty()){
            auto [d, u] = pq.top(); pq.pop();
            if (d > ds[u]) continue;
            for (size_t k = 0; k < adjW[u].size(); k++){
                int v = adjW[u][k].first; ll w = adjW[u][k].second;
                int eid = adjE[u][k].second;
                if (ds[u] + w < ds[v]){ ds[v] = ds[u] + w; pa[v] = u; pe[v] = eid; pq.push({ds[v], v}); }
            }
        }
    }

    // Reconstruct, per room, the candidate path (edge list) to each reachable exit.
    vector<vector<vector<int>>> cand(N + 1); // cand[i][xi] = edge list
    for (int i = 1; i <= N; i++){
        cand[i].assign(X, {});
        for (int xi = 0; xi < X; xi++){
            if (dist[xi][i] >= INF64) continue;
            vector<int> path;
            int cur = i;
            while (!isExit[cur]){ path.push_back(parEdge[xi][cur]); cur = par[xi][cur]; }
            cand[i][xi] = path;
        }
    }

    // Assign rooms (largest population first) minimizing marginal worst-case delay.
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){ return pop_[a] > pop_[b]; });

    vector<ll> load(E + 1, 0);
    vector<int> chosenX(N + 1, -1);
    for (int i : order){
        ll bestScore = -1; int bestXi = -1;
        for (int xi = 0; xi < X; xi++){
            if (dist[xi][i] >= INF64) continue;
            ll delay = 0;
            for (int eid : cand[i][xi]){
                ll ec = max(1LL, worstCap[eid]);
                ll waves = (load[eid] + pop_[i] + ec - 1) / ec;
                delay += elen[eid] * waves;
            }
            if (bestXi == -1 || delay < bestScore){ bestScore = delay; bestXi = xi; }
        }
        chosenX[i] = bestXi;
        for (int eid : cand[i][bestXi]) load[eid] += pop_[i];
    }

    for (int i = 1; i <= N; i++){
        vector<int> path;
        int cur = i;
        path.push_back(cur);
        int xi = chosenX[i];
        while (!isExit[cur]){ cur = par[xi][cur]; path.push_back(cur); }
        printf("%d", (int)path.size());
        for (int v : path) printf(" %d", v);
        printf("\n");
    }
    return 0;
}
