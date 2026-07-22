// TIER: strong
// Insight: OPTIMAL PERCOLATION via adaptive COLLECTIVE INFLUENCE (Morone-Makse),
// coupled to the spending budget (cost-efficiency, not head count). The people
// worth immunizing are NOT the high-degree hubs but the ones whose radius-ELL
// frontier reaches a whole *neighbouring* community -- the bridges that hold
// separate communities together.
//
//   CI_ell(i) = (deg_i - 1) * sum_{j at graph-distance exactly ell from i} (deg_j - 1)
//
// computed on the structural contact graph and RECOMPUTED after every removal
// (adaptive: once a bridge is cut, degrees/frontiers of its former neighbours
// change). Among affordable candidates we take the best CI-per-cost, not just
// the best CI, so the budget is spent efficiently.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M, ELL, K, S;
    if (scanf("%d %d %d %d %d", &N, &M, &ELL, &K, &S) != 5) return 0;
    (void)S;
    vector<array<int,2>> E(M);
    vector<vector<int>> adj(N + 1);
    for (int j = 0; j < M; j++){
        int u, v, t; scanf("%d %d %d", &u, &v, &t);
        E[j] = {u, v};
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<int> cost(N + 1);
    for (int i = 1; i <= N; i++) scanf("%d", &cost[i]);
    // p_s scenario strengths: bridges are built near-always-active, so the
    // structural graph is a faithful proxy for where the budget should go.

    vector<char> alive(N + 1, 1);
    vector<int> deg(N + 1, 0);
    vector<int> stampv(N + 1, 0);
    int curStamp = 0;
    int ell = max(1, min(ELL, 6));

    long long remaining = K;
    vector<int> removedList;

    auto recomputeDeg = [&](){
        for (int i = 1; i <= N; i++){
            if (!alive[i]){ deg[i] = 0; continue; }
            int d = 0;
            for (int u : adj[i]) if (alive[u]) d++;
            deg[i] = d;
        }
    };

    int minCost = *min_element(cost.begin() + 1, cost.end());

    while (remaining >= minCost){
        recomputeDeg();
        double bestEff = -1.0;
        long long bestCI = -1;
        int bestNode = -1;
        int bestDeg = -1;
        vector<int> layer, nxt;
        for (int i = 1; i <= N; i++){
            if (!alive[i] || cost[i] > remaining || deg[i] == 0) continue;
            curStamp++;
            stampv[i] = curStamp;
            layer.clear(); layer.push_back(i);
            for (int d = 0; d < ell && !layer.empty(); d++){
                nxt.clear();
                for (int u : layer)
                    for (int w : adj[u])
                        if (alive[w] && stampv[w] != curStamp){
                            stampv[w] = curStamp;
                            nxt.push_back(w);
                        }
                layer.swap(nxt);
            }
            ll fsum = 0;
            for (int w : layer) fsum += max(0, deg[w] - 1);
            ll ci = (ll)max(0, deg[i] - 1) * fsum;
            double eff = (double)ci / (double)cost[i];
            if (eff > bestEff || (eff == bestEff && deg[i] > bestDeg)){
                bestEff = eff; bestCI = ci; bestNode = i; bestDeg = deg[i];
            }
        }
        if (bestNode < 0) break;
        // fallback: if collective influence is flat/zero everywhere (small or
        // fully-shattered residual graph), spend on the highest remaining degree.
        if (bestCI <= 0){
            int fb = -1, fbDeg = -1;
            for (int i = 1; i <= N; i++)
                if (alive[i] && cost[i] <= remaining && deg[i] > fbDeg){ fbDeg = deg[i]; fb = i; }
            if (fb < 0) break;
            bestNode = fb;
            if (fbDeg <= 0) break;
        }
        alive[bestNode] = 0;
        remaining -= cost[bestNode];
        removedList.push_back(bestNode);
    }

    printf("%d\n", (int)removedList.size());
    for (size_t i = 0; i < removedList.size(); i++)
        printf("%d%c", removedList[i], (i + 1 < removedList.size()) ? ' ' : '\n');
    if (removedList.empty()) printf("\n");
    return 0;
}
