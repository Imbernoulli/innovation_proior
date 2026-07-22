// TIER: greedy
// The obvious "smarter" one-pass heuristic: still always takes the
// durationwise-shortest path for every packet (never reroutes), but
// processes packets in HIGH-VALUE-FIRST order so that when something must
// be dropped, it drops the cheapest thing. This is a real, deserved
// improvement on plain arrival-order processing wherever a bit of scarcity
// forces a couple of drops -- but it is still blind to WHY the shared
// bottleneck edge is scarce: on a trap cell it happily lets high-value
// flexible packets (which would have arrived fine via the slower detour
// too) exhaust the bottleneck's narrow window before any low-value captive
// packet (which has no detour at all) gets a turn.
#include "../routing_lib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M, K; ll T;
    cin >> N >> M >> K >> T;
    vector<Edge> edges(M);
    for (int i = 0; i < M; i++) cin >> edges[i].u >> edges[i].v >> edges[i].dur;
    vector<int> S(K), Tt(K);
    vector<ll> R(K), D(K), V(K);
    for (int k = 0; k < K; k++) cin >> S[k] >> Tt[k] >> R[k] >> D[k] >> V[k];

    vector<vector<int>> adj = buildAdj(N, edges);

    vector<int> order(K);
    for (int k = 0; k < K; k++) order[k] = k;
    sort(order.begin(), order.end(), [&](int a, int b){
        if (V[a] != V[b]) return V[a] > V[b];
        return a < b;
    });

    unordered_set<ll> used;
    vector<int> resultLen(K, -1);
    vector<vector<int>> resultPath(K);
    vector<vector<ll>> resultDepart(K);

    for (int k : order){
        vector<int> path = shortestPath(N, edges, adj, S[k], Tt[k]);
        vector<ll> depart;
        if (!path.empty() && tryPath(path, edges, used, R[k], D[k], true, &depart)){
            resultLen[k] = (int)path.size();
            resultPath[k] = path;
            resultDepart[k] = depart;
        }
    }

    for (int k = 0; k < K; k++){
        if (resultLen[k] < 0){ cout << -1 << '\n'; continue; }
        cout << resultLen[k];
        for (int j = 0; j < resultLen[k]; j++) cout << ' ' << resultDepart[k][j] << ' ' << resultPath[k][j];
        cout << '\n';
    }
    return 0;
}
