// TIER: trivial
// Naive baseline: for every packet, take the durationwise-shortest path
// (ignoring all contention), process packets in input order, and just wait
// for the same edge to free up if its next slot is taken. Never reroutes.
// This is exactly the checker's internal baseline construction.
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
    unordered_set<ll> used;

    for (int k = 0; k < K; k++){
        vector<int> path = shortestPath(N, edges, adj, S[k], Tt[k]);
        vector<ll> depart;
        if (!path.empty() && tryPath(path, edges, used, R[k], D[k], true, &depart)){
            cout << path.size();
            for (size_t j = 0; j < path.size(); j++) cout << ' ' << depart[j] << ' ' << path[j];
            cout << '\n';
        } else {
            cout << -1 << '\n';
        }
    }
    return 0;
}
