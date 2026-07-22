// TIER: greedy
// The obvious first pass: plain nearest-neighbour TSP construction using the RAW,
// undisturbed road network -- read the scenarios only to consume the input, never
// let them influence the route. From the current stop, always drive to the nearest
// unvisited stop by true shortest-path distance. This ignores which roads any
// scenario would actually block, so on networks with a cheap "shortcut" edge that
// several scenarios rely on being open, this order repeatedly depends on it.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll INF = (ll)4e18;

int N, M, K;
vector<array<int,3>> edges;
vector<vector<pair<int,int>>> adj;   // (neighbor, weight)

vector<ll> dijkstra(int src){
    vector<ll> dist(N + 1, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<pair<ll,int>>> pq;
    dist[src] = 0; pq.push({0, src});
    while (!pq.empty()){
        auto [d,u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto& [v,w] : adj[u]) if (dist[u]+w < dist[v]){ dist[v]=dist[u]+w; pq.push({dist[v],v}); }
    }
    return dist;
}

int main(){
    scanf("%d %d %d", &N, &M, &K);
    adj.assign(N+1, {});
    for (int i = 0; i < M; i++){
        int u,v,w; scanf("%d %d %d", &u,&v,&w);
        adj[u].push_back({v,w});
        adj[v].push_back({u,w});
    }
    for (int s = 0; s < K; s++){
        int B; ll w; scanf("%d %lld", &B, &w);
        for (int j = 0; j < B; j++){ int id; scanf("%d", &id); }
    }

    vector<char> visited(N+1, 0);
    visited[1] = 1;
    int cur = 1;
    vector<int> order;
    for (int step = 0; step < N-1; step++){
        vector<ll> dist = dijkstra(cur);
        int best = -1; ll bd = INF;
        for (int v = 2; v <= N; v++) if (!visited[v] && dist[v] < bd){ bd = dist[v]; best = v; }
        order.push_back(best);
        visited[best] = 1;
        cur = best;
    }
    for (size_t i = 0; i < order.size(); i++) printf("%d%c", order[i], i+1<order.size()?' ':'\n');
    return 0;
}
