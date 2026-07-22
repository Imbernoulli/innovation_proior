// TIER: greedy
// The obvious approach: minimize total ditch length. Build each field's
// LENGTH-shortest path (Dijkstra by ditch length -- the "cheapest network"
// instinct), group fields by which branch out of the spring they share, and
// split flow equally: equally among branches, then equally again among the
// fields sharing a branch. This never looks at how lossy a route is, so a
// field stuck on a long forced chain, or a short but low-retention route,
// gets exactly the same naive share as an easy nearby field -- and ends up
// starved once its heavy multiplicative loss is applied.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int N, M, F; ll D, CAP, L;
    scanf("%d %d %d %lld %lld %lld", &N, &M, &F, &D, &CAP, &L);
    vector<int> fieldId(F);
    for (int i = 0; i < F; i++) scanf("%d", &fieldId[i]);
    vector<int> eu(M), ev(M); vector<ll> elen(M); vector<int> eret(M);
    vector<vector<pair<int,int>>> adj(N);
    for (int i = 0; i < M; i++){
        scanf("%d %d %lld %d", &eu[i], &ev[i], &elen[i], &eret[i]);
        adj[eu[i]].push_back({ev[i], i});
        adj[ev[i]].push_back({eu[i], i});
    }

    // Dijkstra by ditch length from the spring.
    vector<ll> dist(N, LLONG_MAX);
    vector<int> parentEdge(N, -1);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[0] = 0; pq.push({0, 0});
    while (!pq.empty()){
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto &pr : adj[u]){
            int v = pr.first, eid = pr.second;
            ll nd = d + elen[eid];
            if (nd < dist[v]){ dist[v] = nd; parentEdge[v] = eid; pq.push({nd, v}); }
        }
    }

    vector<vector<int>> path(F);
    vector<int> firstEdge(F);
    for (int i = 0; i < F; i++){
        int v = fieldId[i];
        vector<int> p; p.push_back(v);
        int fe = -1;
        while (v != 0){
            int eid = parentEdge[v];
            fe = eid;
            v = (eu[eid] == v) ? ev[eid] : eu[eid];
            p.push_back(v);
        }
        reverse(p.begin(), p.end());
        path[i] = p;
        firstEdge[i] = fe;
    }

    // group fields by shared first branch out of the spring
    map<int, vector<int>> groups;
    for (int i = 0; i < F; i++) groups[firstEdge[i]].push_back(i);
    int m = (int)groups.size();
    double branchShare = min((double)D / (double)m, (double)CAP);

    vector<double> X(F, 0.0);
    for (auto &kv : groups){
        int s = (int)kv.second.size();
        double per = branchShare / (double)s;
        for (int idx : kv.second) X[idx] = per;
    }

    printf("%d\n", F);
    for (int i = 0; i < F; i++){
        printf("%d %.9f %d", fieldId[i], X[i], (int)path[i].size());
        for (int x : path[i]) printf(" %d", x);
        printf("\n");
    }
    return 0;
}
