// TIER: strong
// The insight: the facility-location instinct is right, but hop distance is
// the WRONG metric for a random walk. Cheaply sketch the resistance/access
// geometry instead of solving the true O(n^3) linear system: weight each
// road (u,v) by 1/min(deg(u),deg(v)). Inside a dense block every road has
// many parallel alternates, so high-degree endpoints make that road's
// marginal contribution to access time tiny; a thin bridge road has no
// alternate route at all, so its full weight-1-ish cost survives. Running
// the SAME greedy facility-location construction on this weighted metric
// (Dijkstra instead of BFS) already stops over-spending depots deep inside
// well-connected blocks and instead reaches further out along expensive
// corridors -- including splitting an extremely long bridge mid-span, which
// the weighted metric (unlike hop count) recognizes as high value. A short
// local-search refinement pass (try relocating each depot to whichever
// currently-worst-served hub would help most) then cleans up the placement.
#include <bits/stdc++.h>
using namespace std;

int n, m, k;
vector<long long> pop_;
vector<vector<pair<int,double>>> wadj;

const double INF_D = 1e18;

vector<double> dijkstraFrom(int src){
    vector<double> dist(n + 1, INF_D);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<pair<double,int>>> pq;
    dist[src] = 0.0; pq.push({0.0, src});
    while (!pq.empty()){
        auto pr = pq.top(); pq.pop();
        double d = pr.first; int u = pr.second;
        if (d > dist[u] + 1e-12) continue;
        for (auto &e : wadj[u]){
            int v = e.first; double nd = d + e.second;
            if (nd < dist[v] - 1e-12){ dist[v] = nd; pq.push({nd, v}); }
        }
    }
    return dist;
}

vector<double> multiSourceDijkstra(const vector<int>& sources){
    vector<double> dist(n + 1, INF_D);
    priority_queue<pair<double,int>, vector<pair<double,int>>, greater<pair<double,int>>> pq;
    for (int s : sources){ dist[s] = 0.0; pq.push({0.0, s}); }
    while (!pq.empty()){
        auto pr = pq.top(); pq.pop();
        double d = pr.first; int u = pr.second;
        if (d > dist[u] + 1e-12) continue;
        for (auto &e : wadj[u]){
            int v = e.first; double nd = d + e.second;
            if (nd < dist[v] - 1e-12){ dist[v] = nd; pq.push({nd, v}); }
        }
    }
    return dist;
}

int main(){
    cin >> n >> m >> k;
    pop_.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> pop_[i];
    vector<vector<int>> adj(n + 1);
    vector<pair<int,int>> edgeList(m);
    for (int i = 0; i < m; i++){
        int u, v; cin >> u >> v;
        adj[u].push_back(v); adj[v].push_back(u);
        edgeList[i] = {u, v};
    }
    vector<int> deg_(n + 1, 0);
    for (int v = 1; v <= n; v++) deg_[v] = (int)adj[v].size();

    wadj.assign(n + 1, {});
    for (auto &e : edgeList){
        int u = e.first, v = e.second;
        double w = 1.0 / (double)max(1, min(deg_[u], deg_[v]));
        wadj[u].push_back({v, w});
        wadj[v].push_back({u, w});
    }

    // --- greedy construction on the resistance-sketch metric ---
    vector<char> isDepot(n + 1, 0);
    vector<int> chosen;
    vector<double> curDist(n + 1, INF_D);
    for (int step = 0; step < k; step++){
        int bestV = -1; double bestGain = -1.0;
        for (int c = 1; c <= n; c++){
            if (isDepot[c]) continue;
            vector<double> dist = dijkstraFrom(c);
            double gain = 0.0;
            for (int v = 1; v <= n; v++)
                if (dist[v] < curDist[v] - 1e-12) gain += (double)pop_[v] * (curDist[v] - dist[v]);
            if (gain > bestGain){ bestGain = gain; bestV = c; }
        }
        isDepot[bestV] = 1; chosen.push_back(bestV);
        vector<double> dist = dijkstraFrom(bestV);
        for (int v = 1; v <= n; v++) curDist[v] = min(curDist[v], dist[v]);
    }

    // --- local search: relocate each depot to whichever vertex (scanning
    //     ALL of them -- affordable since the whole point of the weighted
    //     metric is a cheap O(m log n) Dijkstra, not the O(n^3) exact solve)
    //     minimizes total population-weighted distance given the OTHER
    //     depots fixed. Repeated until no depot moves. ---
    for (int round = 0; round < 3; round++){
        bool improved = false;
        for (int j = 0; j < (int)chosen.size(); j++){
            vector<int> others;
            for (int t = 0; t < (int)chosen.size(); t++) if (t != j) others.push_back(chosen[t]);
            vector<double> distOthers = others.empty() ? vector<double>(n + 1, INF_D) : multiSourceDijkstra(others);

            vector<double> distSelf = dijkstraFrom(chosen[j]);
            double curCost = 0.0;
            for (int v = 1; v <= n; v++) curCost += (double)pop_[v] * min(distOthers[v], distSelf[v]);

            int bestC = chosen[j]; double bestCost = curCost;
            for (int c = 1; c <= n; c++){
                bool isOther = false;
                for (int o : others) if (o == c){ isOther = true; break; }
                if (isOther) continue;
                vector<double> dc = dijkstraFrom(c);
                double tot = 0.0;
                for (int v = 1; v <= n; v++) tot += (double)pop_[v] * min(distOthers[v], dc[v]);
                if (tot < bestCost - 1e-9){ bestCost = tot; bestC = c; }
            }
            if (bestC != chosen[j]){ chosen[j] = bestC; improved = true; }
        }
        if (!improved) break;
    }

    for (size_t i = 0; i < chosen.size(); i++) cout << chosen[i] << (i + 1 < chosen.size() ? ' ' : '\n');
    return 0;
}
