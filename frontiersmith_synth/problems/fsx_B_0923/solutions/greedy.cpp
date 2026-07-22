// TIER: greedy
// The obvious first instinct: "minimize access to nearest depot" reads like
// textbook k-median/facility-location, so just use HOP distance as the
// metric. Repeatedly add the vertex that most reduces the population-
// weighted sum of hop-distance to the nearest chosen depot (classic greedy
// facility location), recomputing via BFS from every candidate each round.
// This treats every road the same everywhere in the city -- it has no idea
// that a road inside a dense block is backed up by many parallel detours
// (so it barely matters electrically) while a road on a thin bridge is the
// ONLY way through (so a random walker trapped there wanders back and forth
// far longer than its hop-count would suggest).
#include <bits/stdc++.h>
using namespace std;

int n, m, k;
vector<long long> pop_;
vector<vector<int>> adj;

vector<int> bfsFrom(int src){
    vector<int> dist(n + 1, -1);
    queue<int> q; dist[src] = 0; q.push(src);
    while (!q.empty()){
        int u = q.front(); q.pop();
        for (int v : adj[u]) if (dist[v] == -1){ dist[v] = dist[u] + 1; q.push(v); }
    }
    return dist;
}

int main(){
    cin >> n >> m >> k;
    pop_.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) cin >> pop_[i];
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++){ int u, v; cin >> u >> v; adj[u].push_back(v); adj[v].push_back(u); }

    vector<int> curDist(n + 1, INT_MAX / 2);
    vector<char> isDepot(n + 1, 0);
    vector<int> chosen;

    for (int step = 0; step < k; step++){
        int bestV = -1; long long bestGain = -1;
        for (int c = 1; c <= n; c++){
            if (isDepot[c]) continue;
            vector<int> dist = bfsFrom(c);
            long long gain = 0;
            for (int v = 1; v <= n; v++){
                int d = dist[v];
                if (d < curDist[v]) gain += pop_[v] * (long long)(curDist[v] - d);
            }
            if (gain > bestGain){ bestGain = gain; bestV = c; }
        }
        isDepot[bestV] = 1; chosen.push_back(bestV);
        vector<int> dist = bfsFrom(bestV);
        for (int v = 1; v <= n; v++) curDist[v] = min(curDist[v], dist[v]);
    }

    for (size_t i = 0; i < chosen.size(); i++) cout << chosen[i] << (i + 1 < chosen.size() ? ' ' : '\n');
    return 0;
}
