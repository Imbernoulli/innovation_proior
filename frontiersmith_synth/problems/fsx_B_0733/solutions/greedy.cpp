// TIER: greedy
// The obvious approach: hitting time "feels like" hop-count, so repeatedly find
// the single currently-farthest vertex (BFS distance from the depot, used as a
// stand-in for resistance/hitting-time) and wire it to whichever candidate
// shortens ITS distance the most. Always spends the FULL budget k (more
// shortcuts can't hurt... or so it seems), never checking whether a pick
// actually reduces the worst-case objective or how much mass (m, hence the
// global 2*m*R tax on EVERY courier) it costs. It never considers that two
// symmetric bottleneck arms need to be fixed TOGETHER to move the true max.
#include <bits/stdc++.h>
using namespace std;

int n, m0, t, k, M;
vector<pair<int,int>> cand;
vector<vector<int>> adj;

int main(){
    cin >> n >> m0 >> t >> k >> M;
    adj.assign(n + 1, {});
    for (int i = 0; i < m0; i++){
        int u, v; cin >> u >> v;
        adj[u].push_back(v); adj[v].push_back(u);
    }
    cand.resize(M);
    for (int i = 0; i < M; i++){ cin >> cand[i].first >> cand[i].second; }

    vector<char> usedCand(M + 1, 0);
    vector<int> chosen;

    for (int step = 0; step < k; step++){
        // BFS from t on the current graph
        vector<int> dist(n + 1, -1);
        queue<int> q; dist[t] = 0; q.push(t);
        while (!q.empty()){
            int u = q.front(); q.pop();
            for (int v : adj[u]) if (dist[v] == -1){ dist[v] = dist[u] + 1; q.push(v); }
        }
        vector<int> order;
        for (int v = 1; v <= n; v++) if (v != t) order.push_back(v);
        sort(order.begin(), order.end(), [&](int a, int b){
            if (dist[a] != dist[b]) return dist[a] > dist[b];
            return a < b;
        });

        bool picked = false;
        for (int v : order){
            int bestIdx = -1, bestOtherDist = INT_MAX;
            for (int i = 1; i <= M; i++){
                if (usedCand[i]) continue;
                int a = cand[i - 1].first, b = cand[i - 1].second;
                int other = -1;
                if (a == v) other = b;
                else if (b == v) other = a;
                else continue;
                int od = (other == t ? 0 : dist[other]);
                if (od < bestOtherDist){ bestOtherDist = od; bestIdx = i; }
            }
            if (bestIdx != -1){
                usedCand[bestIdx] = 1;
                chosen.push_back(bestIdx);
                int a = cand[bestIdx - 1].first, b = cand[bestIdx - 1].second;
                adj[a].push_back(b); adj[b].push_back(a);
                picked = true;
                break;
            }
        }
        if (!picked){
            for (int i = 1; i <= M; i++){
                if (usedCand[i]) continue;
                usedCand[i] = 1;
                chosen.push_back(i);
                int a = cand[i - 1].first, b = cand[i - 1].second;
                adj[a].push_back(b); adj[b].push_back(a);
                picked = true;
                break;
            }
        }
        if (!picked) break; // no candidates left at all
    }

    cout << chosen.size() << "\n";
    for (size_t i = 0; i < chosen.size(); i++) cout << chosen[i] << (i + 1 < chosen.size() ? ' ' : '\n');
    if (chosen.empty()) cout << "\n";
    return 0;
}
