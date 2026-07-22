// TIER: strong
// The insight: value hides behind narrow GATEWAYS (vertex cuts). A grove's stands all share
// the SAME set of gateway neighbours, so grouping valued stands by their neighbour-set
// recovers each gateway for free; a BFS gives every gateway stand its burn deadline. Sealing
// a gateway of size C > B needs ceil(C/B) steps of lead, so this becomes a value-weighted
// DEADLINE-SCHEDULING problem: commit budget by value-density to the gateways still sealable
// in time (latest-fit so early slots stay free for tighter deadlines), pre-closing distant
// chokes BEFORE the flames arrive -- exactly what the frontier-defending greedy cannot do.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int INF = 1e9;

int main(){
    int N, M, B, S;
    if (scanf("%d %d %d %d", &N, &M, &B, &S) != 4) return 0;
    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld", &w[i]);
    vector<int> src(S);
    for (int i = 0; i < S; i++) scanf("%d", &src[i]);
    vector<vector<int>> adj(N + 1);
    for (int e = 0; e < M; e++){
        int u, v; scanf("%d %d", &u, &v);
        adj[u].push_back(v); adj[v].push_back(u);
    }

    // BFS burn distance from sources (protection-free)
    vector<int> dist(N + 1, INF);
    { queue<int> q; for (int s : src) if (dist[s] == INF){ dist[s] = 0; q.push(s); }
      while (!q.empty()){ int u = q.front(); q.pop();
          for (int v : adj[u]) if (dist[v] == INF){ dist[v] = dist[u] + 1; q.push(v); } } }

    // Group reachable valued stands by their (sorted) neighbour-set == their gateway.
    struct Mod { vector<int> gate; ll value; };
    unordered_map<string, int> idx;
    vector<Mod> mods;
    for (int v = 1; v <= N; v++){
        if (w[v] <= 0 || dist[v] >= INF) continue;
        vector<int> nb = adj[v];
        sort(nb.begin(), nb.end());
        nb.erase(unique(nb.begin(), nb.end()), nb.end());
        string key; key.reserve(nb.size() * 7);
        for (int x : nb){ key += to_string(x); key += ','; }
        auto it = idx.find(key);
        if (it == idx.end()){ idx[key] = mods.size(); mods.push_back({nb, w[v]}); }
        else mods[it->second].value += w[v];
    }

    // deadline of a gateway stand g = dist[g]: it must be protected by that step (before it
    // burns). Order modules by value density (value per gateway stand), commit greedily.
    vector<int> order(mods.size());
    for (size_t i = 0; i < mods.size(); i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b){
        // value_a / |gate_a|  vs  value_b / |gate_b|
        return (__int128)mods[a].value * (ll)mods[b].gate.size()
             > (__int128)mods[b].value * (ll)mods[a].gate.size();
    });

    int maxStep = 0;
    for (auto &m : mods) for (int g : m.gate) maxStep = max(maxStep, dist[g]);
    vector<int> used(maxStep + 2, 0);               // used[t] protections already booked at step t
    vector<vector<int>> plan(maxStep + 2);          // plan[t] = ids to protect at step t

    for (int oi : order){
        Mod &m = mods[oi];
        // try to reserve every gateway stand at a step <= its deadline (latest-fit)
        vector<pair<int,int>> book;                 // (step, id)
        // process gateway stands by earliest deadline first for a tighter fit
        vector<int> gs = m.gate;
        sort(gs.begin(), gs.end(), [&](int a, int b){ return dist[a] < dist[b]; });
        bool okAll = true;
        // local capacity view so we can roll back
        for (int g : gs){
            int dl = dist[g];
            if (dl < 1 || dl > maxStep){ okAll = false; break; }
            int placed = -1;
            for (int s = dl; s >= 1; s--){
                // count tentative bookings already made this module at step s
                int tent = 0; for (auto &pr : book) if (pr.first == s) tent++;
                if (used[s] + tent < B){ placed = s; break; }
            }
            if (placed < 0){ okAll = false; break; }
            book.push_back({placed, g});
        }
        if (!okAll) continue;
        for (auto &pr : book){ used[pr.first]++; plan[pr.first].push_back(pr.second); }
    }

    // emit schedule
    int K = 0; for (int s = 1; s <= maxStep; s++) if (!plan[s].empty()) K = s;
    printf("%d\n", K);
    for (int s = 1; s <= K; s++){
        printf("%d", (int)plan[s].size());
        for (int id : plan[s]) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
