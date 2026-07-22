// TIER: greedy
// "Patrol the current max-flow edges": compute each shipment's cheapest
// (zero-patrol) route, tally the total value crossing every EDGE, and patrol
// the top-B edges by that tally. This is the standard textbook marginal-
// value interdiction heuristic -- it reasons about individual edges, never
// about whether an edge's parallel channels (its bundle) get closed too.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll INF = (ll)4e18;

int n, m, K;
ll Bbudget;
vector<int> U, V; vector<ll> Cost, Cap;
vector<vector<pair<int,int>>> fwdAdj, revAdj; // (edgeIdx, node)

vector<ll> distToT(int t) {
    vector<ll> dist(n, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[t] = 0; pq.push({0, t});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto& [eidx, v] : revAdj[u]) {
            ll w = Cost[eidx];
            if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; pq.push({dist[v], v}); }
        }
    }
    return dist;
}

int main() {
    cin >> n >> m >> K >> Bbudget;
    U.resize(m); V.resize(m); Cost.resize(m); Cap.resize(m);
    fwdAdj.assign(n, {}); revAdj.assign(n, {});
    for (int i = 0; i < m; i++) {
        cin >> U[i] >> V[i] >> Cost[i] >> Cap[i];
        fwdAdj[U[i]].push_back({i, V[i]});
        revAdj[V[i]].push_back({i, U[i]});
    }
    vector<int> S(K), T(K); vector<ll> Vol(K), Val(K);
    for (int i = 0; i < K; i++) cin >> S[i] >> T[i] >> Vol[i] >> Val[i];

    vector<ll> tally(m, 0);
    unordered_map<int, vector<ll>> memo;
    for (int i = 0; i < K; i++) {
        auto it = memo.find(T[i]);
        if (it == memo.end()) it = memo.emplace(T[i], distToT(T[i])).first;
        vector<ll>& dist = it->second;
        if (dist[S[i]] >= INF) continue;
        int cur = S[i]; int steps = 0;
        ll contrib = Vol[i] * Val[i];
        while (cur != T[i] && steps <= n + 2) {
            steps++;
            int chosen = -1;
            for (auto& [eidx, v] : fwdAdj[cur]) {
                if (dist[cur] == Cost[eidx] + dist[v]) {
                    if (chosen == -1 || eidx < chosen) chosen = eidx;
                }
            }
            if (chosen == -1) break;
            tally[chosen] += contrib;
            cur = V[chosen];
        }
    }

    vector<int> idx(m);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (tally[a] != tally[b]) return tally[a] > tally[b];
        return a < b;
    });

    int take = (int)min((ll)m, Bbudget);
    cout << take << "\n";
    for (int i = 0; i < take; i++) cout << idx[i] << " \n"[i+1==take];
    if (take == 0) cout << "\n";
    return 0;
}
