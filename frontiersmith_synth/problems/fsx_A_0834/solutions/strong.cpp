// TIER: strong
// Corridor-bundle reasoning: group edges by (u,v) -- a bundle is exactly the
// set of parallel channels between two hubs, INCLUDING any pricier bypass
// channel that shares the same endpoints. For each bundle, simulate what
// closing every one of its channels actually captures (the real, dynamic
// best-response value, not a static edge tally), then greedily knapsack the
// bundles by value-per-channel-closed until the budget runs out. This is the
// insight a per-edge greedy cannot see: an edge only pays off once its WHOLE
// bundle is shut, so ranking by bundle ROI (not edge value) is what matters.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll INF = (ll)4e18;

int n, m, K;
ll Bbudget;
vector<int> U, V; vector<ll> Cost, Cap;
vector<vector<pair<int,int>>> fwdAdj, revAdj;
vector<int> S_, T_; vector<ll> Vol_, Val_;

vector<ll> distToT(int t, const vector<char>& patrolled) {
    vector<ll> dist(n, INF);
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    dist[t] = 0; pq.push({0, t});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        for (auto& [eidx, v] : revAdj[u]) {
            ll w = Cost[eidx] + (patrolled[eidx] ? (ll)2000000 : 0);
            if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; pq.push({dist[v], v}); }
        }
    }
    return dist;
}

ll simulate(const vector<char>& patrolled) {
    ll total = 0;
    unordered_map<int, vector<ll>> memo;
    for (int i = 0; i < K; i++) {
        auto it = memo.find(T_[i]);
        if (it == memo.end()) it = memo.emplace(T_[i], distToT(T_[i], patrolled)).first;
        vector<ll>& dist = it->second;
        if (dist[S_[i]] >= INF) continue;
        int cur = S_[i]; int steps = 0;
        while (cur != T_[i] && steps <= n + 2) {
            steps++;
            int chosen = -1;
            for (auto& [eidx, v] : fwdAdj[cur]) {
                ll w = Cost[eidx] + (patrolled[eidx] ? (ll)2000000 : 0);
                if (dist[cur] == w + dist[v]) {
                    if (chosen == -1 || eidx < chosen) chosen = eidx;
                }
            }
            if (chosen == -1) break;
            if (patrolled[chosen]) {
                ll seize = min(Vol_[i], Cap[chosen]);
                total += seize * Val_[i];
                break;
            }
            cur = V[chosen];
        }
    }
    return total;
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
    S_.resize(K); T_.resize(K); Vol_.resize(K); Val_.resize(K);
    for (int i = 0; i < K; i++) cin >> S_[i] >> T_[i] >> Vol_[i] >> Val_[i];

    // group edges into bundles by (u,v)
    map<pair<int,int>, vector<int>> groups;
    for (int i = 0; i < m; i++) groups[{U[i], V[i]}].push_back(i);

    struct Bundle { vector<int> eidx; ll value; ll cost; };
    vector<Bundle> bundles;
    for (auto& [key, list] : groups) {
        vector<char> pat(m, 0);
        for (int e : list) pat[e] = 1;
        ll val = simulate(pat);
        bundles.push_back({list, val, (ll)list.size()});
    }

    sort(bundles.begin(), bundles.end(), [](const Bundle& a, const Bundle& b) {
        // rank by value-per-channel ROI, descending
        return (double)a.value * b.cost > (double)b.value * a.cost;
    });

    vector<int> chosen;
    ll used = 0;
    for (auto& bd : bundles) {
        if (bd.value <= 0) continue;
        if (used + bd.cost <= Bbudget) {
            used += bd.cost;
            for (int e : bd.eidx) chosen.push_back(e);
        }
    }
    // fill any leftover budget with the highest-value still-unpatrolled bundle
    // that partially fits is not attempted (partial closure is worthless by
    // construction) -- leftover budget is simply left unused.

    sort(chosen.begin(), chosen.end());
    cout << chosen.size() << "\n";
    for (size_t i = 0; i < chosen.size(); i++) cout << chosen[i] << " \n"[i+1==chosen.size()];
    if (chosen.empty()) cout << "\n";
    return 0;
}
