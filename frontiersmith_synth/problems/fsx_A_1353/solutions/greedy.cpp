// TIER: greedy
// The obvious first instinct: repeatedly fire whichever CURRENTLY UNSTABLE
// pile has the largest raw chip count, one firing at a time, until the
// budget is spent. Blind to a pile's degree (and hence to how many firings
// it actually needs, or how much criticality-weighted risk finishing it
// would remove) -- it just chases the biggest number on screen.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    ll n64, m64, K;
    cin >> n64 >> m64 >> K;
    int n = (int)n64, m = (int)m64;
    int sink = n;
    vector<vector<pair<int,ll>>> adj(n + 1);
    vector<ll> deg(n, 0);
    for (int i = 0; i < m; i++) {
        ll a, b, w;
        cin >> a >> b >> w;
        adj[a].push_back({(int)b, w});
        adj[b].push_back({(int)a, w});
        if (a < n) deg[a] += w;
        if (b < n) deg[b] += w;
    }
    vector<ll> h(n);
    for (int i = 0; i < n; i++) cin >> h[i];
    // criticality weights c[] are in the input too, but this heuristic
    // never looks at them -- that's exactly the trap.

    vector<ll> height = h;
    vector<ll> f(n, 0);
    priority_queue<pair<ll,int>> pq;
    for (int v = 0; v < n; v++) if (height[v] >= deg[v]) pq.push({height[v], v});

    ll budget = K;
    while (budget > 0 && !pq.empty()) {
        auto top = pq.top(); pq.pop();
        ll ht = top.first; int v = top.second;
        if (ht != height[v] || height[v] < deg[v]) continue; // stale entry
        height[v] -= deg[v];
        f[v]++;
        budget--;
        for (auto& pr : adj[v]) {
            int u = pr.first; ll w = pr.second;
            if (u == sink) continue;
            height[u] += w;
            if (height[u] >= deg[u]) pq.push({height[u], u});
        }
        if (height[v] >= deg[v]) pq.push({height[v], v});
    }

    for (int i = 0; i < n; i++) cout << f[i] << " \n"[i + 1 == n];
    return 0;
}
