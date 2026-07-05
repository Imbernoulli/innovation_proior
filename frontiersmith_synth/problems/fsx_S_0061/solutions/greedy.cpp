// TIER: greedy
// Subtree-count greedy (one-pass, no re-planning): build a single shortest-path tree from the
// hub, score each tree link by the total latency of the target streams routed through it, then
// cut the highest-scoring links (skipping any cut that would strand a target), up to k.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, q, k;
vector<int> targets;
vector<int> eu, ev;
vector<ll> ew;
vector<vector<array<ll, 3>>> adj;

// Dijkstra; fills dist and parent-edge pe. Returns dist vector via reference.
void dij(const vector<char>& rem, vector<ll>& dist, vector<int>& pe) {
    dist.assign(n + 1, LLONG_MAX);
    pe.assign(n + 1, -1);
    priority_queue<pair<ll, int>, vector<pair<ll, int>>, greater<>> pq;
    dist[s] = 0;
    pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;
        for (auto& e : adj[u]) {
            int v = (int)e[0];
            ll w = e[1];
            int idx = (int)e[2];
            if (rem[idx]) continue;
            if (d + w < dist[v]) { dist[v] = d + w; pe[v] = idx; pq.push({dist[v], v}); }
        }
    }
}

bool allTargetsReachable(const vector<char>& rem) {
    vector<char> vis(n + 1, 0);
    queue<int> Q;
    Q.push(s); vis[s] = 1;
    while (!Q.empty()) {
        int u = Q.front(); Q.pop();
        for (auto& e : adj[u]) {
            int v = (int)e[0]; int idx = (int)e[2];
            if (rem[idx] || vis[v]) continue;
            vis[v] = 1; Q.push(v);
        }
    }
    for (int t : targets) if (!vis[t]) return false;
    return true;
}

int main() {
    scanf("%d %d %d %d %d", &n, &m, &s, &q, &k);
    targets.resize(q);
    for (int i = 0; i < q; i++) scanf("%d", &targets[i]);
    eu.resize(m + 1); ev.resize(m + 1); ew.resize(m + 1);
    adj.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u, v; ll w;
        scanf("%d %d %lld", &u, &v, &w);
        eu[i] = u; ev[i] = v; ew[i] = w;
        adj[u].push_back({(ll)v, w, (ll)i});
        adj[v].push_back({(ll)u, w, (ll)i});
    }

    vector<char> rem(m + 1, 0);
    vector<ll> dist; vector<int> pe;
    dij(rem, dist, pe);

    // Post-order aggregate: subval[v] = latency of targets in v's subtree (via the tree edge
    // pe[v]). Order nodes by decreasing dist so children are processed before parents.
    vector<char> isT(n + 1, 0);
    for (int t : targets) isT[t] = 1;
    vector<int> order;
    for (int v = 1; v <= n; v++) if (dist[v] != LLONG_MAX) order.push_back(v);
    sort(order.begin(), order.end(), [&](int a, int b) { return dist[a] > dist[b]; });

    vector<ll> subval(n + 1, 0);
    for (int v : order) if (isT[v]) subval[v] += dist[v];
    // propagate to parent
    for (int v : order) {
        if (v == s || pe[v] == -1) continue;
        int idx = pe[v];
        int parent = (eu[idx] == v ? ev[idx] : eu[idx]);
        subval[parent] += subval[v];
    }

    // candidate tree links: pe[v] for v != s, score = subval[v]
    vector<pair<ll, int>> cand; // (score, edgeIndex)
    for (int v = 1; v <= n; v++) {
        if (v == s || pe[v] == -1) continue;
        if (subval[v] > 0) cand.push_back({subval[v], pe[v]});
    }
    sort(cand.rbegin(), cand.rend());

    vector<int> out;
    for (auto& pr : cand) {
        if ((int)out.size() >= k) break;
        int idx = pr.second;
        if (rem[idx]) continue;
        rem[idx] = 1;
        if (!allTargetsReachable(rem)) { rem[idx] = 0; continue; }
        out.push_back(idx);
    }

    printf("%d\n", (int)out.size());
    for (int x : out) printf("%d\n", x);
    return 0;
}
