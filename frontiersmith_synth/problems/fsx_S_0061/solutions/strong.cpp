// TIER: strong
// Best-improvement local search with re-planning: each round, re-plan all shortest paths from
// the hub, take the current shortest-path-tree links (ranked by subtree latency), and among the
// top candidates cut the one whose removal yields the largest TRUE increase in total target
// latency while keeping every target connected. Re-planning each step chases successive
// bottlenecks and captures compounding detours -> stronger and structurally different from the
// one-pass subtree greedy.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, q, k;
vector<int> targets;
vector<int> eu, ev;
vector<ll> ew;
vector<vector<array<ll, 3>>> adj;

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

ll totalLatency(const vector<ll>& dist, bool& ok) {
    ll sum = 0;
    for (int t : targets) {
        if (dist[t] == LLONG_MAX) { ok = false; return 0; }
        sum += dist[t];
    }
    ok = true;
    return sum;
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

    vector<char> isT(n + 1, 0);
    for (int t : targets) isT[t] = 1;

    // cap the number of candidate evaluations per round to keep well within the time limit
    const int MAXCAND = 48;

    vector<char> rem(m + 1, 0);
    vector<int> out;

    for (int step = 0; step < k; step++) {
        vector<ll> dist; vector<int> pe;
        dij(rem, dist, pe);
        bool ok;
        ll cur = totalLatency(dist, ok);
        if (!ok) break;

        // subtree latency aggregation to rank candidate tree links
        vector<int> order;
        for (int v = 1; v <= n; v++) if (dist[v] != LLONG_MAX) order.push_back(v);
        sort(order.begin(), order.end(), [&](int a, int b) { return dist[a] > dist[b]; });
        vector<ll> subval(n + 1, 0);
        for (int v : order) if (isT[v]) subval[v] += dist[v];
        for (int v : order) {
            if (v == s || pe[v] == -1) continue;
            int idx = pe[v];
            int parent = (eu[idx] == v ? ev[idx] : eu[idx]);
            subval[parent] += subval[v];
        }
        vector<pair<ll, int>> cand;
        for (int v = 1; v <= n; v++) {
            if (v == s || pe[v] == -1) continue;
            if (rem[pe[v]]) continue;
            if (subval[v] > 0) cand.push_back({subval[v], pe[v]});
        }
        sort(cand.rbegin(), cand.rend());
        if (cand.size() > (size_t)MAXCAND) cand.resize(MAXCAND);

        // evaluate the true gain of each candidate cut with a real re-plan
        ll bestGain = 0;
        int bestE = -1;
        for (auto& pr : cand) {
            int idx = pr.second;
            if (rem[idx]) continue;
            rem[idx] = 1;
            vector<ll> d2; vector<int> pe2;
            dij(rem, d2, pe2);
            bool ok2;
            ll nv = totalLatency(d2, ok2);
            rem[idx] = 0;
            if (!ok2) continue; // would strand a target
            ll gain = nv - cur;
            if (gain > bestGain) { bestGain = gain; bestE = idx; }
        }
        if (bestE == -1) break;
        rem[bestE] = 1;
        out.push_back(bestE);
    }

    printf("%d\n", (int)out.size());
    for (int x : out) printf("%d\n", x);
    return 0;
}
