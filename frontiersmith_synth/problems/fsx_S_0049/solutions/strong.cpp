// TIER: strong
// Best-improvement local search: repeatedly re-plan the fastest route; among its
// still-open, still-affordable roads, close the one whose removal yields the largest
// true increase in s-t distance. Re-planning each step spreads cuts across successive
// bottlenecks -> stronger than one-shot greedy.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, s, t;
ll B;
vector<int> eu, ev;
vector<ll> ew, ec;
vector<vector<array<ll, 3>>> adj;

ll dij(const vector<char>& rem, vector<int>& pe) {
    vector<ll> dist(n + 1, LLONG_MAX);
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
    return dist[t];
}

int main() {
    scanf("%d %d %d %d %lld", &n, &m, &s, &t, &B);
    eu.resize(m + 1); ev.resize(m + 1); ew.resize(m + 1); ec.resize(m + 1);
    adj.assign(n + 1, {});
    for (int i = 1; i <= m; i++) {
        int u, v; ll w, c;
        scanf("%d %d %lld %lld", &u, &v, &w, &c);
        eu[i] = u; ev[i] = v; ew[i] = w; ec[i] = c;
        adj[u].push_back({(ll)v, w, (ll)i});
        adj[v].push_back({(ll)u, w, (ll)i});
    }

    vector<char> rem(m + 1, 0);
    ll used = 0;
    vector<int> out;

    while (true) {
        vector<int> pe;
        ll base = dij(rem, pe);
        if (base == LLONG_MAX) break;
        vector<int> pathe;
        int cur = t;
        while (cur != s && pe[cur] != -1) {
            int idx = pe[cur];
            pathe.push_back(idx);
            cur = (eu[idx] == cur ? ev[idx] : eu[idx]);
        }
        ll bestgain = 0;
        int beste = -1;
        for (int idx : pathe) {
            if (rem[idx]) continue;
            if (used + ec[idx] > B) continue;
            rem[idx] = 1;
            vector<int> tmp;
            ll d2 = dij(rem, tmp);
            rem[idx] = 0;
            if (d2 == LLONG_MAX) continue;
            ll gain = d2 - base;
            if (gain > bestgain) { bestgain = gain; beste = idx; }
        }
        if (beste == -1) break;
        rem[beste] = 1;
        used += ec[beste];
        out.push_back(beste);
    }

    printf("%d\n", (int)out.size());
    for (int x : out) printf("%d\n", x);
    return 0;
}
