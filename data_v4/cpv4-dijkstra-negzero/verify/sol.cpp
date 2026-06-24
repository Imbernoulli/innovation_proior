#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m, s;
    if (!(cin >> n >> m >> s)) return 0;

    vector<vector<pair<int,long long>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
    }

    const long long NEG_INF = LLONG_MIN / 4;   // "no path yet" / unreachable
    const long long POS_INF = LLONG_MAX / 4;   // bottleneck of the zero-edge path at the source

    vector<long long> best(n + 1, NEG_INF);
    vector<char> done(n + 1, 0);
    best[s] = POS_INF;                         // path of length 0: min over empty edge set = +inf

    // max-min Dijkstra: pop the node with the currently largest bottleneck.
    priority_queue<pair<long long,int>> pq;    // (bottleneck, node), max-heap
    pq.push({best[s], s});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (done[u]) continue;                 // stale / already finalized
        if (d != best[u]) continue;            // outdated entry
        done[u] = 1;
        for (auto [v, w] : adj[u]) {
            long long cand = min(best[u], w);  // weakest link along this path
            if (cand > best[v]) {
                best[v] = cand;
                pq.push({best[v], v});
            }
        }
    }

    for (int v = 1; v <= n; v++) {
        if (best[v] == NEG_INF) cout << "UNREACHABLE";
        else if (v == s && best[v] == POS_INF) cout << "INF";  // source with no self-route
        else cout << best[v];
        cout << "\n";
    }
    return 0;
}
