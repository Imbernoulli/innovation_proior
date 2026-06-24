#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    long long S;
    if (!(cin >> n >> m >> S)) return 0;

    // edges[u] = list of (v, color, fare)
    vector<vector<array<long long,3>>> edges(n + 1);
    // colors are 1..C; we compress per-node arrival operator state.
    // State = (node, last_color). last_color = 0 means "no edge used yet" (start).
    // We index states. To keep it bounded we only create states reachable through edges.

    // Collect all colors to map them; but a state's "last_color" is the color of the
    // edge we arrived on. We use the raw color value as part of the key.
    // dist over (node, color). Use a hash map keyed by node*BIG + color, but colors
    // can be large; instead store per node a map<color,dist>.
    for (int i = 0; i < m; i++) {
        long long u, v, c, w;
        cin >> u >> v >> c >> w;
        edges[u].push_back({v, c, w});
    }

    // best[node] : map from last_color -> min cost arriving at node having last
    // traversed an edge of that color.
    vector<unordered_map<long long,long long>> best(n + 1);

    // Priority queue of (cost, node, last_color).
    priority_queue<array<long long,3>, vector<array<long long,3>>, greater<array<long long,3>>> pq;

    // Start at node 1 with last_color = 0 (a sentinel meaning "no operator yet").
    // 0 is never used as a real color (real colors are >= 1).
    best[1][0] = 0;
    pq.push({0, 1, 0});

    long long answer = -1;

    while (!pq.empty()) {
        auto top = pq.top(); pq.pop();
        long long d = top[0], u = top[1], lc = top[2];
        auto it = best[u].find(lc);
        if (it == best[u].end() || it->second != d) continue; // stale
        if (u == n) { answer = d; break; }
        for (auto &e : edges[u]) {
            long long v = e[0], c = e[1], w = e[2];
            // surcharge only if we already used an edge (lc != 0) and color changes.
            long long nd = d + w + ((lc != 0 && c != lc) ? S : 0);
            auto vit = best[v].find(c);
            if (vit == best[v].end() || nd < vit->second) {
                best[v][c] = nd;
                pq.push({nd, v, c});
            }
        }
    }

    cout << answer << "\n";
    return 0;
}
