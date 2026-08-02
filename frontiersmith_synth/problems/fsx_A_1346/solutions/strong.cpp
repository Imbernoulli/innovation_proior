// TIER: strong
#include <bits/stdc++.h>
using namespace std;

// Same fixed deepest-first elimination order (the order itself is not the free
// variable this solution exploits -- the GUARD SET at each step is). The insight:
// instead of only ever checking "does ONE neighbor dominate v?" (greedy) or giving up
// straight to "guard with everyone" (trivial), run a small greedy SET-COVER search over
// v's live neighbors. Each live neighbor u's own live closed neighborhood may already
// shadow several of v's OTHER live neighbors -- a retract/corner-decomposition
// structure a single-dominator test cannot see, since no ONE neighbor need cover
// everything, just the union of a well-chosen few. This costs at most as much as the
// full-neighbor fallback (worst case it ends up picking every neighbor anyway) and is
// far cheaper whenever redundant coverage exists.
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n; long long m;
    cin >> n >> m;
    vector<vector<int>> adj(n);
    for (long long i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    for (auto &lst : adj) sort(lst.begin(), lst.end());

    vector<int> depth(n, -1);
    queue<int> q;
    depth[0] = 0; q.push(0);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : adj[u]) if (depth[v] == -1) { depth[v] = depth[u] + 1; q.push(v); }
    }

    vector<int> order(n);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (depth[a] != depth[b]) return depth[a] > depth[b];
        return a < b;
    });
    order.pop_back();

    vector<char> alive(n, 1);
    for (int v : order) {
        vector<int> liveNb;
        for (int u : adj[v]) if (alive[u]) liveNb.push_back(u);

        vector<char> uncovered(liveNb.size(), 1);
        int remaining = (int)liveNb.size();
        vector<char> picked(liveNb.size(), 0);
        vector<int> guards;

        while (remaining > 0) {
            int bestIdx = -1, bestGain = -1;
            for (int ci = 0; ci < (int)liveNb.size(); ci++) {
                if (picked[ci]) continue;
                int u = liveNb[ci];
                int gain = 0;
                for (int wi = 0; wi < (int)liveNb.size(); wi++) {
                    if (!uncovered[wi]) continue;
                    int w = liveNb[wi];
                    if (w == u || binary_search(adj[u].begin(), adj[u].end(), w)) gain++;
                }
                if (gain > bestGain) { bestGain = gain; bestIdx = ci; }
            }
            if (bestIdx == -1) break; // shouldn't happen: fall through, guards still valid so far
            int u = liveNb[bestIdx];
            picked[bestIdx] = 1;
            guards.push_back(u);
            for (int wi = 0; wi < (int)liveNb.size(); wi++) {
                if (!uncovered[wi]) continue;
                int w = liveNb[wi];
                if (w == u || binary_search(adj[u].begin(), adj[u].end(), w)) {
                    uncovered[wi] = 0; remaining--;
                }
            }
        }
        if (guards.empty() && !liveNb.empty()) guards = liveNb; // defensive fallback (never triggered)

        cout << v << ' ' << guards.size();
        for (int g : guards) cout << ' ' << g;
        cout << '\n';
        alive[v] = 0;
    }
    return 0;
}
