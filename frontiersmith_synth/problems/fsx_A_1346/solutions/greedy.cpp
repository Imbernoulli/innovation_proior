// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// Same fixed deepest-first elimination order as trivial. At each step, first try the
// TEXTBOOK single-guard "corner" test (does some ONE live neighbor's closed
// neighborhood already contain everything v needs?) -- this is the classical
// dismantlable-graph recognition rule an average coder reaches for first. Whenever no
// single dominator exists, this falls back to guarding with every live neighbor (same
// as trivial for that step) -- it never tries a team of 2+ carefully chosen guards, so
// it cannot see the redundancy the planted gadgets rely on.
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

        int dominator = -1;
        for (int u : liveNb) {
            // does u's live closed neighborhood cover v's live closed neighborhood?
            bool ok = true;
            for (int w : liveNb) {
                if (w == u) continue;
                if (!binary_search(adj[u].begin(), adj[u].end(), w)) { ok = false; break; }
            }
            if (ok) { dominator = u; break; }
        }

        vector<int> guards;
        if (dominator != -1) guards.push_back(dominator);
        else guards = liveNb; // give up on cleverness, guard with everyone live

        cout << v << ' ' << guards.size();
        for (int g : guards) cout << ' ' << g;
        cout << '\n';
        alive[v] = 0;
    }
    return 0;
}
