// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// The "obvious" recipe: a metronome with an extreme tempo deviation only
// "bothers" the arms directly touching its own seat, so a coder's first
// instinct is to protect well-connected hub seats (many arms sharing the
// load) and hand the wild metronomes to lightly-connected seats (they only
// have one or two arms to overload anyway). Concretely: sort seats by
// ascending arm-count and hand out frequencies sorted by DESCENDING
// magnitude, so the biggest deviations land on the fewest-armed seats.
// This never reasons about which SIDE of the table's bottleneck cuts a seat
// sits on, so on a graph with a genuine structural bottleneck it can still
// leave one side net-heavy and the other net-light.
int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    vector<ll> f(n + 1);
    for (int i = 1; i <= n; i++) cin >> f[i];

    vector<int> nodes(n);
    for (int i = 0; i < n; i++) nodes[i] = i + 1;
    mt19937 rng(100);
    shuffle(nodes.begin(), nodes.end(), rng);          // deterministic tie-break
    stable_sort(nodes.begin(), nodes.end(), [&](int a, int b) {
        return adj[a].size() < adj[b].size();          // ascending degree
    });

    vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i + 1;
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return llabs(f[a]) > llabs(f[b]);               // descending |value|
    });

    vector<int> p(n + 1);
    for (int k = 0; k < n; k++) p[nodes[k]] = idx[k];

    for (int i = 1; i <= n; i++) cout << p[i] << (i < n ? ' ' : '\n');
    return 0;
}
