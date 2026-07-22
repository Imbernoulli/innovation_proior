// TIER: strong
// The insight: the exchange list is a MIXTURE of within-clan chatter and
// scattered cross-clan bridges. Un-mix it FIRST (group guests by the given
// clan label), run the same weight-descending chain construction SEPARATELY
// inside each clan (so it never has the chance to interleave clans), then
// decide the order of the clan-blocks with a small greedy TSP-style pass over
// the aggregated inter-clan exchange weight -- only then concatenate.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct DSU {
    vector<int> p;
    void init(int n) { p.resize(n + 1); for (int i = 0; i <= n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; }
};

// Build one contiguous chain over `nodes` using only edges whose BOTH
// endpoints lie in `nodes` (weight-descending, degree cap 2, no cycles),
// then stitch leftover sub-chains together (id order) -- identical local
// heuristic to the flat greedy, just scoped to one clan's guest set.
vector<int> chainWithin(const vector<int>& nodes, const vector<array<int,3>>& edges) {
    if (nodes.empty()) return {};
    unordered_set<int> inSet(nodes.begin(), nodes.end());
    int maxId = *max_element(nodes.begin(), nodes.end());
    vector<int> deg(maxId + 1, 0);
    vector<array<int,2>> nbr(maxId + 1, {0, 0});
    DSU dsu; dsu.init(maxId);

    vector<array<int,3>> local;
    for (auto& e : edges) {
        if (e[0] > maxId || e[1] > maxId) continue;
        if (inSet.count(e[0]) && inSet.count(e[1])) local.push_back(e);
    }
    sort(local.begin(), local.end(), [](const array<int,3>& a, const array<int,3>& b) {
        if (a[2] != b[2]) return a[2] > b[2];
        if (a[0] != b[0]) return a[0] < b[0];
        return a[1] < b[1];
    });
    for (auto& e : local) {
        int u = e[0], v = e[1];
        if (u == v || deg[u] >= 2 || deg[v] >= 2) continue;
        int ru = dsu.find(u), rv = dsu.find(v);
        if (ru == rv) continue;
        dsu.p[ru] = rv;
        nbr[u][deg[u]++] = v;
        nbr[v][deg[v]++] = u;
    }
    vector<int> sortedNodes = nodes;
    sort(sortedNodes.begin(), sortedNodes.end());
    unordered_set<int> visited;
    vector<int> chain;
    chain.reserve(nodes.size());
    for (int g : sortedNodes) {
        if (visited.count(g) || deg[g] == 2) continue;
        int prev = -1, cur = g;
        while (cur != -1 && !visited.count(cur)) {
            visited.insert(cur);
            chain.push_back(cur);
            int nxt = -1;
            for (int k = 0; k < deg[cur]; k++) if (nbr[cur][k] != prev) { nxt = nbr[cur][k]; break; }
            prev = cur;
            cur = nxt;
        }
    }
    for (int g : sortedNodes) if (!visited.count(g)) { chain.push_back(g); visited.insert(g); }
    return chain;
}

int main() {
    int N, K, M;
    if (!(cin >> N >> K >> M)) return 0;
    vector<int> phase(N + 1);
    for (int i = 1; i <= N; i++) cin >> phase[i];

    vector<array<int,3>> edges(M);
    for (int i = 0; i < M; i++) cin >> edges[i][0] >> edges[i][1] >> edges[i][2];

    // ---- un-mix: group guests by clan ----
    vector<vector<int>> members(K + 1);
    for (int g = 1; g <= N; g++) members[phase[g]].push_back(g);

    // ---- aggregate inter-clan weight for block-order planning ----
    vector<vector<ll>> W(K + 1, vector<ll>(K + 1, 0));
    for (auto& e : edges) {
        int pu = phase[e[0]], pv = phase[e[1]];
        if (pu != pv) { W[pu][pv] += e[2]; W[pv][pu] += e[2]; }
    }

    // ---- build a within-clan chain (local Pettis-Hansen) for every clan ----
    vector<vector<int>> block(K + 1);
    for (int k = 1; k <= K; k++) block[k] = chainWithin(members[k], edges);

    // ---- greedy nearest-neighbour order over the K clan-blocks, guided by
    //      total inter-clan exchange weight (so heavily-bridged clans end up
    //      adjacent, keeping the unavoidable cross-clan cost cheap) ----
    vector<int> clanOrder;
    vector<char> used(K + 1, 0);
    int start = 1;
    ll bestDeg = -1;
    for (int k = 1; k <= K; k++) {
        if (members[k].empty()) continue;
        ll s = 0; for (int l = 1; l <= K; l++) s += W[k][l];
        if (s > bestDeg) { bestDeg = s; start = k; }
    }
    clanOrder.push_back(start); used[start] = 1;
    while ((int)clanOrder.size() < K) {
        int front = clanOrder.front(), back = clanOrder.back();
        int bestK = -1; ll bestW = -1; bool atBack = true;
        for (int k = 1; k <= K; k++) {
            if (used[k] || members[k].empty()) continue;
            if (W[back][k] > bestW) { bestW = W[back][k]; bestK = k; atBack = true; }
            if (W[front][k] > bestW) { bestW = W[front][k]; bestK = k; atBack = false; }
        }
        if (bestK == -1) { for (int k = 1; k <= K; k++) if (!used[k] && !members[k].empty()) { bestK = k; atBack = true; break; } }
        if (bestK == -1) break;
        used[bestK] = 1;
        if (atBack) clanOrder.push_back(bestK); else clanOrder.insert(clanOrder.begin(), bestK);
    }
    for (int k = 1; k <= K; k++) if (!used[k] && !members[k].empty()) clanOrder.push_back(k);

    vector<int> result;
    result.reserve(N);
    for (int k : clanOrder) for (int g : block[k]) result.push_back(g);

    for (size_t i = 0; i < result.size(); i++) cout << result[i] << (i + 1 < result.size() ? ' ' : '\n');
    return 0;
}
