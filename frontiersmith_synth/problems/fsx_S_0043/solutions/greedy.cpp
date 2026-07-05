// TIER: greedy
// Nearest-neighbour growth: extend from the current node to its cheapest unvisited
// cap-respecting neighbour, forming a path-like mesh; on a dead end, attach the cheapest
// available link from any covered node to an uncovered one. Falls back to the chain if it
// cannot span. This is a genuine heuristic but weaker than a globally optimised tree.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<int> cap(N + 1);
    for (int i = 1; i <= N; i++) scanf("%d", &cap[i]);
    vector<vector<pair<int,int>>> adj(N + 1); // (cost, neighbor)
    for (int k = 0; k < M; k++) {
        int u, v, w; scanf("%d %d %d", &u, &v, &w);
        adj[u].push_back({w, v});
        adj[v].push_back({w, u});
    }
    for (int i = 1; i <= N; i++) sort(adj[i].begin(), adj[i].end());

    vector<char> vis(N + 1, 0);
    vector<int> deg(N + 1, 0);
    vector<pair<int,int>> chosen;
    int cnt = 1, cur = 1;
    vis[1] = 1;
    vector<int> covered = {1};

    while (cnt < N) {
        // try to extend the path from cur to its cheapest unvisited neighbour
        int best = -1;
        if (deg[cur] < cap[cur]) {
            for (auto &pr : adj[cur]) {
                int nb = pr.second;
                if (!vis[nb] && deg[nb] < cap[nb]) { best = nb; break; }
            }
        }
        if (best != -1) {
            chosen.push_back({cur, best});
            deg[cur]++; deg[best]++;
            vis[best] = 1; covered.push_back(best); cnt++; cur = best;
            continue;
        }
        // dead end: cheapest link from any covered node with a free collar to an uncovered node
        int bu = -1, bw = -1, bc = INT_MAX;
        for (int u : covered) {
            if (deg[u] >= cap[u]) continue;
            for (auto &pr : adj[u]) {
                int nb = pr.second;
                if (vis[nb] || deg[nb] >= cap[nb]) continue;
                if (pr.first < bc) { bc = pr.first; bu = u; bw = nb; }
                break; // adj sorted: first free unvisited is cheapest for this u
            }
        }
        if (bu != -1) {
            chosen.push_back({bu, bw});
            deg[bu]++; deg[bw]++;
            vis[bw] = 1; covered.push_back(bw); cnt++; cur = bw;
            continue;
        }
        break; // genuinely stuck
    }

    if (cnt == N) {
        printf("%d\n", (int)chosen.size());
        for (auto &pr : chosen) printf("%d %d\n", pr.first, pr.second);
    } else {
        printf("%d\n", N - 1); // safe fallback: guaranteed chain
        for (int i = 1; i <= N - 1; i++) printf("%d %d\n", i, i + 1);
    }
    return 0;
}
