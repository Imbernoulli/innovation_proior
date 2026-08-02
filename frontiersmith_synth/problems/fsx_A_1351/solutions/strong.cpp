// TIER: strong
// The insight has two parts, both about NOT reaching for what looks locally
// best:
//  (1) never mirror (pair[] = 0 everywhere) -- mirroring is the
//      achievement-game reflex ("an extra move never hurts") that backfires
//      in an avoidance game, where every extra claim only adds risk.
//  (2) never grab the highest-connectivity ("hub") vertices for yourself
//      either, for the SAME reason: owning a heavily-entangled vertex makes
//      every one of its poison-neighbours permanently dangerous for you
//      specifically, one-sidedly -- those same neighbours stay perfectly
//      safe for you as long as the hub ends up belonging to Rival instead.
//      Rival's own fixed algorithm is a degree-seeking greedy, so if you
//      simply decline to compete for the few most-entangled vertices, Rival
//      obligingly absorbs that risk itself.
// So: split vertices into "ordinary" (poison-degree <= 2) and "hub-like"
// (degree > 2). Build a greedy independent set (no two chosen vertices share
// a poison edge) from the ordinary vertices only, claim that first, then any
// leftover ordinary vertices, and defer the hub-like vertices to dead last.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K;
    scanf("%d", &K);
    for (int a = 0; a < K; a++) {
        int n, m, first;
        scanf("%d %d %d", &n, &m, &first);
        vector<vector<int>> adj(n + 1);
        for (int j = 0; j < m; j++) {
            int u, v; scanf("%d %d", &u, &v);
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        vector<int> ordinary, hubby;
        for (int v = 1; v <= n; v++) {
            if ((int)adj[v].size() > 2) hubby.push_back(v);
            else ordinary.push_back(v);
        }
        sort(ordinary.begin(), ordinary.end(), [&](int x, int y) {
            if (adj[x].size() != adj[y].size()) return adj[x].size() < adj[y].size();
            return x < y;
        });
        vector<char> inI(n + 1, 0);
        vector<int> prioList;
        for (int v : ordinary) {
            bool ok = true;
            for (int nb : adj[v]) if (inI[nb]) { ok = false; break; }
            if (ok) { inI[v] = 1; prioList.push_back(v); }
        }
        for (int v : ordinary) if (!inI[v]) prioList.push_back(v);
        for (int v : hubby) prioList.push_back(v);       // dead last: let Rival take these

        for (int v = 1; v <= n; v++) printf("%d ", 0);   // no reactive mirroring
        printf("\n");
        for (int v : prioList) printf("%d ", v);
        printf("\n");
    }
    return 0;
}
