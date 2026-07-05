// TIER: greedy
// Single-move local search from the contiguous split. Flipping a crew changes the cut by
// (weight to own habitat) - (weight to opposite habitat): its same-side edges become cut,
// its cross edges stop being cut. Repeatedly flip every crew with positive such gain that
// keeps the headcount band feasible, over a few passes (recomputed each pass).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m, L;
    if (scanf("%d %d %d", &n, &m, &L) != 3) return 0;
    vector<vector<pair<int,ll>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    // side: 0 = Habitat A, 1 = Habitat B
    vector<int> side(n + 1, 0);
    int h = n / 2, a = h;
    for (int i = 1; i <= n; i++) side[i] = (i <= h) ? 0 : 1;

    auto balanceOkAfterFlip = [&](int node) {
        int na = a + (side[node] == 1 ? 1 : -1); // moving into/out of A
        int nb = n - na;
        return llabs((ll)na - (ll)nb) <= L;
    };

    vector<ll> deg_same(n + 1, 0), deg_opp(n + 1, 0);
    auto recompute = [&]() {
        fill(deg_same.begin(), deg_same.end(), 0);
        fill(deg_opp.begin(), deg_opp.end(), 0);
        for (int u = 1; u <= n; u++)
            for (auto& pr : adj[u]) {
                if (side[pr.first] == side[u]) deg_same[u] += pr.second;
                else deg_opp[u] += pr.second;
            }
    };

    for (int pass = 0; pass < 8; pass++) {
        recompute();
        bool moved = false;
        for (int u = 1; u <= n; u++) {
            ll gain = deg_same[u] - deg_opp[u];   // cut change if we flip u
            if (gain > 0 && balanceOkAfterFlip(u)) {
                a += (side[u] == 1 ? 1 : -1);
                side[u] ^= 1;
                moved = true;
            }
        }
        if (!moved) break;
    }

    vector<int> A;
    for (int i = 1; i <= n; i++) if (side[i] == 0) A.push_back(i);
    printf("%d\n", (int)A.size());
    for (int x : A) printf("%d\n", x);
    return 0;
}
