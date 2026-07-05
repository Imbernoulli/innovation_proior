// TIER: greedy
// Per-node best response (ignores hub bonuses). Start all-zero, two sweeps:
// each node picks the label that clears the most incident edge weight given
// its neighbours' current labels.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

int main() {
    int n, m, q, H;
    if (scanf("%d %d %d %d", &n, &m, &q, &H) != 4) return 0;
    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    vector<u64> emask(m);
    vector<int> deg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v, k; ll w;
        scanf("%d %d %lld %d", &u, &v, &w, &k);
        u64 mask = 0;
        for (int j = 0; j < k; j++) { int f; scanf("%d", &f); mask |= (1ULL << f); }
        eu[i] = u; ev[i] = v; ew[i] = w; emask[i] = mask;
        deg[u]++; deg[v]++;
    }
    for (int i = 0; i < H; i++) { int h; ll g; scanf("%d %lld", &h, &g); }

    // CSR adjacency: (neighbor, w, mask)
    vector<int> start(n + 2, 0);
    for (int i = 1; i <= n; i++) start[i + 1] = start[i] + deg[i];
    int tot = start[n + 1];
    vector<int> adjNb(tot); vector<ll> adjW(tot); vector<u64> adjM(tot);
    vector<int> cur(n + 2); for (int i = 1; i <= n; i++) cur[i] = start[i];
    for (int i = 0; i < m; i++) {
        int u = eu[i], v = ev[i];
        int p = cur[u]++; adjNb[p] = v; adjW[p] = ew[i]; adjM[p] = emask[i];
        int p2 = cur[v]++; adjNb[p2] = u; adjW[p2] = ew[i]; adjM[p2] = emask[i];
    }

    vector<int> x(n + 1, 0);
    for (int pass = 0; pass < 2; pass++) {
        for (int u = 1; u <= n; u++) {
            ll bestScore = -1; int bestL = 0;
            for (int L = 0; L < q; L++) {
                ll s = 0;
                for (int p = start[u]; p < start[u + 1]; p++) {
                    int r = (L + x[adjNb[p]]) % q;
                    if (!((adjM[p] >> r) & 1ULL)) s += adjW[p];
                }
                if (s > bestScore) { bestScore = s; bestL = L; }
            }
            x[u] = bestL;
        }
    }
    for (int i = 1; i <= n; i++) printf("%d ", x[i]);
    printf("\n");
    return 0;
}
