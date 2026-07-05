// TIER: trivial
// Best CONSTANT labeling x_u = c* for all u -- exactly the checker's baseline B.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

struct E { int u, v; ll w; u64 mask; };

int main() {
    int n, m, q, H;
    if (scanf("%d %d %d %d", &n, &m, &q, &H) != 4) return 0;
    vector<E> edges(m);
    for (int i = 0; i < m; i++) {
        int u, v, k; ll w;
        scanf("%d %d %lld %d", &u, &v, &w, &k);
        u64 mask = 0;
        for (int j = 0; j < k; j++) { int f; scanf("%d", &f); mask |= (1ULL << f); }
        edges[i] = {u, v, w, mask};
    }
    vector<int> hubNode(H);
    vector<ll> hubBonus(H);
    for (int i = 0; i < H; i++) scanf("%d %lld", &hubNode[i], &hubBonus[i]);

    vector<char> isHub(n + 1, 0);
    unordered_map<int, u64> hubMaskOf;
    for (int i = 0; i < H; i++) { isHub[hubNode[i]] = 1; hubMaskOf[hubNode[i]] = 0ULL; }

    vector<ll> forbW(q, 0), forbHub(q, 0);
    for (auto& e : edges) {
        for (int r = 0; r < q; r++) if (e.mask >> r & 1ULL) forbW[r] += e.w;
        if (e.u >= 1 && e.u <= n && isHub[e.u]) hubMaskOf[e.u] |= e.mask;
        if (e.v >= 1 && e.v <= n && isHub[e.v]) hubMaskOf[e.v] |= e.mask;
    }
    for (int i = 0; i < H; i++) {
        u64 hm = hubMaskOf[hubNode[i]];
        for (int r = 0; r < q; r++) if (hm >> r & 1ULL) forbHub[r] += hubBonus[i];
    }
    int rstar = 0; ll best = LLONG_MAX;
    for (int r = 0; r < q; r++) {
        ll v = forbW[r] + forbHub[r];
        if (v < best) { best = v; rstar = r; }
    }
    int inv2 = (q + 1) / 2;
    int cstar = (int)((ll)rstar * inv2 % q);
    for (int i = 1; i <= n; i++) printf("%d ", cstar);
    printf("\n");
    return 0;
}
