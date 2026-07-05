// TIER: greedy
// Minimum-cost degree-bounded spanning tree via Kruskal on the non-backbone
// edges (degree capped at D-2 so the backbone can always finish the tree),
// then backbone completion. Grabs the ultra-cheap trap chain -> low cost but a
// long fragile chain rooted at the hub -> HIGH stress. Beats trivial on cost.
#include <bits/stdc++.h>
using namespace std;

int par[6005];
int find(int x){ while(par[x]!=x){ par[x]=par[par[x]]; x=par[x]; } return x; }

int main() {
    int n, m, D; long long L;
    if (scanf("%d %d %d %lld", &n, &m, &D, &L) != 4) return 0;
    vector<int> eu(m + 1), ev(m + 1);
    vector<long long> ew(m + 1);
    for (int j = 1; j <= m; j++) scanf("%d %d %lld", &eu[j], &ev[j], &ew[j]);

    for (int i = 0; i < n; i++) par[i] = i;
    vector<int> deg(n, 0);
    vector<int> chosen;

    vector<int> order;
    for (int j = n; j <= m; j++) order.push_back(j);   // non-backbone edges
    sort(order.begin(), order.end(), [&](int a, int b){ return ew[a] < ew[b]; });

    int capk = D - 2; if (capk < 1) capk = 1;
    for (int j : order) {
        int u = eu[j], v = ev[j];
        if (find(u) != find(v) && deg[u] < capk && deg[v] < capk) {
            par[find(u)] = find(v);
            deg[u]++; deg[v]++;
            chosen.push_back(j);
        }
    }
    // backbone completion (edge index i connects i-1,i); reserved degree keeps <=D
    for (int i = 1; i <= n - 1; i++) {
        if (find(i - 1) != find(i)) {
            par[find(i - 1)] = find(i);
            deg[i - 1]++; deg[i]++;
            chosen.push_back(i);
        }
    }

    printf("%d\n", (int)chosen.size());
    for (int e : chosen) printf("%d\n", e);
    return 0;
}
