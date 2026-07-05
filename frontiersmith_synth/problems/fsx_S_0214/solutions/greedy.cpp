// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<vector<pair<int,int>>> pat(m);
    vector<long long> pw(m);
    for (int j = 0; j < m; j++) {
        int k; scanf("%d", &k);
        pat[j].resize(k);
        for (int e = 0; e < k; e++) { int s, o; scanf("%d %d", &s, &o); pat[j][e] = {s, o}; }
        long long w; scanf("%lld", &w); pw[j] = w;
    }
    vector<int> idx(m);
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&](int a, int b){ return pw[a] > pw[b]; });

    vector<int> assign(n + 1, -1); // -1 = free
    for (int id : idx) {
        bool feasible = true;
        for (auto& pr : pat[id])
            if (assign[pr.first] != -1 && assign[pr.first] != pr.second) { feasible = false; break; }
        if (feasible)
            for (auto& pr : pat[id]) assign[pr.first] = pr.second;
    }
    for (int i = 1; i <= n; i++) {
        int v = assign[i] == -1 ? 0 : assign[i];
        printf("%d%c", v, i == n ? '\n' : ' ');
    }
    if (n == 0) printf("\n");
    return 0;
}
