// TIER: trivial
// Matches the checker's own reference construction exactly: the K=min(n,3) highest
// total-bid bidders, grouped together as a single claimed ring.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int T;
    scanf("%d", &T);
    for (int t = 0; t < T; t++) {
        int n, m;
        scanf("%d %d", &n, &m);
        vector<long long> s(m);
        for (int a = 0; a < m; a++) scanf("%lld", &s[a]);
        vector<long long> total(n, 0);
        for (int i = 0; i < n; i++) {
            for (int a = 0; a < m; a++) {
                long long b;
                scanf("%lld", &b);
                total[i] += b;
            }
        }
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            if (total[a] != total[b]) return total[a] > total[b];
            return a < b;
        });
        int K = min(n, 3);
        printf("1\n%d", K);
        for (int r = 0; r < K; r++) printf(" %d", idx[r]);
        printf("\n");
    }
    return 0;
}
