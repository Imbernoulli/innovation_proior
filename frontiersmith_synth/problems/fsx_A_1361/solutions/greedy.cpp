// TIER: greedy
// "average coder" heuristic: unusually high bidders must be the cartel -- sort by average
// bid and flag the top ~35% as ONE group. Uses raw bid LEVEL, ignoring job size entirely,
// so it is defeated both by legitimately expensive honest bidders and by cover bids that
// were deliberately priced to stay in the plausible range.
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
        sort(idx.begin(), idx.end(), [&](int a, int b) { return total[a] > total[b]; });
        int k = min(n, 5);
        printf("1\n%d", k);
        for (int i = 0; i < k; i++) printf(" %d", idx[i]);
        printf("\n");
    }
    return 0;
}
