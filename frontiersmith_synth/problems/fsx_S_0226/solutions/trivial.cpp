// TIER: trivial
// Best uniform coloring = exactly the checker's baseline B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, D;
    if (scanf("%d %d %d", &n, &m, &D) != 3) return 0;
    vector<long long> colorWeight(D, 0);
    for (int j = 0; j < m; j++) {
        int k; scanf("%d", &k);
        vector<int> cols;
        for (int e = 0; e < k; e++) {
            int v, c; scanf("%d %d", &v, &c);
            cols.push_back(c);
        }
        long long w; scanf("%lld", &w);
        sort(cols.begin(), cols.end());
        cols.erase(unique(cols.begin(), cols.end()), cols.end());
        for (int c : cols) colorWeight[c] += w;
    }
    int best = 0;
    for (int c = 1; c < D; c++) if (colorWeight[c] > colorWeight[best]) best = c;
    for (int i = 0; i < n; i++) printf("%d%c", best, i + 1 == n ? '\n' : ' ');
    return 0;
}
