// TIER: trivial
// All acts on the Side Stage (all-zero) -- exactly the checker's baseline -> ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int c = 0; c < m; c++) {
        int k, w;
        scanf("%d %d", &k, &w);
        for (int t = 0; t < k; t++) { int L; scanf("%d", &L); }
    }
    for (int i = 0; i < n; i++) printf("0%c", i + 1 == n ? '\n' : ' ');
    return 0;
}
