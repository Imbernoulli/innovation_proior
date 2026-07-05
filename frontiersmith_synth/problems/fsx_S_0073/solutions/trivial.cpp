// TIER: trivial
// Do-nothing baseline: collapse no chambers. F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t, k;
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("0\n");
    return 0;
}
