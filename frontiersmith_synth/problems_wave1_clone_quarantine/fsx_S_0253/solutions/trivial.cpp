// TIER: trivial
// Do-nothing baseline: clear no corridors. Shortest route == original -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t; long long C;
    if (scanf("%d %d %d %d %lld", &n, &m, &s, &t, &C) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v, w, c; scanf("%d %d %d %d", &u, &v, &w, &c); }
    printf("0\n");
    return 0;
}
