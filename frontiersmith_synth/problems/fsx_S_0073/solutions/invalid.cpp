// TIER: invalid
// Deliberately infeasible: claims to collapse chamber s (the cave mouth), which
// the checker forbids -> must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, s, t, k;
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &k) != 5) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    // print exactly one collapsed chamber = s, which is illegal
    printf("1\n%d\n", s);
    return 0;
}
