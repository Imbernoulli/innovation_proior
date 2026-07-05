// TIER: invalid
// Deliberately infeasible: build zero stations, leaving every demand block unmonitored -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    int D; scanf("%d", &D);
    for (int i = 0; i < D; i++) { int x; scanf("%d", &x); }
    int P; scanf("%d", &P);
    for (int i = 0; i < P; i++) { int a, b, c; scanf("%d %d %d", &a, &b, &c); }
    printf("0\n"); // covers nothing while demand exists -> infeasible
    return 0;
}
