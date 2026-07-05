// TIER: invalid
// Deliberately infeasible: installs a single cable, leaving the fabric disconnected
// (unless n==2). The checker rejects a non-spanning fabric -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int v = 1; v <= n; v++) { int d; scanf("%d", &d); }
    for (int i = 1; i <= m; i++) { int u, v; ll w; scanf("%d %d %lld", &u, &v, &w); }
    printf("1\n1\n"); // only one cable -> not spanning for n>=4
    return 0;
}
