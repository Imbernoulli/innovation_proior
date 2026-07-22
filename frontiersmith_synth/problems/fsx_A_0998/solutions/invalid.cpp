// TIER: invalid
// Deliberately infeasible: claims 1 zealot but names household id 0, which
// is out of the required [1,n] range -> the checker must reject it (score
// 0) via its bounded output read, regardless of any test's structure.
#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, m, R, tau;
    if (scanf("%d %d %d %d", &n, &m, &R, &tau) != 4) return 0;
    for (int i = 0; i < n; i++){ int p, c, s; scanf("%d %d %d", &p, &c, &s); }
    for (int i = 0; i < m; i++){ int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("1\n0\n");
    return 0;
}
