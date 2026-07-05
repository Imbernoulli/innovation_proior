// TIER: invalid
// Deliberately infeasible: prints stage value 2 for the first performer, which is
// outside {0,1} -> the checker's bounded read rejects it -> score 0.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m; ll tau;
    if (scanf("%d %d %lld", &n, &m, &tau) != 3) return 0;
    for (int i = 1; i <= n; i++) { int x; scanf("%d", &x); }
    for (int e = 0; e < m; e++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    printf("2\n");                         // out-of-range stage
    for (int i = 2; i <= n; i++) printf("0\n");
    return 0;
}
