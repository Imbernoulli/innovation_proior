// TIER: trivial
// Reference (id) split: sensors 1..n/2 -> station 0, rest -> station 1.
// This is exactly the checker's internal baseline B, so it scores ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int e = 0; e < m; e++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("%d%c", (i <= n / 2) ? 0 : 1, i == n ? '\n' : ' ');
    return 0;
}
