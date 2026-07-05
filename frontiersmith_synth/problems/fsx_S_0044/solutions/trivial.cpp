// TIER: trivial
// The alternating dock: a_i = i % 2 -- exactly the checker's internal baseline.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    for (int i = 0; i < m; i++) { int u, v, w; scanf("%d %d %d", &u, &v, &w); }
    for (int i = 1; i <= n; i++) printf("%d\n", i % 2);
    return 0;
}
