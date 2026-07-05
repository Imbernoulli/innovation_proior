// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, K;
    if (scanf("%d %d %d", &n, &m, &K) != 3) return 0;
    for (int i = 0; i < n; i++) { int p, s; scanf("%d %d", &p, &s); }
    for (int i = 0; i < m; i++) { int u, v, w, g; scanf("%d %d %d %d", &u, &v, &w, &g); }
    // do-nothing: park every beacon on channel 1 -> exactly the checker's baseline B
    for (int i = 0; i < n; i++) printf("1%c", i + 1 < n ? ' ' : '\n');
    return 0;
}
