// TIER: invalid
// Deliberately infeasible: barricade EVERY walkway. This blows the barricade budget and
// cuts the Grand Finale off from the front gate, so the checker must score it 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, s, t, P;
    if (scanf("%d %d %d %d %d", &n, &m, &s, &t, &P) != 5) return 0;
    for (int i = 0; i < m; i++) {
        int u, v, w, c;
        scanf("%d %d %d %d", &u, &v, &w, &c);
    }
    printf("%d\n", m);
    for (int i = 1; i <= m; i++) printf("%d\n", i);
    return 0;
}
