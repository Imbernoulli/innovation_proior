// TIER: greedy
// The "obvious" idea: this is a symmetric alternating claiming game, and the
// textbook trick for such games is strategy stealing via mirroring -- respond
// to the opponent's move v by claiming v's natural partner (the other
// endpoint of the FIRST poison edge that touches v). This is exactly the
// achievement-game intuition (an extra move / a mirrored reply never hurts)
// applied blindly to an AVOIDANCE game, where it can hurt: mirroring never
// completes the *base* pair edge it mirrors, but it walks straight into
// whatever *other* poison edges connect the mirror target to vertices this
// player already happens to own, which the generator plants densely on the
// harder test files.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int K;
    scanf("%d", &K);
    for (int a = 0; a < K; a++) {
        int n, m, first;
        scanf("%d %d %d", &n, &m, &first);
        vector<int> pr(n + 1, 0);
        for (int j = 0; j < m; j++) {
            int u, v; scanf("%d %d", &u, &v);
            if (pr[u] == 0) pr[u] = v;
            if (pr[v] == 0) pr[v] = u;
        }
        for (int v = 1; v <= n; v++) printf("%d ", pr[v]);
        printf("\n");
        for (int v = 1; v <= n; v++) printf("%d ", v);   // ascending fallback
        printf("\n");
    }
    return 0;
}
