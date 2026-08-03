// TIER: trivial
// Feature only the single highest-hype performer. This is exactly the checker's
// baseline B, so it scores ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    vector<int> w(n + 1);
    int best = 1;
    for (int i = 1; i <= n; i++) {
        scanf("%d", &w[i]);
        if (w[i] > w[best]) best = i;
    }
    // edges are irrelevant to this baseline; a single vertex is always independent.
    printf("1\n%d\n", best);
    return 0;
}
