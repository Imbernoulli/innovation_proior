// TIER: trivial
// First-fit reference schedule -- exactly the checker's baseline construction.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int P, B;
    if (scanf("%d %d", &P, &B) != 2) return 0;
    vector<long long> cap(P + 1), rem(P + 1);
    for (int p = 1; p <= P; p++) { scanf("%lld", &cap[p]); rem[p] = cap[p]; }
    vector<vector<int>> w(P + 1, vector<int>(B + 1)), v(P + 1, vector<int>(B + 1));
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++) scanf("%d %d", &w[p][j], &v[p][j]);

    vector<int> a(B + 1, 0);
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++)
            if (rem[p] >= w[p][j]) { rem[p] -= w[p][j]; a[j] = p; break; }

    for (int j = 1; j <= B; j++) printf("%d%c", a[j], j == B ? '\n' : ' ');
    return 0;
}
