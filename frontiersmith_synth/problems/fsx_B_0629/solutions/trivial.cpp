// TIER: trivial
// One template per wing = that wing's OWN first demand string, at the same fixed
// radius the checker uses for its internal per-wing baseline anchor. This touches
// every wing (so the objective, a minimum, is never zero) but does not adapt the
// radius, does not look for shared motifs, and never revisits a wing -- it mirrors,
// but does not beat, the checker's reference construction.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int K, Mmax, L, Rmax;
    ll Budget;
    if (scanf("%d %d %d %d %lld", &K, &Mmax, &L, &Rmax, &Budget) != 5) return 0;

    vector<string> anchor(K);
    for (int k = 0; k < K; k++) {
        int n;
        scanf("%d", &n);
        static char buf[64];
        for (int i = 0; i < n; i++) {
            scanf("%s", buf);
            if (i == 0) anchor[k] = string(buf);
        }
    }

    int radius = max(1, Rmax / 3);
    printf("%d\n", K);
    for (int k = 0; k < K; k++) printf("%s %d\n", anchor[k].c_str(), radius);
    return 0;
}
