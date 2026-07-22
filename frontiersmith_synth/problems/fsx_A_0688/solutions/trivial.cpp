// TIER: trivial
// Do-nothing-clever baseline: among affordable sites, open the single one with
// the LARGEST service rate mu (so it alone can absorb all demand without
// violating capacity), and funnel every demand node to it. Ignores distance
// and load-balancing entirely.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int n, m; ll Bud, QW;
    scanf("%d %d %lld %lld", &n, &m, &Bud, &QW);
    vector<ll> dx(n+1), dy(n+1), dlam(n+1);
    for (int i = 1; i <= n; i++) scanf("%lld %lld %lld", &dx[i], &dy[i], &dlam[i]);
    vector<ll> fx(m+1), fy(m+1), fmu(m+1), fcost(m+1);
    for (int j = 1; j <= m; j++) scanf("%lld %lld %lld %lld", &fx[j], &fy[j], &fmu[j], &fcost[j]);

    int best = -1;
    for (int j = 1; j <= m; j++) {
        if (fcost[j] <= Bud) {
            if (best == -1 || fmu[j] > fmu[best]) best = j;
        }
    }
    if (best == -1) best = 1;   // fallback (should not happen: generator guarantees an affordable hub)

    printf("1\n%d\n", best);
    for (int i = 1; i <= n; i++) printf("%d ", best);
    printf("\n");
    return 0;
}
