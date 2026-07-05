// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) scanf("%lld", &w[i]);
    // ignore all conflict edges; equip only the single most valuable site.
    int best = 1;
    for (int i = 2; i <= n; i++) if (w[i] > w[best]) best = i;
    printf("1\n%d\n", best);
    return 0;
}
