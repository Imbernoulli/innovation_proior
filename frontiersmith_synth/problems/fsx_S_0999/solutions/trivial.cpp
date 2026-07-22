// TIER: trivial
// Do-nothing baseline: read the edges in input order, send the first capDay of them to the
// day shift, the rest to the night shift. Exactly the checker's own internal baseline B.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    double tau;
    scanf("%lf", &tau);
    long long capDay, capNight;
    scanf("%lld %lld", &capDay, &capNight);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        scanf("%d %d %d", &u, &v, &w);
        if (i < capDay) printf("0\n"); else printf("1\n");
    }
    return 0;
}
