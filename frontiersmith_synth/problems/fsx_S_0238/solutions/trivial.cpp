// TIER: trivial
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    // consume the rest is unnecessary; just emit all-OFF baseline
    for (int v = 0; v < n; v++) printf("%d%c", 0, v + 1 == n ? '\n' : ' ');
    if (n == 0) printf("\n");
    return 0;
}
