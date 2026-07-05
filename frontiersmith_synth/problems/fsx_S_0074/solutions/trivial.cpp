// TIER: trivial
// Install a relay on EVERY habitable sector -> always feasible, cost == baseline B -> ratio ~0.1
#include <bits/stdc++.h>
using namespace std;

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) return 0;
    for (int i = 0; i < H; i++) { char buf[64]; scanf("%s", buf); }
    int n; scanf("%d", &n);
    vector<int> cost(n + 1), rad(n + 1);
    for (int v = 1; v <= n; v++) scanf("%d", &cost[v]);
    for (int v = 1; v <= n; v++) scanf("%d", &rad[v]);

    printf("%d\n", n);
    for (int v = 1; v <= n; v++) printf("%d%c", v, v == n ? '\n' : ' ');
    return 0;
}
