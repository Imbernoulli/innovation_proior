// TIER: trivial
// Install a regulator in every zone. Always feasible; equals the baseline cost.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, r;
    scanf("%d %d %d", &n, &m, &r);
    vector<int> c(n + 1);
    for (int v = 1; v <= n; v++) scanf("%d", &c[v]);
    // (edges irrelevant for the trivial cover)
    printf("%d\n", n);
    for (int v = 1; v <= n; v++) printf("%d\n", v);
    return 0;
}
