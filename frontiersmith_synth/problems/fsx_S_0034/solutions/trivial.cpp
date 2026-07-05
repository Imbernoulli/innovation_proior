// TIER: trivial
// All-reverse assignment: emit n zeros. This is exactly the checker's baseline B,
// so it scores the calibration ratio ~0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    // consume the rest is unnecessary; just print n zeros
    for (int v = 0; v < n; v++) printf("%s0", v ? " " : "");
    printf("\n");
    return 0;
}
