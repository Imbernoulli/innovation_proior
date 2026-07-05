// TIER: invalid
// Deliberately infeasible: claims to unplug one link but prints an out-of-range edge index.
// The checker's bounded read (1..m) rejects it, so this scores 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    // read enough to be safe, then emit a bogus index
    int n, m, P, k;
    if (scanf("%d %d %d %d", &n, &m, &P, &k) != 4) { printf("1\n999999999\n"); return 0; }
    printf("1\n%d\n", m + 1000000); // out of range -> infeasible
    return 0;
}
