// TIER: trivial
// Do-nothing baseline: submit the identity schedule. Cost F = B => ratio 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, L;
    if (scanf("%d %d", &n, &L) != 2) return 0;
    for (int i = 1; i <= n; i++) {
        int s; scanf("%d", &s);
        for (int j = 0; j < s; j++) { int p; scanf("%d", &p); }
    }
    for (int i = 1; i <= n; i++) printf("%d%c", i, i == n ? '\n' : ' ');
    return 0;
}
