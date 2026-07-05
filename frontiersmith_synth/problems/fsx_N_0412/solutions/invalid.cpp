// TIER: invalid
// Deliberately infeasible: prints heading value 2 (out of {0,1}) -> must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int v = 1; v <= n; v++) printf("%d%c", 2, v == n ? '\n' : ' ');
    return 0;
}
