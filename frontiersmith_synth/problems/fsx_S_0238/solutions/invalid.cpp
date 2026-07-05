// TIER: invalid
// Deliberately infeasible: emits out-of-range switch states (2), must score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int v = 0; v < n; v++) printf("%d%c", 2, v + 1 == n ? '\n' : ' ');
    if (n == 0) printf("2\n");
    return 0;
}
