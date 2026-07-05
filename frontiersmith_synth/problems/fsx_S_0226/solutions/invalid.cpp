// TIER: invalid
// Deliberately infeasible: emit an out-of-range color (D) -> checker rejects, scores 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, D;
    if (scanf("%d %d %d", &n, &m, &D) != 3) return 0;
    // consume rest of input (not needed, but harmless)
    for (int i = 0; i < n; i++) printf("%d%c", D, i + 1 == n ? '\n' : ' ');
    return 0;
}
