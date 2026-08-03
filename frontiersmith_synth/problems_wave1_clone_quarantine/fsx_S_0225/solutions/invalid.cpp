// TIER: invalid
// Deliberately infeasible: emit an out-of-range channel (K+1) -> must score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, K;
    scanf("%d %d %d", &n, &m, &K);
    for (int i = 0; i < m; i++) {
        int u, v, w, d;
        scanf("%d %d %d %d", &u, &v, &w, &d);
    }
    // first token already out of range (channels are 1..K)
    for (int i = 0; i < n; i++) printf("%d%c", K + 1, i + 1 == n ? '\n' : ' ');
    return 0;
}
