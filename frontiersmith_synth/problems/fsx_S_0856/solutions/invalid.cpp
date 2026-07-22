// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: an all-raised grid. (1,1,1,1) is always one of
// the forbidden windows by this problem's construction, and every test has
// R >= 2, so this always violates feasibility and must score 0.
int main() {
    int R, C, K;
    scanf("%d %d %d", &R, &C, &K);
    for (int i = 0; i < K; i++) {
        int w, x, y, z;
        scanf("%d %d %d %d", &w, &x, &y, &z);
    }
    string row(C, '1');
    for (int r = 0; r < R; r++) printf("%s\n", row.c_str());
    return 0;
}
