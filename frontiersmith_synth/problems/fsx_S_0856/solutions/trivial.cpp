// TIER: trivial
#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, K;
    scanf("%d %d %d", &R, &C, &K);
    for (int i = 0; i < K; i++) {
        int w, x, y, z;
        scanf("%d %d %d %d", &w, &x, &y, &z);
    }
    // Always-legal static row: column 0 raised, all other columns lowered.
    // (0,0,0,0) and (1,0,1,0) windows are never forbidden by construction,
    // and this row only ever produces those two window shapes against itself.
    string row(C, '0');
    row[0] = '1';
    for (int r = 0; r < R; r++) {
        printf("%s\n", row.c_str());
    }
    return 0;
}
