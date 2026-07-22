// TIER: invalid
#include <bits/stdc++.h>
using namespace std;

// Deliberately infeasible: prints a tile index one past the catalog size at
// every cell, which the checker's bounded read ouf.readInt(1,T,...) must reject.
int main() {
    int R, W, T;
    scanf("%d %d %d", &R, &W, &T);
    for (int i = 0; i < T; i++) { int a,b,c,d; scanf("%d %d %d %d", &a,&b,&c,&d); }
    int bad = T + 1;
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < W; j++) printf(j ? " %d" : "%d", bad);
        printf("\n");
    }
    return 0;
}
