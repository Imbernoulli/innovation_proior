// TIER: greedy
// The obvious one-pass recipe: MAXIMIZE seam harmony only, ignore signature
// diversity. Find the best color pair (u,v) and make every band the period-1 tile
// (rows v,v,u). Every seam then scores M[u][v] (the max), but all bands share one
// signature -> Groups = 1, so F = 1*Seam + 1. Misses the Groups multiplier.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int B, h, P, Q;
    if (scanf("%d %d %d %d", &B, &h, &P, &Q) != 4) return 0;
    vector<vector<int>> M(P, vector<int>(P));
    for (int i = 0; i < P; i++) for (int j = 0; j < P; j++) scanf("%d", &M[i][j]);
    int bu = 0, bv = 0, best = -1;
    for (int u = 0; u < P; u++) for (int v = 0; v < P; v++)
        if (M[u][v] > best) { best = M[u][v]; bu = u; bv = v; }
    // tile rows: top=bv, mid=bv, bottom=bu  -> seam(bottom bu, next top bv)=M[bu][bv]
    for (int b = 0; b < B; b++) {
        printf("1\n");
        printf("%d\n%d\n%d\n", bv, bv, bu);
    }
    return 0;
}
